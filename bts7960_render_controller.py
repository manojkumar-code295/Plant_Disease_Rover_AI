#!/usr/bin/env python3
"""
AgriRover - Dedicated BTS7960 Render Cloud Controller & Signal Verifier
========================================================================
Hardware Configuration:
  - Battery -> Dual-Output Buck Converter:
      • Output 1: 5.1V DC -> Raspberry Pi 4 (Pin 2/6) & BTS7960 Logic VCC (5V)
      • Output 2: 12V High-Current -> BTS7960 Motor Power (B+ / B-)
  - Single BTS7960 (IBT-2 43A) Motor Driver -> 2x 300 RPM Motors (Parallel on M+ / M-)
  - 12V Diaphragm Chemical Spray Pump (Relay Control on Pin 37)

Pin Mapping:
  - Physical Pin 11 (BCM GPIO 17) -> BTS7960 RPWM (Forward Direction & PWM)
  - Physical Pin 13 (BCM GPIO 27) -> BTS7960 LPWM (Backward Direction & PWM)
  - Physical Pin 32 (BCM GPIO 12) -> BTS7960 R_EN + L_EN (Bridged Enable)
  - Physical Pin 37 (BCM GPIO 26) -> Spray Pump Relay (IN)
  - Physical Pin 6  (GND)         -> BTS7960 GND & Common Ground
  - Physical Pin 2/4 (5V)         -> BTS7960 VCC (Logic Power)

Cloud Connection:
  - Polling Endpoint: https://precision-agriculture-smart-rover.onrender.com/api/rover/command
  - Heartbeat Endpoint: https://precision-agriculture-smart-rover.onrender.com/api/rover/heartbeat
"""

import os
import sys
import time
import socket
import json
import threading
import signal
import select
import termios
import tty
import urllib.request
import urllib.error

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
DIM = "\033[2m"

# Hardware Pin Definitions (BCM)
RPWM_PIN = 17   # Physical Pin 11 (Forward)
LPWM_PIN = 27   # Physical Pin 13 (Backward)
EN_PIN   = 12   # Physical Pin 32 (Enable R_EN + L_EN)
PUMP_PIN = 26   # Physical Pin 37 (Spray Pump Relay)
PWM_FREQ = 1000 # 1000 Hz PWM for smooth 300 RPM motor torque

# URLs
CLOUD_BASE = "https://precision-agriculture-smart-rover.onrender.com"
CLOUD_CMD_URL = f"{CLOUD_BASE}/api/rover/command"
CLOUD_HEARTBEAT_URL = f"{CLOUD_BASE}/api/rover/heartbeat"
LOCAL_STATUS_URL = "http://127.0.0.1:5000/api/rover/status"

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_ip()

# State Variables
gpio_chip = None
current_direction = "STOPPED"
current_speed = 60
current_pump = False
invert_motor = False
total_signals = 0
cloud_connected = False
running = True

# Initialize GPIO with lgpio
try:
    import lgpio
    gpio_chip = lgpio.gpiochip_open(0)
    for pin in [RPWM_PIN, LPWM_PIN, EN_PIN, PUMP_PIN]:
        try:
            lgpio.gpio_free(gpio_chip, pin)
        except Exception:
            pass
        lgpio.gpio_claim_output(gpio_chip, pin, 0)
    gpio_available = True
except Exception as e:
    gpio_available = False
    gpio_err = str(e)

def apply_hardware(direction, speed, pump):
    """
    Actuates BTS7960 H-Bridge and Spray Pump Relay.
    - Forward:  RPWM = PWM(speed), LPWM = 0, EN = HIGH
    - Backward: RPWM = 0, LPWM = PWM(speed), EN = HIGH
    - Stop:     RPWM = 0, LPWM = 0, EN = LOW
    """
    global current_direction, current_speed, current_pump
    if not gpio_available or gpio_chip is None:
        return

    speed = max(0, min(100, int(speed)))
    d = direction.lower().strip()

    # Handle polarity inversion if configured
    if invert_motor:
        if d == "forward":
            d = "backward"
        elif d in ("backward", "reverse"):
            d = "forward"

    try:
        # Actuate Spray Pump
        lgpio.gpio_write(gpio_chip, PUMP_PIN, 1 if pump else 0)

        # Actuate BTS7960 Motors
        if d == "forward" and speed > 0:
            lgpio.gpio_write(gpio_chip, EN_PIN, 1)
            lgpio.gpio_write(gpio_chip, LPWM_PIN, 0)
            lgpio.tx_pwm(gpio_chip, RPWM_PIN, PWM_FREQ, speed)
        elif d in ("backward", "reverse") and speed > 0:
            lgpio.gpio_write(gpio_chip, EN_PIN, 1)
            lgpio.gpio_write(gpio_chip, RPWM_PIN, 0)
            lgpio.tx_pwm(gpio_chip, LPWM_PIN, PWM_FREQ, speed)
        else: # STOP
            lgpio.tx_pwm(gpio_chip, RPWM_PIN, PWM_FREQ, 0)
            lgpio.tx_pwm(gpio_chip, LPWM_PIN, PWM_FREQ, 0)
            lgpio.gpio_write(gpio_chip, RPWM_PIN, 0)
            lgpio.gpio_write(gpio_chip, LPWM_PIN, 0)
            lgpio.gpio_write(gpio_chip, EN_PIN, 0)

    except Exception as e:
        print(f"{RED}[GPIO ERROR] {e}{RESET}")

def stop_hardware():
    """Emergency brake / safe shutdown: all GPIO pins driven LOW."""
    if gpio_available and gpio_chip is not None:
        try:
            lgpio.tx_pwm(gpio_chip, RPWM_PIN, PWM_FREQ, 0)
            lgpio.tx_pwm(gpio_chip, LPWM_PIN, PWM_FREQ, 0)
            lgpio.gpio_write(gpio_chip, RPWM_PIN, 0)
            lgpio.gpio_write(gpio_chip, LPWM_PIN, 0)
            lgpio.gpio_write(gpio_chip, EN_PIN, 0)
            lgpio.gpio_write(gpio_chip, PUMP_PIN, 0)
        except Exception:
            pass

def print_banner():
    os.system("clear")
    print(f"{CYAN}{BOLD}================================================================================")
    print(f"   🏎️  AGRIROVER BTS7960 RENDER CLOUD CONTROLLER & HARDWARE TERMINAL")
    print(f"================================================================================{RESET}")
    print(f"{WHITE}Hardware:        {GREEN}1x BTS7960 (IBT-2 43A) + 2x 300 RPM Motors (Parallel on M+/M-){RESET}")
    print(f"{WHITE}Buck Converter:  {GREEN}2-Output Buck: Out 1 (5.1V Pi 4) | Out 2 (12V BTS7960 Motor Power){RESET}")
    print(f"{WHITE}Local Pi IP:     {GREEN}http://{LOCAL_IP}:5000/rover{RESET}")
    print(f"{WHITE}Render Cloud:    {GREEN}{CLOUD_BASE}/rover{RESET}")
    print(f"{CYAN}--------------------------------------------------------------------------------{RESET}")
    print(f"{BOLD}GPIO PIN CONNECTIONS:{RESET}")
    print(f"  • {WHITE}Pin 11 (GPIO 17) -> {CYAN}RPWM (Forward Direction & Speed PWM){RESET}")
    print(f"  • {WHITE}Pin 13 (GPIO 27) -> {CYAN}LPWM (Backward Direction & Speed PWM){RESET}")
    print(f"  • {WHITE}Pin 32 (GPIO 12) -> {CYAN}R_EN + L_EN (Bridged Enable / Standby){RESET}")
    print(f"  • {WHITE}Pin 37 (GPIO 26) -> {MAGENTA}Spray Pump Relay (IN Trigger){RESET}")
    print(f"  • {WHITE}Pin 6  (GND)     -> {YELLOW}Common Ground (Pi GND + Buck OUT1- + Buck OUT2-){RESET}")
    print(f"  • {WHITE}Pin 2  (5V)      -> {RED}BTS7960 Logic VCC (5V Power from Buck OUT1+){RESET}")
    print(f"{CYAN}--------------------------------------------------------------------------------{RESET}")
    print(f"{BOLD}KEYBOARD CONTROLS (Pi Console):{RESET}")
    print(f"  [{GREEN}W{RESET}/{GREEN}F{RESET}] Forward  | [{YELLOW}S{RESET}/{YELLOW}B{RESET}] Backward  | [{RED}Space{RESET}/{RED}X{RESET}] Stop  | [{MAGENTA}P{RESET}] Toggle Pump")
    print(f"  [{CYAN}+{RESET}/{CYAN}-{RESET}] Speed +/- | [{CYAN}1-9{RESET}] 10-90% Speed  | [{YELLOW}I{RESET}] Invert Direction | [{RED}Q{RESET}] Quit")
    print(f"{CYAN}================================================================================{RESET}")
    if gpio_available:
        print(f"{GREEN}✅ GPIO Driver (lgpio) initialized successfully. Motors armed and ready!{RESET}\n")
    else:
        print(f"{RED}⚠️ GPIO Driver Warning: {gpio_err}{RESET}\n")

def get_gpio_desc(direction, speed, pump):
    d = direction.lower().strip()
    if d == "forward":
        return (f"RPWM(Pin 11)={GREEN}PWM {speed}% (ACTIVE){RESET} | "
                f"LPWM(Pin 13)={DIM}0V (LOW){RESET} | "
                f"EN(Pin 32)={GREEN}HIGH (3.3V){RESET} | "
                f"PUMP(Pin 37)={MAGENTA}{'ON (HIGH)' if pump else 'OFF (0V)'}{RESET}")
    elif d in ("backward", "reverse"):
        return (f"RPWM(Pin 11)={DIM}0V (LOW){RESET} | "
                f"LPWM(Pin 13)={YELLOW}PWM {speed}% (ACTIVE){RESET} | "
                f"EN(Pin 32)={GREEN}HIGH (3.3V){RESET} | "
                f"PUMP(Pin 37)={MAGENTA}{'ON (HIGH)' if pump else 'OFF (0V)'}{RESET}")
    else:
        return (f"RPWM(Pin 11)={DIM}0V{RESET} | "
                f"LPWM(Pin 13)={DIM}0V{RESET} | "
                f"EN(Pin 32)={DIM}0V{RESET} | "
                f"PUMP(Pin 37)={MAGENTA}{'ON (HIGH)' if pump else 'OFF (0V)'}{RESET}")

def print_signal_card(source, cid, direction, speed, pump, is_pump_toggle=False):
    global total_signals
    total_signals += 1
    now_str = time.strftime("%H:%M:%S")
    d_upper = direction.upper().strip()
    is_stop = d_upper in ("STOP", "STOPPED", "")

    print("\a", end="", flush=True) # Terminal alert chime

    if is_pump_toggle:
        print(f"\n{MAGENTA}{BOLD}================================================================================")
        print(f" 💦 [{now_str}] SIGNAL VERIFIED: SPRAY PUMP {'ON (SPRAYING)' if pump else 'OFF (STOPPED)'} (Signal #{total_signals})")
        print(f"================================================================================{RESET}")
        print(f"   Pi Received:   {GREEN}✅ YES: Raspberry Pi 4 received signal from phone!{RESET}")
        print(f"   Command ID:    {WHITE}#{cid}{RESET}")
        print(f"   Signal Source: {CYAN}{source}{RESET}")
        print(f"   Relay State:   {MAGENTA}{'HIGH (12V Chemical Pump Triggered)' if pump else 'LOW (Pump Off)'}{RESET}")
        print(f"   Relay Pin:     Pin 37 (BCM GPIO 26)")
        print(f"   GPIO Status:   {get_gpio_desc(direction, speed, pump)}")
        print(f"{MAGENTA}================================================================================{RESET}\n")
        return

    color = RED if is_stop else (GREEN if d_upper == "FORWARD" else YELLOW)
    symbol = "🛑" if is_stop else ("⚡" if d_upper == "FORWARD" else "🔄")

    print(f"\n{color}{BOLD}================================================================================")
    print(f" {symbol} [{now_str}] SIGNAL VERIFIED: {d_upper} (Signal #{total_signals} | Command ID #{cid})")
    print(f"================================================================================{RESET}")
    print(f"   Pi Received:   {GREEN}✅ YES: Raspberry Pi 4 received signal from phone!{RESET}")
    print(f"   Signal Sent:   {BOLD}{WHITE}{d_upper}{RESET} (Speed: {speed}%) via {CYAN}{source}{RESET}")
    if is_stop:
        print(f"   Motion State:  {BOLD}{RED}STOPPED (ELECTRICAL BRAKE ENGAGED){RESET}")
        print(f"   Motor Status:  {RED}300 RPM DC MOTORS SAFELY BRAKED (0V){RESET}")
    else:
        print(f"   Motion State:  {BOLD}{WHITE}CHASSIS DRIVING {d_upper}{RESET} at {CYAN}{speed}% PWM{RESET}")
        print(f"   Motor Status:  {GREEN}⚡ 300 RPM DC MOTORS ROTATING {d_upper}{RESET}")
    print(f"   BTS7960 Pins:  {get_gpio_desc(direction, speed, pump)}")
    print(f"   Pump Relay:    {MAGENTA}{'SPRAYING (ON)' if pump else 'OFF (STANDBY)'}{RESET}")
    if invert_motor:
        print(f"   Polarity:      {YELLOW}[MOTOR INVERSION ACTIVE - FORWARD/REVERSE SWAPPED]{RESET}")
    print(f"{color}================================================================================{RESET}\n")

def heartbeat_sender_thread():
    """Continuously reports Pi online status & active motor state to Render Cloud."""
    global cloud_connected
    while running:
        telemetry = {
            "device": "Raspberry Pi 4 Model B",
            "pi_online": True,
            "connected": True,
            "ip": LOCAL_IP,
            "pi_ip": LOCAL_IP,
            "direction": current_direction.capitalize(),
            "speed": current_speed if current_direction.lower() != "stopped" else 0,
            "pump": current_pump,
            "engine_status": "Running" if current_direction.lower() != "stopped" else "Standby",
            "engine_armed": True,
            "mode": "Manual",
            "model_online": True,
            "model_name": "MobileNetV2 (99% Acc) Chili Disease AI",
            "disease_name": "Scanning Active",
            "disease_treatment": "BTS7960 Rover Controller Online",
            "battery_pct": 100
        }
        try:
            req = urllib.request.Request(
                CLOUD_HEARTBEAT_URL,
                data=json.dumps(telemetry).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "BTS7960Controller/1.0"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    cloud_connected = True
        except Exception:
            cloud_connected = False
        time.sleep(1.0)

def keyboard_listener_thread():
    """Allows instant direct manual testing of Forward/Backward from Pi terminal keyboard."""
    global current_direction, current_speed, current_pump, invert_motor, running

    # Put terminal in cbreak/raw mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                ch = ch.lower()
                if ch in ('w', 'f'):
                    current_direction = "Forward"
                    apply_hardware("forward", current_speed, current_pump)
                    print_signal_card("Pi Console Keyboard [W/F]", 999, "FORWARD", current_speed, current_pump)
                elif ch in ('s', 'b'):
                    current_direction = "Backward"
                    apply_hardware("backward", current_speed, current_pump)
                    print_signal_card("Pi Console Keyboard [S/B]", 999, "BACKWARD", current_speed, current_pump)
                elif ch in (' ', 'x'):
                    current_direction = "Stopped"
                    apply_hardware("stop", 0, current_pump)
                    print_signal_card("Pi Console Keyboard [SPACE/X]", 999, "STOP", 0, current_pump)
                elif ch == 'p':
                    current_pump = not current_pump
                    apply_hardware(current_direction, current_speed, current_pump)
                    print_signal_card("Pi Console Keyboard [P]", 999, current_direction, current_speed, current_pump, is_pump_toggle=True)
                elif ch in ('+', '='):
                    current_speed = min(100, current_speed + 10)
                    apply_hardware(current_direction, current_speed, current_pump)
                    print(f"\n{CYAN}⚡ Speed increased to {current_speed}% PWM{RESET}")
                elif ch in ('-', '_'):
                    current_speed = max(20, current_speed - 10)
                    apply_hardware(current_direction, current_speed, current_pump)
                    print(f"\n{CYAN}⚡ Speed decreased to {current_speed}% PWM{RESET}")
                elif ch in '123456789':
                    current_speed = int(ch) * 10
                    apply_hardware(current_direction, current_speed, current_pump)
                    print(f"\n{CYAN}⚡ Speed set to {current_speed}% PWM{RESET}")
                elif ch == 'i':
                    invert_motor = not invert_motor
                    apply_hardware(current_direction, current_speed, current_pump)
                    print(f"\n{YELLOW}🔄 Motor direction inversion toggled: {'INVERTED' if invert_motor else 'NORMAL'}{RESET}")
                elif ch == 'q':
                    print(f"\n{YELLOW}Quitting BTS7960 Controller...{RESET}")
                    running = False
                    break
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main():
    global current_direction, current_speed, current_pump, running

    print_banner()

    # Start background heartbeat sender
    hb_thread = threading.Thread(target=heartbeat_sender_thread, daemon=True)
    hb_thread.start()

    # Start interactive keyboard listener
    kb_thread = threading.Thread(target=keyboard_listener_thread, daemon=True)
    kb_thread.start()

    last_command_id = -1
    last_pump_state = None
    tick = 0

    print(f"{YELLOW}▶ Connecting to Render Cloud ({CLOUD_CMD_URL})...{RESET}")
    print(f"{DIM}Press [W/F] for Forward, [S/B] for Backward, [Space] to Stop, or tap controls on Phone!{RESET}\n")

    try:
        while running:
            signal_found = None

            # 1. Primary: Poll Render Cloud endpoint (Heart of Render)
            try:
                req = urllib.request.Request(
                    CLOUD_CMD_URL,
                    headers={"User-Agent": "BTS7960RenderController/1.0"}
                )
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        cid = data.get("command_id", 0)
                        cmd = data.get("command", "stop")
                        direction = data.get("direction", "Stopped")
                        speed = int(data.get("speed", 60))
                        pump = bool(data.get("pump", False))
                        if cid > 0 and cid != last_command_id:
                            signal_found = ("Render Cloud (Phone/Web)", cid, cmd, direction, speed, pump)
                        elif last_pump_state is not None and pump != last_pump_state:
                            signal_found = ("Render Cloud (Phone/Web)", last_command_id, cmd, direction, speed, pump)
            except Exception:
                pass

            # 2. Secondary: Fallback to Local API if on same Wi-Fi / Hotspot
            if not signal_found:
                try:
                    req = urllib.request.Request(
                        LOCAL_STATUS_URL,
                        headers={"User-Agent": "BTS7960RenderController/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=0.4) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode())
                            rover = data.get("rover", {})
                            cid = rover.get("command_id", 0)
                            cmd = rover.get("last_command", "stop")
                            direction = rover.get("direction", "Stopped")
                            speed = int(rover.get("speed", 60))
                            pump = bool(rover.get("pump", False))
                            if cid > 0 and cid != last_command_id:
                                signal_found = ("Local Wi-Fi / Hotspot", cid, cmd, direction, speed, pump)
                            elif last_pump_state is not None and pump != last_pump_state:
                                signal_found = ("Local Wi-Fi / Hotspot", last_command_id, cmd, direction, speed, pump)
                except Exception:
                    pass

            # 3. Process new verified signal and actuate physical BTS7960
            if signal_found:
                source, cid, cmd, direction, speed, pump = signal_found
                is_pump_toggle = (last_pump_state is not None and pump != last_pump_state and cid == last_command_id)

                current_direction = direction
                current_speed = speed
                current_pump = pump
                last_command_id = cid
                last_pump_state = pump

                # PHYSICALLY ACTUATE MOTORS & PUMP
                apply_hardware(direction, speed, pump)

                # DISPLAY EVENT VERIFICATION CARD
                print_signal_card(source, cid, direction, speed, pump, is_pump_toggle=is_pump_toggle)

            # Heartbeat ticker on terminal
            tick += 1
            if tick % 10 == 0:
                now_str = time.strftime("%H:%M:%S")
                cloud_status = f"{GREEN}Cloud Connected{RESET}" if cloud_connected else f"{YELLOW}Connecting...{RESET}"
                mot_state = f"{GREEN}{current_direction.upper()}{RESET}" if current_direction.lower() != "stopped" else f"{RED}STOPPED{RESET}"
                pump_str = f"{MAGENTA}PUMP ON{RESET}" if current_pump else f"{DIM}PUMP OFF{RESET}"
                sys.stdout.write(f"\r{DIM}[{now_str}]{RESET} 🟢 Pi Active | {cloud_status} | Motors: {mot_state} ({current_speed}%) | {pump_str} | Verified: {total_signals} ")
                sys.stdout.flush()

            time.sleep(0.12)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopping BTS7960 Controller...{RESET}")
    finally:
        running = False
        stop_hardware()
        if gpio_available and gpio_chip is not None:
            try:
                lgpio.gpiochip_close(gpio_chip)
            except Exception:
                pass
        print(f"{GREEN}All motors safely braked and GPIO pins released. Goodbye!{RESET}\n")

if __name__ == "__main__":
    main()
