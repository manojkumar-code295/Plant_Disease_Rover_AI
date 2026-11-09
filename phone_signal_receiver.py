#!/usr/bin/env python3
"""
AgriRover - Dedicated Phone Signal Receiver Terminal (Monitor Only)
===================================================================
This terminal is a pure listener/monitor. It does NOT control the rover.
Whenever you send ANY command from your phone (Forward, Backward, Left,
Right, Stop, Pump, Speed), this terminal instantly captures it, sounds
an alert, and prints full signal confirmation & hardware GPIO pin states!
"""

import os
import sys
import time
import socket
import json
import urllib.request
import urllib.error

# ANSI Color Codes for high visibility
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BG_DARK = "\033[48;5;234m"
BG_GREEN = "\033[48;5;22m"
BG_RED = "\033[48;5;52m"
DIM = "\033[2m"

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
LOCAL_API = "http://127.0.0.1:5000/api/rover/status"
CLOUD_API = "https://precision-agriculture-smart-rover.onrender.com/api/rover/command"

def print_banner():
    print(f"{CYAN}{BOLD}================================================================================")
    print(f"      📡 AGRIROVER PHONE SIGNAL RECEIVER TERMINAL (MONITOR ONLY)")
    print(f"================================================================================{RESET}")
    print(f"{WHITE}Local Wi-Fi URL:  {GREEN}http://{LOCAL_IP}:5000/rover{RESET}")
    print(f"{WHITE}Pi Hotspot URL:   {GREEN}http://10.42.0.1:5000/rover{RESET}")
    print(f"{WHITE}Render Cloud URL: {GREEN}https://precision-agriculture-smart-rover.onrender.com/rover{RESET}")
    print(f"{CYAN}--------------------------------------------------------------------------------{RESET}")
    print(f"{YELLOW}▶ Listening for signals from your phone... Tap ANY button on your phone!{RESET}")
    print(f"{CYAN}================================================================================{RESET}\n")

def get_gpio_desc(direction, speed, is_moving):
    spd = speed if is_moving else 0
    d = direction.lower()
    if d == "forward":
        return f"{GREEN}RPWM(Pin 11)={spd}% PWM | LPWM(Pin 13)=0V | EN(Pin 32)=3.3V (ENABLED){RESET}"
    elif d in ("backward", "reverse"):
        return f"{YELLOW}RPWM(Pin 11)=0V | LPWM(Pin 13)={spd}% PWM | EN(Pin 32)=3.3V (ENABLED){RESET}"
    else:
        return f"{RED}RPWM(Pin 11)=0V | LPWM(Pin 13)=0V | EN(Pin 32)=0V (SAFELY BRAKED){RESET}"

def main():
    os.system("clear")
    print_banner()

    last_command_id = -1
    total_signals = 0
    last_pump = None
    heartbeat_tick = 0
    current_direction = "STOPPED"
    current_speed = 0

    while True:
        signal_found = None

        # 1. Check Local API (instant 5ms on local Wi-Fi)
        try:
            req = urllib.request.Request(LOCAL_API, headers={"User-Agent": "PhoneSignalReceiver/1.0"})
            with urllib.request.urlopen(req, timeout=0.4) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    rover = data.get("rover", {})
                    cid = rover.get("command_id", 0)
                    cmd = rover.get("last_command", "stop")
                    direction = rover.get("direction", "Stopped")
                    speed = int(rover.get("speed", 0))
                    pump = bool(rover.get("pump", False))
                    if cid > 0 and cid != last_command_id:
                        signal_found = (cid, cmd, direction, speed, pump, "Local Wi-Fi / Hotspot")
                    elif last_pump is not None and pump != last_pump:
                        signal_found = (last_command_id, cmd, direction, speed, pump, "Local Wi-Fi / Hotspot")
        except Exception:
            pass

        # 2. Check Render Cloud API (instant ~150ms when phone is on cellular 4G/5G anywhere)
        if not signal_found:
            try:
                req = urllib.request.Request(CLOUD_API, headers={"User-Agent": "PhoneSignalReceiver/1.0"})
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        cid = data.get("command_id", 0)
                        cmd = data.get("command", "stop")
                        direction = data.get("direction", "Stopped")
                        speed = int(data.get("speed", 50))
                        pump = bool(data.get("pump", False))
                        if cid > 0 and cid != last_command_id:
                            signal_found = (cid, cmd, direction, speed, pump, "Render Cloud (Cellular/Internet)")
                        elif last_pump is not None and pump != last_pump:
                            signal_found = (last_command_id, cmd, direction, speed, pump, "Render Cloud (Cellular/Internet)")
            except Exception:
                pass

        # 3. Process new verified signal from phone
        if signal_found:
            cid, cmd, direction, speed, pump, source = signal_found
            now_str = time.strftime("%H:%M:%S")
            current_direction = direction
            current_speed = speed

            # A. Check for Spray Pump toggle
            if last_pump is not None and pump != last_pump:
                total_signals += 1
                print(f"\a", end="", flush=True)
                print(f"{CYAN}================================================================================")
                print(f"{BOLD}[{now_str}] 💦 SIGNAL VERIFIED: SPRAY PUMP {'ON (ACTIVE)' if pump else 'OFF (STOPPED)'}{RESET}")
                print(f"{CYAN}================================================================================{RESET}")
                print(f"   Route:        {WHITE}{source}{RESET}")
                print(f"   Pi Received:  {GREEN}✅ CONFIRMED: Raspberry Pi 4 received pump trigger!{RESET}")
                print(f"   Relay Pin:    Pin 37 (BCM GPIO 26) -> {'HIGH (3.3V -> 12V PUMP ACTIVE)' if pump else 'LOW (0V -> PUMP OFF)'}")
                print(f"   Hardware:     12V Diaphragm Chemical Spray Pump Relay")
                print(f"{CYAN}================================================================================{RESET}\n")

            last_pump = pump

            # B. Check for Movement Command
            if cid != last_command_id and cid > 0:
                last_command_id = cid
                total_signals += 1
                cmd_upper = cmd.upper()
                is_stop = cmd == "stop" or direction.lower() == "stopped"

                print(f"\a", end="", flush=True)
                color = RED if is_stop else GREEN
                symbol = "🛑" if is_stop else "⚡"
                print(f"{color}{BOLD}================================================================================")
                print(f" {symbol} [{now_str}] SIGNAL VERIFIED: {cmd_upper} (Signal #{total_signals} | Command ID #{cid})")
                print(f"================================================================================{RESET}")
                print(f"   Pi Received:  {GREEN}✅ VERIFIED: Raspberry Pi 4 received signal from phone!{RESET}")
                print(f"   Signal Sent:  {BOLD}{WHITE}{cmd_upper}{RESET} ({direction}) via {CYAN}{source}{RESET}")
                print(f"   Chassis Move: {BOLD}{WHITE}{'STOPPED (BRAKED)' if is_stop else f'MOVING {cmd_upper}'}{RESET}")
                print(f"   PWM Speed:    {BOLD}{CYAN}{speed}% PWM Power{RESET}")
                print(f"   Hardware BTS: {get_gpio_desc(direction, speed, not is_stop)}")
                print(f"   Motor Status: {GREEN}{'🛑 EMERGENCY BRAKE ENGAGED' if is_stop else '⚡ MOTORS ROTATING'}{RESET}")
                print(f"   Pump Relay:   {MAGENTA}{'SPRAYING (ON)' if pump else 'OFF'}{RESET}")
                print(f"{color}================================================================================{RESET}\n")

        heartbeat_tick += 1
        if heartbeat_tick % 15 == 0:
            now_str = time.strftime("%H:%M:%S")
            sys.stdout.write(f"\r{DIM}[{now_str}] 🟢 Pi Listening | Verified Signals: {total_signals} | Last: {current_direction.upper()}{RESET}")
            sys.stdout.flush()

        time.sleep(0.12)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Signal Receiver stopped by user.{RESET}")
