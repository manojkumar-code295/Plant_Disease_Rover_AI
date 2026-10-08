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
        return f"{GREEN}IN1(Pin 11)=HIGH, IN2(Pin 13)=LOW | IN3(Pin 15)=LOW, IN4(Pin 16)=HIGH | ENA(Pin 32)={spd}% ENB(Pin 33)={spd}%{RESET}"
    elif d in ("backward", "reverse"):
        return f"{YELLOW}IN1(Pin 11)=LOW, IN2(Pin 13)=HIGH | IN3(Pin 15)=HIGH, IN4(Pin 16)=LOW | ENA(Pin 32)={spd}% ENB(Pin 33)={spd}%{RESET}"
    elif d == "left":
        return f"{CYAN}IN1(Pin 11)=LOW, IN2(Pin 13)=HIGH | IN3(Pin 15)=LOW, IN4(Pin 16)=HIGH | ENA(Pin 32)={spd}% ENB(Pin 33)={spd}%{RESET}"
    elif d == "right":
        return f"{CYAN}IN1(Pin 11)=HIGH, IN2(Pin 13)=LOW | IN3(Pin 15)=HIGH, IN4(Pin 16)=LOW | ENA(Pin 32)={spd}% ENB(Pin 33)={spd}%{RESET}"
    else:
        return f"{RED}ALL PINS LOW (0V) - MOTORS SAFELY STOPPED{RESET}"

def main():
    os.system("clear")
    print_banner()

    last_command_id = -1
    total_signals = 0
    last_pump = None
    heartbeat_tick = 0

    while True:
        try:
            req = urllib.request.Request(LOCAL_API, headers={"User-Agent": "PhoneSignalReceiver/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    rover = data.get("rover", {})
                    cid = rover.get("command_id", 0)
                    cmd = rover.get("last_command", "stop")
                    direction = rover.get("direction", "Stopped")
                    speed = int(rover.get("speed", 0))
                    pump = bool(rover.get("pump", False))
                    connected = bool(rover.get("connected", False))

                    # Check for pump changes
                    if last_pump is not None and pump != last_pump:
                        now_str = time.strftime("%H:%M:%S")
                        total_signals += 1
                        print(f"\a", end="", flush=True) # Terminal bell
                        print(f"{CYAN}================================================================================")
                        print(f"{BOLD}[{now_str}] 💦 SPRAY PUMP: {'ON (SPRAYING)' if pump else 'OFF (STOPPED)'}{RESET}")
                        print(f"{CYAN}================================================================================{RESET}")
                        print(f"   Status:       {GREEN if pump else RED}{'SPRAY PUMP RUNNING (12V ACTIVE)' if pump else 'SPRAY PUMP STOPPED'}{RESET}")
                        print(f"   GPIO Pin:     Pin 37 (GPIO 26) -> {'HIGH' if pump else 'LOW'}")
                        print(f"{CYAN}================================================================================{RESET}\n")

                    last_pump = pump

                    # Check for new rover command from phone
                    if cid != last_command_id and cid > 0:
                        last_command_id = cid
                        total_signals += 1
                        now_str = time.strftime("%H:%M:%S")
                        cmd_upper = cmd.upper()
                        is_stop = cmd == "stop" or direction.lower() == "stopped"

                        print(f"\a", end="", flush=True) # Terminal bell

                        color = RED if is_stop else GREEN
                        symbol = "🛑" if is_stop else "⚡"
                        print(f"{color}{BOLD}================================================================================")
                        print(f" {symbol} [{now_str}] DIRECTION: {cmd_upper}")
                        print(f"================================================================================{RESET}")
                        print(f"   Action:       {BOLD}{WHITE}{'STOPPED' if is_stop else f'MOVING {cmd_upper}'}{RESET}")
                        print(f"   Speed:        {BOLD}{CYAN}{speed}% PWM{RESET}")
                        print(f"   Motor Driver: {get_gpio_desc(direction, speed, not is_stop)}")
                        print(f"   Pump Relay:   {MAGENTA}{'SPRAYING (ON)' if pump else 'OFF'}{RESET}")
                        print(f"   Motor Status: {GREEN}{'BRAKE APPLIED' if is_stop else 'RUNNING'}{RESET}")
                        print(f"{color}================================================================================{RESET}\n")

        except Exception as e:
            pass

        heartbeat_tick += 1
        if heartbeat_tick % 15 == 0:
            now_str = time.strftime("%H:%M:%S")
            sys.stdout.write(f"\r{DIM}[{now_str}] 🟢 Ready | Current: {direction.upper()} | Speed: {speed}%{RESET}")
            sys.stdout.flush()

        time.sleep(0.15)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Signal Receiver stopped by user.{RESET}")
