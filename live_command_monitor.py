#!/usr/bin/env python3
"""
AgriRover Live Command & AI Telemetry Monitor
Streams real-time control inputs received from the website dashboard.
"""
import sys
import os
import subprocess
import time

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    os.system("clear" if os.name != "nt" else "cls")
    print(f"{CYAN}{BOLD}{'=' * 65}{RESET}")
    print(f"{GREEN}{BOLD}      🌿 AGRIROVER - REAL-TIME APP INPUT MONITOR 🚜{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 65}{RESET}")
    print(f"Monitoring live commands from: {YELLOW}https://precision-agriculture-smart-rover.onrender.com{RESET}")
    print(f"Press buttons (Forward, Left, Right, Reverse, Pump, Speed) on website.")
    print(f"Updates will appear below with millisecond precision.")
    print(f"{CYAN}{'-' * 65}{RESET}\n", flush=True)

def main():
    print_banner()
    cmd = ["journalctl", "-u", "user@1000.service", "-f", "-n", "10", "-o", "cat"]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.strip()
            if "ROVER COMMAND ->" in line:
                # Format: [2026-10-02 11:32:49] ROVER COMMAND -> Forward | Speed: 100% | Pump: OFF
                ts = line.split("]")[0].replace("[", "") if "]" in line else time.strftime("%H:%M:%S")
                payload = line.split("ROVER COMMAND ->")[-1].strip()
                
                direction_color = GREEN if "Forward" in payload else (YELLOW if ("Left" in payload or "Right" in payload) else (RED if "Reverse" in payload else CYAN))
                pump_color = f"{GREEN}{BOLD}ON" if "Pump: ON" in payload else f"{RED}OFF"
                
                print(f"{CYAN}[{ts}]{RESET} {BOLD}INPUT RECEIVED 🎮 -> {direction_color}{payload.split('|')[0].strip()}{RESET} | {payload.split('|')[1].strip()} | Pump: {pump_color}{RESET}", flush=True)
            
            elif "CLOUD SYNC -> Render updated:" in line:
                ts = line.split("]")[0].replace("[", "") if "]" in line else time.strftime("%H:%M:%S")
                det = line.split("Render updated:")[-1].strip()
                det_color = GREEN if "Healthy" in det else YELLOW
                print(f"{CYAN}[{ts}]{RESET} {MAGENTA}AI DETECTED 🔬 -> {det_color}{det}{RESET}", flush=True)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Monitor stopped by user.{RESET}")
    except Exception as e:
        print(f"\n{RED}Error reading stream: {e}{RESET}")

if __name__ == "__main__":
    main()
