#!/usr/bin/env python3
"""
AgriRover Live Command Terminal Monitor
Displays real-time AI disease detection, cloud sync status,
rover commands, and hardware GPIO states.
Supports interactive keyboard controls directly from terminal.
"""

import os
import sys
import time
import socket
import json
import threading
import subprocess
import urllib.request
import urllib.parse

# ANSI Escape Sequences
CLEAR = "\033[2J\033[H"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
WHITE = "\033[97m"
BG_DARK = "\033[48;5;234m"

LOCAL_API = "http://127.0.0.1:5000"
CLOUD_API = "https://precision-agriculture-smart-rover.onrender.com"

running = True
recent_logs = []
logs_lock = threading.Lock()

def add_log(msg, tag="INFO"):
    t = time.strftime("%H:%M:%S")
    with logs_lock:
        recent_logs.append((t, tag, msg))
        if len(recent_logs) > 12:
            recent_logs.pop(0)

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def check_internet():
    try:
        s = socket.create_connection(("8.8.8.8", 53), timeout=1.5)
        s.close()
        return True
    except Exception:
        return False

def check_cloud():
    try:
        req = urllib.request.Request(f"{CLOUD_API}/api/rover/command")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False

def read_gpio_pins():
    """Reads Linux pinctrl for rover motor and pump pins."""
    try:
        out = subprocess.check_output(
            ["pinctrl", "get", "17,27,22,23,12,13,26"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        pins = {}
        for line in out.strip().splitlines():
            # e.g.: "17: op -- pd | hi // GPIO17 = output"
            parts = line.split(":")
            if len(parts) == 2:
                p_num = parts[0].strip()
                val = "HI" if "| hi" in parts[1] else "LO"
                pins[p_num] = val
        return pins
    except Exception:
        return {}

def send_local_command(cmd, speed=None):
    try:
        payload = {"command": cmd}
        if speed is not None:
            payload["speed"] = speed
        req = urllib.request.Request(
            f"{LOCAL_API}/api/rover/control",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("success", False)
    except Exception as e:
        return False

def telemetry_poller():
    global running
    last_cmd_id = -1
    last_disease = ""
    add_log("Telemetry poller connected to local rover daemon", "INIT")

    while running:
        try:
            req = urllib.request.Request(f"{LOCAL_API}/api/dashboard-state")
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                # Check for new rover command
                rover = data.get("rover", {})
                cid = rover.get("command_id", 0)
                if cid != last_cmd_id and cid > 0:
                    dir_name = rover.get("direction", "Stopped")
                    spd = rover.get("speed", 0)
                    add_log(f"ROVER COMMAND: {dir_name.upper()} (Speed: {spd}%)", "CMD")
                    last_cmd_id = cid

                # Check for new disease detection
                res = data.get("result", {})
                disease = res.get("disease", "")
                conf = res.get("confidence", 0)
                if disease and disease != last_disease and disease != "NO LEAF DETECTED":
                    add_log(f"AI DETECTED: {disease} ({conf:.1f}%)", "AI")
                    last_disease = disease

        except Exception:
            pass
        time.sleep(0.3)

def input_listener():
    global running
    while running:
        try:
            line = input().strip().lower()
            if line in ("q", "quit", "exit"):
                running = False
                print(f"\n{YELLOW}Exiting monitor... Dropping to shell.{RESET}")
                break
            elif line in ("w", "fwd", "forward", "start"):
                send_local_command("start")
                add_log("Terminal input: START (FORWARD)", "KEY")
            elif line in ("s", "stop", " "):
                send_local_command("stop")
                add_log("Terminal input: STOP", "KEY")
            elif line in ("a", "l", "left"):
                send_local_command("left")
                add_log("Terminal input: LEFT", "KEY")
            elif line in ("d", "r", "right"):
                send_local_command("right")
                add_log("Terminal input: RIGHT", "KEY")
            elif line in ("x", "rev", "backward", "back"):
                send_local_command("backward")
                add_log("Terminal input: BACKWARD", "KEY")
            elif line in ("p", "pump"):
                try:
                    req = urllib.request.Request(
                        f"{LOCAL_API}/api/rover/pump",
                        data=json.dumps({"state": True}).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    urllib.request.urlopen(req, timeout=1.0)
                    add_log("Terminal input: PUMP TOGGLED", "KEY")
                except Exception:
                    pass
        except EOFError:
            time.sleep(1.0)
        except KeyboardInterrupt:
            running = False
            break

def main():
    global running
    poller_t = threading.Thread(target=telemetry_poller, daemon=True)
    poller_t.start()

    input_t = threading.Thread(target=input_listener, daemon=True)
    input_t.start()

    last_net_check = 0
    is_online = False
    is_cloud_ok = False
    local_ip = get_ip()

    try:
        while running:
            now = time.time()
            if now - last_net_check > 5.0:
                is_online = check_internet()
                is_cloud_ok = check_cloud()
                local_ip = get_ip()
                last_net_check = now

            # Fetch state for display
            rover_data = {}
            result_data = {}
            try:
                req = urllib.request.Request(f"{LOCAL_API}/api/dashboard-state")
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
                    rover_data = d.get("rover", {})
                    result_data = d.get("result", {})
            except Exception:
                pass

            pins = read_gpio_pins()

            # Render Screen
            buf = []
            buf.append(CLEAR)
            buf.append(f"{GREEN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════════╗{RESET}")
            buf.append(f"{GREEN}{BOLD}║         🌱 PRECISION AGRICULTURE SMART ROVER - LIVE COMMAND TERMINAL        ║{RESET}")
            buf.append(f"{GREEN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════════╝{RESET}")

            # Network Status Bar
            net_badge = f"{GREEN}● CONNECTED{RESET}" if is_online else f"{RED}● OFFLINE{RESET}"
            cloud_badge = f"{CYAN}● SYNCHRONIZED{RESET}" if is_cloud_ok else f"{YELLOW}● CONNECTING...{RESET}"
            buf.append(f" {BOLD}Internet:{RESET} {net_badge}  |  {BOLD}Local IP:{RESET} {WHITE}{local_ip}{RESET}  |  {BOLD}Cloud Relay:{RESET} {cloud_badge}")
            buf.append(f" {DIM}Cloud URL: {CLOUD_API}{RESET}")
            buf.append(f"{DIM}───────────────────────────────────────────────────────────────────────────────{RESET}")

            # AI & Camera Telemetry
            ai_stat = rover_data.get("ai_status", "Scanning...")
            fps = rover_data.get("fps", 0.0)
            det = result_data.get("disease", "Scanning...")
            conf = result_data.get("confidence", 0.0)
            buf.append(f" {BOLD}📹 AI Camera:{RESET} {GREEN}Native Picamera2 (640x480){RESET}  |  {BOLD}FPS:{RESET} {CYAN}{fps:.1f}{RESET}")
            buf.append(f" {BOLD}🔬 AI Model:{RESET}  {WHITE}MobileNetV2 (99% Acc){RESET}  |  {BOLD}Status:{RESET} {ai_stat}")
            det_color = GREEN if "healthy" in det.lower() else YELLOW
            buf.append(f" {BOLD}🌿 Detection:{RESET} {det_color}{det}{RESET} ({conf:.1f}%)")
            buf.append(f"{DIM}───────────────────────────────────────────────────────────────────────────────{RESET}")

            # Rover Actuation & Physical Hardware Pins
            direction = rover_data.get("direction", "Stopped")
            speed = rover_data.get("speed", 0)
            pump = rover_data.get("pump", False)
            dir_color = GREEN if direction in ("Forward", "start") else (RED if direction == "Stopped" else CYAN)
            pump_str = f"{CYAN}ACTIVE (SPRAYING){RESET}" if pump else f"{DIM}OFF{RESET}"

            buf.append(f" {BOLD}🚜 Rover Drive:{RESET} {dir_color}{direction.upper()}{RESET}  |  {BOLD}Speed:{RESET} {speed}%  |  {BOLD}Pump Relay:{RESET} {pump_str}")

            # Pins row
            p17 = pins.get("17", "--")
            p27 = pins.get("27", "--")
            p22 = pins.get("22", "--")
            p23 = pins.get("23", "--")
            p12 = pins.get("12", "--")
            p13 = pins.get("13", "--")
            p26 = pins.get("26", "--")

            def pin_fmt(v):
                return f"{GREEN}HI{RESET}" if v == "HI" else f"{DIM}LO{RESET}"

            buf.append(f" {BOLD}⚡ Hardware Pins:{RESET} Left: IN1(17)={pin_fmt(p17)} IN2(27)={pin_fmt(p27)} EN(12)={pin_fmt(p12)} | Right: IN3(22)={pin_fmt(p22)} IN4(23)={pin_fmt(p23)} EN(13)={pin_fmt(p13)}")
            buf.append(f"                 Spray Pump: GPIO 26={pin_fmt(p26)}")
            buf.append(f"{DIM}───────────────────────────────────────────────────────────────────────────────{RESET}")

            # Live Event Log
            buf.append(f" {BOLD}📋 Live Event Log:{RESET}")
            with logs_lock:
                for t, tag, msg in recent_logs[-7:]:
                    tag_col = GREEN if tag == "CMD" else (CYAN if tag == "AI" else (YELLOW if tag == "KEY" else WHITE))
                    buf.append(f"   {DIM}[{t}]{RESET} {tag_col}[{tag:4}]{RESET} {msg}")

            buf.append(f"{DIM}───────────────────────────────────────────────────────────────────────────────{RESET}")
            buf.append(f" {BOLD}⌨️  Controls:{RESET} [W/F]=Forward [S/Space]=Stop [A]=Left [D]=Right [X]=Reverse [P]=Pump [Q]=Exit")
            buf.append(f" {DIM}Enter command > {RESET}")

            sys.stdout.write("\n".join(buf))
            sys.stdout.flush()

            time.sleep(0.4)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print(f"\n{GREEN}AgriRover Terminal closed. Active bash session ready.{RESET}\n")

if __name__ == "__main__":
    main()
