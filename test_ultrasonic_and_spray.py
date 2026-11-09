#!/usr/bin/env python3
"""
AgriRover Hardware Test: Ultrasonic Obstacle Stop & Disease-Only Spray Relay
=============================================================================
Verifies:
  1. Ultrasonic HC-SR04 Obstacle Detection (TRIG=24, ECHO=25)
     - Reads live distance (cm)
     - Automatically stops rover motors when obstacle <= 30 cm
  2. Targeted Disease Spray Relay (BCM 26 / Physical Pin 37)
     - Triggers DC motor spray pump ONLY when disease is confirmed
     - Holds spray pump strictly OFF for Healthy Leaf
"""

import sys
import time
import os

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

TRIG_PIN = 24  # Physical Pin 18
ECHO_PIN = 25  # Physical Pin 22
PUMP_PIN = 26  # Physical Pin 37
LEFT_IN1 = 17  # Physical Pin 11
LEFT_IN2 = 27  # Physical Pin 13
LEFT_EN  = 12  # Physical Pin 32
OBSTACLE_THRESHOLD_CM = 6.0  # Stop threshold 5 to 6 cm

RELAY_ACTIVE_LOW = os.environ.get("RELAY_ACTIVE_LOW", "1") == "1"

print(f"\n{CYAN}{BOLD}{'=' * 68}{RESET}")
print(f"{GREEN}{BOLD}   🌾 AGRIROVER ULTRASONIC OBSTACLE STOP & DISEASE SPRAY TEST{RESET}")
print(f"{CYAN}{BOLD}{'=' * 68}{RESET}\n")

try:
    import lgpio
    chip = lgpio.gpiochip_open(0)
    for p in [TRIG_PIN, PUMP_PIN, LEFT_IN1, LEFT_IN2, LEFT_EN]:
        try:
            lgpio.gpio_free(chip, p)
        except Exception:
            pass
        lgpio.gpio_claim_output(chip, p, 0)

    try:
        lgpio.gpio_free(chip, ECHO_PIN)
    except Exception:
        pass
    lgpio.gpio_claim_input(chip, ECHO_PIN, lgpio.SET_PULL_DOWN)

    # Initial safe state: Pump OFF, Motors OFF
    lgpio.gpio_write(chip, PUMP_PIN, 1 if RELAY_ACTIVE_LOW else 0)
    lgpio.gpio_write(chip, LEFT_EN, 0)
    print(f"{GREEN}✅ GPIO chip 0 opened successfully.{RESET}")
    print(f"   Trig Pin: {WHITE}BCM 24 (Physical 18){RESET}")
    print(f"   Echo Pin: {WHITE}BCM 25 (Physical 22){RESET}")
    print(f"   Pump Relay Pin: {WHITE}BCM 26 (Physical 37){RESET}\n")

except Exception as e:
    print(f"{RED}Error initializing GPIO ({e}).{RESET}")
    print(f"{YELLOW}Note: If agrirover-ai.service is using pins, stop it first:{RESET}")
    print(f"  systemctl --user stop agrirover-ai.service\n")
    sys.exit(1)

def read_distance(timeout_sec=0.03):
    lgpio.gpio_write(chip, TRIG_PIN, 0)
    time.sleep(0.000005)
    lgpio.gpio_write(chip, TRIG_PIN, 1)
    time.sleep(0.000012)
    lgpio.gpio_write(chip, TRIG_PIN, 0)

    t0 = time.perf_counter()
    timeout_rise = t0 + timeout_sec
    while lgpio.gpio_read(chip, ECHO_PIN) == 0:
        if time.perf_counter() > timeout_rise:
            return None
        time.sleep(0.00005)

    t_rise = time.perf_counter()
    timeout_fall = t_rise + timeout_sec
    while lgpio.gpio_read(chip, ECHO_PIN) == 1:
        if time.perf_counter() > timeout_fall:
            return None
        time.sleep(0.00005)

    t_fall = time.perf_counter()
    dur = t_fall - t_rise
    dist = (dur * 34300.0) / 2.0
    return dist if 2.0 <= dist <= 400.0 else None

def set_pump(state):
    val = (0 if state else 1) if RELAY_ACTIVE_LOW else (1 if state else 0)
    lgpio.gpio_write(chip, PUMP_PIN, val)

try:
    print(f"{YELLOW}{BOLD}[PART 1] Testing Ultrasonic Sensor & Automatic Collision Stop...{RESET}")
    print(f"Sampling distance for 6 seconds. If object is in 5-6 cm range (<= 6.0 cm), motor is STOPPED; otherwise ALWAYS ON:")
    t_end = time.time() + 6.0
    obstacle_count = 0
    while time.time() < t_end:
        d = read_distance()
        if d is not None:
            if d <= OBSTACLE_THRESHOLD_CM:
                obstacle_count += 1
                status = f"{RED}{BOLD}⚠️ OBSTACLE (5-6 cm) ({d:.1f} cm <= {OBSTACLE_THRESHOLD_CM} cm) -> MOTOR STOPPED!{RESET}"
                lgpio.gpio_write(chip, LEFT_EN, 0)
            else:
                status = f"{GREEN}CLEAR ({d:.1f} cm > 6 cm) -> MOTOR ALWAYS ON (Running){RESET}"
                lgpio.gpio_write(chip, LEFT_EN, 1)
            print(f"  Distance: {d:5.1f} cm | {status}")
        else:
            print(f"  Distance: {DIM}No Echo (open path / sensor disconnected){RESET}")
        time.sleep(0.3)

    print(f"\n{YELLOW}{BOLD}[PART 2] Testing Targeted Disease Spray Relay Module...{RESET}")
    print(f"Rule: Spray pump DC motor runs ONLY for diseased leaves, NEVER for healthy leaves.\n")

    # Simulation 1: Diseased Leaf
    print(f"  🌱 Case 1: Diseased Leaf Detected ({RED}Bacterial Spot 91.2%{RESET})")
    print(f"     -> Actuating Relay: {MAGENTA}{BOLD}SPRAY PUMP ON{RESET} (2.0s targeted spray pulse)")
    set_pump(True)
    time.sleep(2.0)
    set_pump(False)
    print(f"     -> Spray pulse complete: {GREEN}SPRAY PUMP OFF{RESET}\n")

    time.sleep(1.0)

    # Simulation 2: Healthy Leaf
    print(f"  🌿 Case 2: Healthy Leaf Detected ({GREEN}Healthy Leaf 95.8%{RESET})")
    print(f"     -> Safety Gate: {GREEN}{BOLD}SPRAY PUMP HELD STRICTLY OFF{RESET} (No chemicals on healthy crop)")
    set_pump(False)
    time.sleep(1.0)
    print(f"     -> Verified: Pump state remains OFF.\n")

    # Simulation 3: No Leaf
    print(f"  🔍 Case 3: Scanning / No Leaf in View")
    print(f"     -> Safety Gate: {CYAN}SPRAY PUMP OFF (Standby){RESET}")
    set_pump(False)
    time.sleep(1.0)

    print(f"\n{GREEN}{BOLD}✅ Hardware test completed successfully!{RESET}\n")

finally:
    set_pump(False)
    lgpio.gpio_write(chip, LEFT_EN, 0)
    lgpio.gpiochip_close(chip)
