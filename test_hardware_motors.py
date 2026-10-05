#!/usr/bin/env python3
"""
AgriRover Hardware Motor Diagnostic Script
Tests individual motor channels one by one to verify wiring, power, and ground.
"""
import time
import sys
import os

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Pin definitions (BCM)
LEFT_IN1 = 17   # Pin 11
LEFT_IN2 = 27   # Pin 13
LEFT_EN  = 12   # Pin 32
RIGHT_IN1 = 22  # Pin 15
RIGHT_IN2 = 23  # Pin 16
RIGHT_EN  = 13  # Pin 33
PUMP_PIN  = 26  # Pin 37

try:
    import lgpio
    chip = lgpio.gpiochip_open(0)
    for p in [LEFT_IN1, LEFT_IN2, LEFT_EN, RIGHT_IN1, RIGHT_IN2, RIGHT_EN, PUMP_PIN]:
        try:
            lgpio.gpio_free(chip, p)
        except Exception:
            pass
        lgpio.gpio_claim_output(chip, p, 0)
except Exception as e:
    print(f"{RED}Cannot claim GPIO directly ({e}).{RESET}")
    print(f"If agrirover-ai.service is running, stop it temporarily: {YELLOW}systemctl --user stop agrirover-ai.service{RESET}")
    sys.exit(1)

def write(pin, val):
    lgpio.gpio_write(chip, pin, 1 if val else 0)

def stop_all():
    for p in [LEFT_IN1, LEFT_IN2, LEFT_EN, RIGHT_IN1, RIGHT_IN2, RIGHT_EN, PUMP_PIN]:
        write(p, 0)

print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET}")
print(f"{GREEN}{BOLD}      🏎️  AGRIROVER HARDWARE MOTOR STEP-BY-STEP TEST{RESET}")
print(f"{CYAN}{BOLD}{'=' * 60}{RESET}\n")

try:
    # Test 1: Left Motor Forward
    print(f"{YELLOW}[TEST 1/4]{RESET} Testing LEFT MOTOR Forward (GPIO 17 + GPIO 12)...")
    write(LEFT_EN, 1)
    write(LEFT_IN1, 1)
    write(LEFT_IN2, 0)
    time.sleep(2.0)
    stop_all()
    print(f"   -> Left motor stopped.\n")
    time.sleep(1.0)

    # Test 2: Left Motor Backward
    print(f"{YELLOW}[TEST 2/4]{RESET} Testing LEFT MOTOR Backward (GPIO 27 + GPIO 12)...")
    write(LEFT_EN, 1)
    write(LEFT_IN1, 0)
    write(LEFT_IN2, 1)
    time.sleep(2.0)
    stop_all()
    print(f"   -> Left motor stopped.\n")
    time.sleep(1.0)

    # Test 3: Right Motor Forward
    print(f"{YELLOW}[TEST 3/4]{RESET} Testing RIGHT MOTOR Forward (GPIO 22 + GPIO 13)...")
    write(RIGHT_EN, 1)
    write(RIGHT_IN1, 1)
    write(RIGHT_IN2, 0)
    time.sleep(2.0)
    stop_all()
    print(f"   -> Right motor stopped.\n")
    time.sleep(1.0)

    # Test 4: Right Motor Backward
    print(f"{YELLOW}[TEST 4/4]{RESET} Testing RIGHT MOTOR Backward (GPIO 23 + GPIO 13)...")
    write(RIGHT_EN, 1)
    write(RIGHT_IN1, 0)
    write(RIGHT_IN2, 1)
    time.sleep(2.0)
    stop_all()
    print(f"   -> Right motor stopped.\n")

    print(f"{GREEN}{BOLD}Hardware test sequence completed!{RESET}")

finally:
    stop_all()
    lgpio.gpiochip_close(chip)
