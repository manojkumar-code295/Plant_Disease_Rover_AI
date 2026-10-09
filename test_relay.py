#!/usr/bin/env python3
"""
AgriRover Relay Module Diagnostic Script
========================================
Tests Relay triggering on Raspberry Pi 4 Physical Pin 37 (BCM GPIO 26).
Tests both:
  1. Active-HIGH Logic (1 = 3.3V ON, 0 = 0V OFF)
  2. Active-LOW Logic  (0 = 0V ON,   1 = 3.3V OFF)
"""

import time
import sys

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"

RELAY_PIN = 26 # Physical Pin 37

try:
    import lgpio
    chip = lgpio.gpiochip_open(0)
    try:
        lgpio.gpio_free(chip, RELAY_PIN)
    except Exception:
        pass
    lgpio.gpio_claim_output(chip, RELAY_PIN, 0)
except Exception as e:
    print(f"{RED}Cannot claim GPIO Pin {RELAY_PIN}: {e}{RESET}")
    sys.exit(1)

print(f"\n{CYAN}{BOLD}================================================================{RESET}")
print(f"{GREEN}{BOLD}      ⚡ AGRIROVER SPRAY PUMP RELAY HARDWARE DIAGNOSTIC{RESET}")
print(f"{CYAN}{BOLD}================================================================{RESET}")
print(f"Pin under test: {WHITE}Physical Pin 37 (BCM GPIO 26){RESET}\n")

try:
    print(f"{YELLOW}▶ TEST 1: Testing ACTIVE-HIGH Logic (Standard 3.3V Trigger){RESET}")
    print(f"  Setting Pin 37 to {GREEN}HIGH (3.3V){RESET} for 2 seconds...")
    lgpio.gpio_write(chip, RELAY_PIN, 1)
    print(f"  --> Listen for a {BOLD}LOUD CLICK{RESET} and check if the Relay LED is ON.")
    time.sleep(2.0)

    print(f"  Setting Pin 37 to {RED}LOW (0V){RESET} for 2 seconds...")
    lgpio.gpio_write(chip, RELAY_PIN, 0)
    print(f"  --> Listen for a {BOLD}CLICK OFF{RESET} and check if the Relay LED is OFF.\n")
    time.sleep(2.0)

    print(f"{YELLOW}▶ TEST 2: Testing ACTIVE-LOW Logic (Optocoupler Ground Trigger){RESET}")
    print(f"  Setting Pin 37 to {GREEN}LOW (0V){RESET} for 2 seconds...")
    lgpio.gpio_write(chip, RELAY_PIN, 0)
    print(f"  --> Listen for a {BOLD}LOUD CLICK{RESET} and check if the Relay LED is ON.")
    time.sleep(2.0)

    print(f"  Setting Pin 37 to {RED}HIGH (3.3V){RESET} for 2 seconds...")
    lgpio.gpio_write(chip, RELAY_PIN, 1)
    print(f"  --> Listen for a {BOLD}CLICK OFF{RESET} and check if the Relay LED is OFF.\n")
    time.sleep(2.0)

    print(f"{YELLOW}▶ TEST 3: Rapid 1-Second Click Cycle (3 Cycles)...{RESET}")
    for i in range(1, 4):
        print(f"  Cycle {i}/3: Setting HIGH (1)...")
        lgpio.gpio_write(chip, RELAY_PIN, 1)
        time.sleep(1.0)
        print(f"  Cycle {i}/3: Setting LOW (0)...")
        lgpio.gpio_write(chip, RELAY_PIN, 0)
        time.sleep(1.0)

    print(f"\n{GREEN}{BOLD}Relay signal test complete!{RESET}\n")

finally:
    lgpio.gpio_write(chip, RELAY_PIN, 0)
    lgpio.gpiochip_close(chip)
