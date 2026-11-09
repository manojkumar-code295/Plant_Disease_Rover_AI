#!/usr/bin/env python3
"""
AgriRover Minimal Standalone Motor Test Script
Tests Forward, Backward, and Stop actuation on L298N / L293D motor driver.

USAGE:
  1. Stop background service if it is running:
       systemctl --user stop agrirover-ai.service
  2. Run this test:
       python3 minimal_motor_test.py
  3. Restart background service when done:
       systemctl --user start agrirover-ai.service
"""

import time
import sys

# Broadcom (BCM) GPIO Pin Mapping
IN1 = 17  # Raspberry Pi Physical Pin 11 -> L298N IN1 (Left Motor Forward)
IN2 = 27  # Raspberry Pi Physical Pin 13 -> L298N IN2 (Left Motor Backward)
IN3 = 22  # Raspberry Pi Physical Pin 15 -> L298N IN3 (Right Motor Forward)
IN4 = 23  # Raspberry Pi Physical Pin 16 -> L298N IN4 (Right Motor Backward)
ENA = 12  # Raspberry Pi Physical Pin 32 -> L298N ENA (Left Speed / Enable)
ENB = 13  # Raspberry Pi Physical Pin 33 -> L298N ENB (Right Speed / Enable)

PINS = [IN1, IN2, IN3, IN4, ENA, ENB]

try:
    import lgpio
    chip = lgpio.gpiochip_open(0)
    for pin in PINS:
        try:
            lgpio.gpio_free(chip, pin)
        except Exception:
            pass
        lgpio.gpio_claim_output(chip, pin, 0)
except Exception as err:
    print(f"\n[ERROR] Unable to claim GPIO pins ({err}).")
    print("If 'agrirover-ai.service' is currently running, stop it first:")
    print("   systemctl --user stop agrirover-ai.service")
    sys.exit(1)


def write_pin(pin, state):
    """Writes HIGH (1) or LOW (0) to a GPIO pin."""
    lgpio.gpio_write(chip, pin, 1 if state else 0)


def forward():
    """
    FORWARD LOGIC:
    Left Motor:  IN1=HIGH, IN2=LOW, ENA=HIGH
    Right Motor: IN3=HIGH, IN4=LOW, ENB=HIGH
    """
    print("  -> MOVING FORWARD (IN1=HIGH, IN2=LOW, IN3=HIGH, IN4=LOW)")
    write_pin(ENA, 1)
    write_pin(ENB, 1)
    write_pin(IN1, 1)
    write_pin(IN2, 0)
    write_pin(IN3, 1)
    write_pin(IN4, 0)


def backward():
    """
    BACKWARD LOGIC:
    Left Motor:  IN1=LOW, IN2=HIGH, ENA=HIGH
    Right Motor: IN3=LOW, IN4=HIGH, ENB=HIGH
    """
    print("  -> MOVING BACKWARD (IN1=LOW, IN2=HIGH, IN3=LOW, IN4=HIGH)")
    write_pin(ENA, 1)
    write_pin(ENB, 1)
    write_pin(IN1, 0)
    write_pin(IN2, 1)
    write_pin(IN3, 0)
    write_pin(IN4, 1)


def stop():
    """
    STOP LOGIC:
    All input pins pulled LOW (0V).
    """
    print("  -> STOPPING MOTORS (All LOW)")
    for pin in PINS:
        write_pin(pin, 0)


def run_test():
    print("=" * 65)
    print("       AGRIROVER L298N MOTOR LOGIC STANDALONE TEST")
    print("=" * 65)

    try:
        print("\n[STEP 1/3] Testing FORWARD for 3 seconds...")
        forward()
        time.sleep(3.0)

        print("\n[STEP 2/3] Pausing 1.5 seconds...")
        stop()
        time.sleep(1.5)

        print("\n[STEP 3/3] Testing BACKWARD for 3 seconds...")
        backward()
        time.sleep(3.0)

        print("\n[COMPLETE] Actuation sequence finished. Stopping all motors.")
        stop()

    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.")
    finally:
        stop()
        lgpio.gpiochip_close(chip)
        print("GPIO resources safely closed.")


if __name__ == "__main__":
    run_test()
