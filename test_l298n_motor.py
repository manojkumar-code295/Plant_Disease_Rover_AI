#!/usr/bin/env python3
"""
=============================================================================
AGRIROVER - STANDALONE L298N / L293D MOTOR DRIVER HARDWARE TEST SCRIPT
=============================================================================
Quickly verify forward, backward, left, right, and stop movement on your rover.
Supports modern Raspberry Pi OS (Debian Bookworm/Trixie) via lgpio & RPi.GPIO.

IMPORTANT: Ensure COMMON GROUND is connected!
Pi GND (Pin 6/9/14/20/39) ---> L298N GND ---> Battery Negative (-)
=============================================================================
"""

import sys
import time

# Pin Definitions (BCM GPIO Numbering)
# Physical Pins: Pin 11, 13, 32 (Left) | Pin 15, 16, 33 (Right)
LEFT_IN1 = 17   # Physical Pin 11 (Left Motor Forward)
LEFT_IN2 = 27   # Physical Pin 13 (Left Motor Backward)
LEFT_ENA = 12   # Physical Pin 32 (Left Motor Speed / ENA)

RIGHT_IN3 = 22  # Physical Pin 15 (Right Motor Forward)
RIGHT_IN4 = 23  # Physical Pin 16 (Right Motor Backward)
RIGHT_ENB = 13  # Physical Pin 33 (Right Motor Speed / ENB)

ALL_PINS = [LEFT_IN1, LEFT_IN2, LEFT_ENA, RIGHT_IN3, RIGHT_IN4, RIGHT_ENB]

# Terminal Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Initialize GPIO using lgpio (Standard on Raspberry Pi OS Bookworm/Trixie)
chip = None
try:
    import lgpio
    chip = lgpio.gpiochip_open(0)
    for pin in ALL_PINS:
        try:
            lgpio.gpio_free(chip, pin)
        except Exception:
            pass
        lgpio.gpio_claim_output(chip, pin, 0)
    USE_LGPIO = True
    print(f"{GREEN}✅ Initialized GPIO chip 0 via lgpio successfully.{RESET}")
except Exception as e_lgpio:
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in ALL_PINS:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        USE_LGPIO = False
        print(f"{GREEN}✅ Initialized GPIO via RPi.GPIO fallback.{RESET}")
    except Exception as e_rpi:
        print(f"{RED}❌ Could not access Raspberry Pi GPIO ({e_lgpio} / {e_rpi}){RESET}")
        print(f"{YELLOW}If agrirover-ai.service is running, stop it temporarily:{RESET}")
        print("  systemctl --user stop agrirover-ai.service")
        sys.exit(1)


def set_pin(pin, state):
    """Sets a GPIO pin to 1 (HIGH) or 0 (LOW)."""
    if USE_LGPIO:
        lgpio.gpio_write(chip, pin, 1 if state else 0)
    else:
        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


def stop():
    """Brakes all motor driver outputs to 0V."""
    for p in ALL_PINS:
        set_pin(p, 0)


def forward(speed=100):
    """
    Drive Forward:
    Left: IN1=HIGH, IN2=LOW, ENA=HIGH
    Right: IN3=HIGH, IN4=LOW, ENB=HIGH
    """
    stop()
    set_pin(LEFT_ENA, 1)
    set_pin(RIGHT_ENB, 1)
    set_pin(LEFT_IN1, 1)
    set_pin(LEFT_IN2, 0)
    set_pin(RIGHT_IN3, 1)
    set_pin(RIGHT_IN4, 0)


def backward(speed=100):
    """
    Drive Backward:
    Left: IN1=LOW, IN2=HIGH, ENA=HIGH
    Right: IN3=LOW, IN4=HIGH, ENB=HIGH
    """
    stop()
    set_pin(LEFT_ENA, 1)
    set_pin(RIGHT_ENB, 1)
    set_pin(LEFT_IN1, 0)
    set_pin(LEFT_IN2, 1)
    set_pin(RIGHT_IN3, 0)
    set_pin(RIGHT_IN4, 1)


def spin_left():
    """Spin Turn Left: Left Reverse, Right Forward."""
    stop()
    set_pin(LEFT_ENA, 1)
    set_pin(RIGHT_ENB, 1)
    set_pin(LEFT_IN1, 0)
    set_pin(LEFT_IN2, 1)
    set_pin(RIGHT_IN3, 1)
    set_pin(RIGHT_IN4, 0)


def spin_right():
    """Spin Turn Right: Left Forward, Right Reverse."""
    stop()
    set_pin(LEFT_ENA, 1)
    set_pin(RIGHT_ENB, 1)
    set_pin(LEFT_IN1, 1)
    set_pin(LEFT_IN2, 0)
    set_pin(RIGHT_IN3, 0)
    set_pin(RIGHT_IN4, 1)


def cleanup():
    """Cleanly resets all pins before exit."""
    stop()
    if USE_LGPIO and chip is not None:
        try:
            for p in ALL_PINS:
                lgpio.gpio_free(chip, p)
            lgpio.gpiochip_close(chip)
        except Exception:
            pass
    elif not USE_LGPIO:
        try:
            GPIO.cleanup()
        except Exception:
            pass


def main():
    print(f"\n{CYAN}{BOLD}{'=' * 65}{RESET}")
    print(f"{GREEN}{BOLD}   🏎️  AGRIROVER L298N / L293D MOTOR DRIVER HARDWARE TEST{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 65}{RESET}")
    print("Wiring Mapping (BCM Numbers):")
    print(f"  • LEFT MOTOR  : IN1 = GPIO {LEFT_IN1} (Pin 11), IN2 = GPIO {LEFT_IN2} (Pin 13), ENA = GPIO {LEFT_ENA} (Pin 32)")
    print(f"  • RIGHT MOTOR : IN3 = GPIO {RIGHT_IN3} (Pin 15), IN4 = GPIO {RIGHT_IN4} (Pin 16), ENB = GPIO {RIGHT_ENB} (Pin 33)")
    print(f"  • {YELLOW}{BOLD}COMMON GND  : Pi GND (Pin 6/9/14/20/39) ---> L298N GND ---> Battery (-){RESET}")
    print(f"{CYAN}{'-' * 65}{RESET}\n")

    try:
        print(f"{YELLOW}[STEP 1/4]{RESET} Testing FORWARD Motion for 2.5 seconds...")
        forward()
        time.sleep(2.5)
        stop()
        print(f"{GREEN}   -> Stopped.{RESET}\n")
        time.sleep(1.0)

        print(f"{YELLOW}[STEP 2/4]{RESET} Testing BACKWARD Motion for 2.5 seconds...")
        backward()
        time.sleep(2.5)
        stop()
        print(f"{GREEN}   -> Stopped.{RESET}\n")
        time.sleep(1.0)

        print(f"{YELLOW}[STEP 3/4]{RESET} Testing TURN LEFT for 1.8 seconds...")
        spin_left()
        time.sleep(1.8)
        stop()
        print(f"{GREEN}   -> Stopped.{RESET}\n")
        time.sleep(1.0)

        print(f"{YELLOW}[STEP 4/4]{RESET} Testing TURN RIGHT for 1.8 seconds...")
        spin_right()
        time.sleep(1.8)
        stop()
        print(f"{GREEN}   -> Stopped.{RESET}\n")

        print(f"{GREEN}{BOLD}✅ Hardware test completed successfully!{RESET}\n")

    except KeyboardInterrupt:
        print(f"\n{RED}Test interrupted by user. Stopping motors safely...{RESET}")
    finally:
        cleanup()
        print("GPIO resources freed.")


if __name__ == "__main__":
    main()
