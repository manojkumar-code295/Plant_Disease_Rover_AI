#!/usr/bin/env python3
"""
=============================================================================
AGRIROVER - STANDALONE BTS7960 (43A H-BRIDGE) MOTOR DRIVER TEST SCRIPT
=============================================================================
Minimal verification script for SINGLE BTS7960 High-Power Motor Driver.
Works on modern Raspberry Pi OS (Debian Bookworm/Trixie) via lgpio.

BTS7960 Logic Header:
  • VCC  ---> Raspberry Pi Pin 2 or 4 (5V Power)
  • GND  ---> Raspberry Pi Pin 6 or 9 (GND - COMMON GROUND)
  • R_EN ---> Raspberry Pi GPIO 12 (Physical Pin 32) [Or jumped to 5V]
  • L_EN ---> Raspberry Pi GPIO 12 (Physical Pin 32) [Or jumped to 5V]
  • RPWM ---> Raspberry Pi GPIO 17 (Physical Pin 11) [Forward PWM]
  • LPWM ---> Raspberry Pi GPIO 27 (Physical Pin 13) [Reverse PWM]
  • R_IS / L_IS ---> Not Connected (Float)

BTS7960 High-Power Terminals:
  • B+ (VCC) ---> 12V Battery Positive (+)
  • B- (GND) ---> 12V Battery Negative (-) [MUST SHARE GND WITH PI!]
  • M+ / M-  ---> DC Drive Motor(s)
=============================================================================
"""

import sys
import time

RPWM_PIN = 17   # Physical Pin 11 (Forward Direction)
LPWM_PIN = 27   # Physical Pin 13 (Reverse Direction)
EN_PIN   = 12   # Physical Pin 32 (R_EN + L_EN Enable)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

chip = None
try:
    import lgpio
    chip = lgpio.gpiochip_open(0)
    for p in [RPWM_PIN, LPWM_PIN, EN_PIN]:
        try:
            lgpio.gpio_free(chip, p)
        except Exception:
            pass
        lgpio.gpio_claim_output(chip, p, 0)
    USE_LGPIO = True
    print(f"{GREEN}✅ Initialized GPIO via lgpio.{RESET}")
except Exception as e_lg:
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for p in [RPWM_PIN, LPWM_PIN, EN_PIN]:
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
        USE_LGPIO = False
        print(f"{GREEN}✅ Initialized GPIO via RPi.GPIO fallback.{RESET}")
    except Exception as e_rpi:
        print(f"{RED}❌ GPIO Init Failed ({e_lg} / {e_rpi}){RESET}")
        print(f"{YELLOW}If agrirover-ai.service is running, stop it temporarily:{RESET}")
        print("  systemctl --user stop agrirover-ai.service")
        sys.exit(1)


def set_pwm(pin, duty):
    """Sets PWM duty cycle (0 to 100%) at 1000 Hz."""
    duty = max(0, min(100, int(duty)))
    if USE_LGPIO:
        lgpio.tx_pwm(chip, pin, 1000, duty)
    else:
        # Simple digital fallback if hardware PWM not available
        GPIO.output(pin, GPIO.HIGH if duty > 0 else GPIO.LOW)


def stop_motor():
    """Brakes BTS7960: 0% PWM and EN LOW."""
    if USE_LGPIO:
        set_pwm(RPWM_PIN, 0)
        set_pwm(LPWM_PIN, 0)
        lgpio.gpio_write(chip, EN_PIN, 0)
    else:
        set_pwm(RPWM_PIN, 0)
        set_pwm(LPWM_PIN, 0)
        GPIO.output(EN_PIN, GPIO.LOW)


def motor_forward(speed=60):
    """
    Drive Motor FORWARD:
    - R_EN & L_EN = HIGH (Enabled)
    - LPWM = 0 (Braked)
    - RPWM = speed% PWM (Forward Drive)
    """
    stop_motor()
    time.sleep(0.02)
    if USE_LGPIO:
        lgpio.gpio_write(chip, EN_PIN, 1)
    else:
        GPIO.output(EN_PIN, GPIO.HIGH)
    set_pwm(LPWM_PIN, 0)
    set_pwm(RPWM_PIN, speed)


def motor_backward(speed=60):
    """
    Drive Motor BACKWARD:
    - R_EN & L_EN = HIGH (Enabled)
    - RPWM = 0 (Braked)
    - LPWM = speed% PWM (Reverse Drive)
    """
    stop_motor()
    time.sleep(0.02)
    if USE_LGPIO:
        lgpio.gpio_write(chip, EN_PIN, 1)
    else:
        GPIO.output(EN_PIN, GPIO.HIGH)
    set_pwm(RPWM_PIN, 0)
    set_pwm(LPWM_PIN, speed)


def cleanup():
    stop_motor()
    if USE_LGPIO and chip is not None:
        try:
            for p in [RPWM_PIN, LPWM_PIN, EN_PIN]:
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
    print(f"{GREEN}{BOLD}    ⚡ AGRIROVER BTS7960 (43A H-BRIDGE) HARDWARE TEST{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 65}{RESET}")
    print("Pin Connections (BCM):")
    print(f"  • RPWM (Forward) : GPIO {RPWM_PIN} (Physical Pin 11)")
    print(f"  • LPWM (Reverse) : GPIO {LPWM_PIN} (Physical Pin 13)")
    print(f"  • R_EN + L_EN    : GPIO {EN_PIN}   (Physical Pin 32)")
    print(f"  • VCC (Logic)    : 5V (Physical Pin 2 or 4)")
    print(f"  • GND (Logic)    : GND (Physical Pin 6, 9, 14, 20, or 39)")
    print(f"  • {YELLOW}{BOLD}COMMON GROUND  : Pi GND <---> BTS7960 GND <---> Battery (-){RESET}")
    print(f"{CYAN}{'-' * 65}{RESET}\n")

    try:
        print(f"{YELLOW}[TEST 1/2]{RESET} Testing FORWARD Motion at 60% speed for 3 seconds...")
        motor_forward(60)
        time.sleep(3.0)
        stop_motor()
        print(f"{GREEN}   -> Motor stopped.{RESET}\n")
        time.sleep(1.0)

        print(f"{YELLOW}[TEST 2/2]{RESET} Testing BACKWARD Motion at 60% speed for 3 seconds...")
        motor_backward(60)
        time.sleep(3.0)
        stop_motor()
        print(f"{GREEN}   -> Motor stopped.{RESET}\n")

        print(f"{GREEN}{BOLD}✅ BTS7960 hardware test completed successfully!{RESET}\n")

    except KeyboardInterrupt:
        print(f"\n{RED}Interrupted by user. Stopping motor safely...{RESET}")
    finally:
        cleanup()
        print("GPIO cleaned up.")


if __name__ == "__main__":
    main()
