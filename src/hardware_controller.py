"""
Hardware Integration and Safety Controller Module.
Manages:
  - DC Motors & Motor Driver (L298N/L293D)
  - Differential Drive & Multi-Speed PWM regulation
  - Obstacle Detection Auto-Braking & Clearance Resumption
  - Precision Water Pump & Relay Module (WATER ONLY)
  - Spray duration, cooldown intervals, and maximum runtime limits
  - Emergency Stop & Auto-Shutoff safeguards
  - Distinct Hardware vs Simulation modes (never fakes hardware status)
"""

import time
import threading
from src.config import (
    IS_RASPBERRY_PI,
    MOTOR_LEFT_ENA, MOTOR_LEFT_IN1, MOTOR_LEFT_IN2,
    MOTOR_RIGHT_ENB, MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4,
    PUMP_RELAY_PIN, STATUS_LED_PIN,
    SPEED_LOW, SPEED_MEDIUM, SPEED_HIGH, DEFAULT_SPEED,
    DEFAULT_SPRAY_DURATION, DEFAULT_SPRAY_INTERVAL, MAX_PUMP_RUNTIME,
    AUTO_SPRAY_CONFIDENCE_MIN
)

class HardwareController:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HardwareController()
        return cls._instance

    def __init__(self):
        # Platform & verification state
        self.hardware_mode = "HARDWARE_GPIO" if IS_RASPBERRY_PI else "SIMULATION"
        self.hardware_verified = False
        self.GPIO = None
        self.pwm_left = None
        self.pwm_right = None

        # Motor Navigation State
        self.control_mode = "manual"  # "manual" or "automatic"
        self.direction = "STOP"
        self.speed = DEFAULT_SPEED
        self.obstacle_stopped = False
        self.last_obstacle_time = 0

        # Water Pump State (WATER ONLY)
        self.pump_enabled = True
        self.pump_active = False
        self.auto_spray_enabled = True
        self.spray_duration = DEFAULT_SPRAY_DURATION
        self.spray_interval = DEFAULT_SPRAY_INTERVAL
        self.last_spray_time = 0
        self.pump_start_time = 0
        self.pump_lock = threading.Lock()

        # Initialize Hardware if on Raspberry Pi
        if IS_RASPBERRY_PI:
            self._init_gpio()
        else:
            print("[Hardware Controller] Simulation Mode active (Laptop Testing). Real GPIO pins will not be actuated.")

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)

            # Outputs
            for pin in [MOTOR_LEFT_IN1, MOTOR_LEFT_IN2, MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4, PUMP_RELAY_PIN, STATUS_LED_PIN]:
                self.GPIO.setup(pin, self.GPIO.OUT)
                self.GPIO.output(pin, self.GPIO.LOW)

            # PWM Speed pins (100 Hz)
            self.GPIO.setup(MOTOR_LEFT_ENA, self.GPIO.OUT)
            self.GPIO.setup(MOTOR_RIGHT_ENB, self.GPIO.OUT)
            self.pwm_left = self.GPIO.PWM(MOTOR_LEFT_ENA, 100)
            self.pwm_right = self.GPIO.PWM(MOTOR_RIGHT_ENB, 100)
            self.pwm_left.start(self.speed)
            self.pwm_right.start(self.speed)

            self.hardware_verified = True
            print("[Hardware Controller] Confirmed: Raspberry Pi GPIO hardware initialized with PWM.")
        except Exception as e:
            print(f"[Hardware Controller] Notice: Hardware GPIO not available ({e}). Running in Simulation Mode.")
            self.hardware_verified = False
            self.hardware_mode = "SIMULATION"

    # ==========================================
    # MOTOR & SPEED CONTROLS
    # ==========================================

    def set_speed(self, speed_percent):
        self.speed = max(15, min(100, int(speed_percent)))
        if self.hardware_verified and self.pwm_left and self.pwm_right:
            self.pwm_left.ChangeDutyCycle(self.speed)
            self.pwm_right.ChangeDutyCycle(self.speed)
        return self.get_telemetry()

    def set_control_mode(self, mode):
        if mode in ["manual", "automatic"]:
            self.control_mode = mode
        return self.get_telemetry()

    def forward(self):
        if self.obstacle_stopped:
            return self.get_telemetry()
        self.direction = "FORWARD"
        if self.hardware_verified:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.LOW)
        return self.get_telemetry()

    def backward(self):
        self.direction = "BACKWARD"
        if self.hardware_verified:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.LOW)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.HIGH)
        return self.get_telemetry()

    def turn_left(self):
        self.direction = "LEFT"
        if self.hardware_verified:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.LOW)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.LOW)
        return self.get_telemetry()

    def turn_right(self):
        self.direction = "RIGHT"
        if self.hardware_verified:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.HIGH)
        return self.get_telemetry()

    def stop(self):
        self.direction = "STOP"
        if self.hardware_verified:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.LOW)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.LOW)
        return self.get_telemetry()

    def handle_obstacle_detected(self, obstacle_label="Obstacle"):
        """Emergency obstacle stop."""
        self.obstacle_stopped = True
        self.last_obstacle_time = time.time()
        self.stop()

    def handle_path_cleared(self):
        """Clears obstacle flag and optionally resumes if in automatic mode."""
        self.obstacle_stopped = False
        if self.control_mode == "automatic":
            self.forward()

    # ==========================================
    # WATER PUMP & SPRAYER CONTROLS (WATER ONLY)
    # ==========================================

    def pump_on(self):
        if not self.pump_enabled:
            return self.get_telemetry()
        self.pump_active = True
        self.pump_start_time = time.time()
        if self.hardware_verified:
            self.GPIO.output(PUMP_RELAY_PIN, self.GPIO.HIGH)
            self.GPIO.output(STATUS_LED_PIN, self.GPIO.HIGH)
        
        # Start max runtime watchdog thread
        threading.Thread(target=self._watchdog_pump_cutoff, daemon=True).start()
        return self.get_telemetry()

    def pump_off(self):
        self.pump_active = False
        if self.hardware_verified:
            self.GPIO.output(PUMP_RELAY_PIN, self.GPIO.LOW)
            self.GPIO.output(STATUS_LED_PIN, self.GPIO.LOW)
        return self.get_telemetry()

    def _watchdog_pump_cutoff(self):
        """Prevents pump from running continuously beyond MAX_PUMP_RUNTIME."""
        time.sleep(MAX_PUMP_RUNTIME)
        if self.pump_active and (time.time() - self.pump_start_time) >= MAX_PUMP_RUNTIME:
            print("[Hardware Safety] Pump reached max continuous runtime limit! Auto-shutoff triggered.")
            self.pump_off()

    def trigger_spray_burst(self, duration=None):
        """Executes a non-blocking calibrated water spray burst with cooldown."""
        dur = duration or self.spray_duration
        now = time.time()

        if (now - self.last_spray_time) < self.spray_interval:
            return {"triggered": False, "reason": "cooldown_active"}

        def _run_spray():
            with self.pump_lock:
                self.last_spray_time = time.time()
                self.pump_on()
                time.sleep(dur)
                self.pump_off()

        threading.Thread(target=_run_spray, daemon=True).start()
        return {"triggered": True, "duration": dur}

    def evaluate_auto_spray(self, is_leaf, is_healthy, confidence):
        """
        Auto-spray logic:
        Triggered when a valid plant leaf is detected, auto_spray is enabled,
        pump is enabled, and cooldown has elapsed.
        """
        if not self.auto_spray_enabled or not self.pump_enabled or not is_leaf:
            return {"triggered": False, "reason": "not_eligible"}

        if confidence < AUTO_SPRAY_CONFIDENCE_MIN:
            return {"triggered": False, "reason": "confidence_low"}

        now = time.time()
        if (now - self.last_spray_time) < self.spray_interval:
            return {"triggered": False, "reason": "cooldown_active"}

        return self.trigger_spray_burst(self.spray_duration)

    def emergency_stop(self):
        """Immediately disables all motors and pump outputs."""
        self.stop()
        self.pump_off()
        print("[Hardware Controller] EMERGENCY STOP ACTIVATED. All motor and pump outputs disabled.")
        return self.get_telemetry()

    def get_telemetry(self):
        now = time.time()
        cooldown_rem = max(0.0, self.spray_interval - (now - self.last_spray_time))
        return {
            "hardware_mode": self.hardware_mode,
            "hardware_verified": self.hardware_verified,
            "direction": self.direction,
            "speed": self.speed,
            "control_mode": self.control_mode,
            "obstacle_stopped": self.obstacle_stopped,
            "pump_enabled": self.pump_enabled,
            "pump_active": self.pump_active,
            "auto_spray_enabled": self.auto_spray_enabled,
            "spray_duration": self.spray_duration,
            "spray_interval": self.spray_interval,
            "cooldown_remaining": round(cooldown_rem, 1)
        }

hardware_controller = HardwareController.get_instance()
