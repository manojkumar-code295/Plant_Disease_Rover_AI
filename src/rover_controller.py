"""
Smart Rover and Sprayer Pump Controller Module.
Supports:
  - Differential Drive Motors (Forward, Backward, Left, Right, Stop)
  - Multi-Speed PWM Speed Regulation (Low, Medium, High, or Custom 0-100%)
  - Precision Sprayer Pump Control (ON, OFF, Burst Spray)
  - Automatic Precision Spray when Disease is Detected (Smart Agri-Mode)
  - Dual Platform Execution:
      * Laptop: Virtual Hardware Simulation with real-time UI telemetry.
      * Raspberry Pi: Hardware GPIO control via RPi.GPIO / PWM.
"""

import time
import threading
from src.config import (
    IS_RASPBERRY_PI,
    MOTOR_LEFT_ENA, MOTOR_LEFT_IN1, MOTOR_LEFT_IN2,
    MOTOR_RIGHT_ENB, MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4,
    PUMP_RELAY_PIN, STATUS_LED_PIN,
    SPEED_LOW, SPEED_MEDIUM, SPEED_HIGH, DEFAULT_SPEED,
    AUTO_SPRAY_BURST_SECONDS, AUTO_SPRAY_CONFIDENCE_THRESHOLD
)

class RoverController:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RoverController()
        return cls._instance

    def __init__(self):
        self.platform_name = "Raspberry Pi (Hardware GPIO)" if IS_RASPBERRY_PI else "Laptop (Virtual Simulation)"
        self.direction = "STOP"
        self.speed = DEFAULT_SPEED
        self.pump_active = False
        self.auto_spray_enabled = False
        self.last_spray_event = None
        self.spray_lock = threading.Lock()
        
        # Hardware specific handles
        self.gpio_available = False
        self.pwm_left = None
        self.pwm_right = None

        if IS_RASPBERRY_PI:
            self._init_raspberry_pi_gpio()
        else:
            print(f"[Rover Controller] Running in Simulation Mode on Laptop ({self.platform_name}).")

    def _init_raspberry_pi_gpio(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)

            # Motor Pins Setup
            for pin in [MOTOR_LEFT_IN1, MOTOR_LEFT_IN2, MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4, PUMP_RELAY_PIN, STATUS_LED_PIN]:
                self.GPIO.setup(pin, self.GPIO.OUT)
                self.GPIO.output(pin, self.GPIO.LOW)

            # PWM Pins Setup (100 Hz)
            self.GPIO.setup(MOTOR_LEFT_ENA, self.GPIO.OUT)
            self.GPIO.setup(MOTOR_RIGHT_ENB, self.GPIO.OUT)
            self.pwm_left = self.GPIO.PWM(MOTOR_LEFT_ENA, 100)
            self.pwm_right = self.GPIO.PWM(MOTOR_RIGHT_ENB, 100)
            self.pwm_left.start(self.speed)
            self.pwm_right.start(self.speed)

            self.gpio_available = True
            print("[Rover Controller] Raspberry Pi GPIO initialized successfully with PWM.")
        except Exception as e:
            print(f"[Rover Controller] Warning: Could not initialize RPi.GPIO ({e}). Falling back to simulation.")
            self.gpio_available = False

    def set_speed(self, speed_percent):
        """Sets motor speed (0 - 100%)"""
        self.speed = max(0, min(100, int(speed_percent)))
        if self.gpio_available and self.pwm_left and self.pwm_right:
            self.pwm_left.ChangeDutyCycle(self.speed)
            self.pwm_right.ChangeDutyCycle(self.speed)
        print(f"[Rover] Speed updated to {self.speed}%")
        return self.get_telemetry()

    def forward(self):
        self.direction = "FORWARD"
        if self.gpio_available:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.LOW)
        print(f"[Rover] Moving FORWARD at {self.speed}% speed")
        return self.get_telemetry()

    def backward(self):
        self.direction = "BACKWARD"
        if self.gpio_available:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.LOW)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.HIGH)
        print(f"[Rover] Moving BACKWARD at {self.speed}% speed")
        return self.get_telemetry()

    def turn_left(self):
        self.direction = "LEFT"
        if self.gpio_available:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.LOW)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.LOW)
        print(f"[Rover] Turning LEFT at {self.speed}% speed")
        return self.get_telemetry()

    def turn_right(self):
        self.direction = "RIGHT"
        if self.gpio_available:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.HIGH)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.HIGH)
        print(f"[Rover] Turning RIGHT at {self.speed}% speed")
        return self.get_telemetry()

    def stop(self):
        self.direction = "STOP"
        if self.gpio_available:
            self.GPIO.output(MOTOR_LEFT_IN1, self.GPIO.LOW)
            self.GPIO.output(MOTOR_LEFT_IN2, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN3, self.GPIO.LOW)
            self.GPIO.output(MOTOR_RIGHT_IN4, self.GPIO.LOW)
        print("[Rover] Motors STOPPED")
        return self.get_telemetry()

    def pump_on(self):
        self.pump_active = True
        if self.gpio_available:
            self.GPIO.output(PUMP_RELAY_PIN, self.GPIO.HIGH)
        print("[Rover] Sprayer Pump turned ON")
        return self.get_telemetry()

    def pump_off(self):
        self.pump_active = False
        if self.gpio_available:
            self.GPIO.output(PUMP_RELAY_PIN, self.GPIO.LOW)
        print("[Rover] Sprayer Pump turned OFF")
        return self.get_telemetry()

    def spray_burst(self, duration=AUTO_SPRAY_BURST_SECONDS):
        """Triggers a non-blocking timed spray burst."""
        def _burst():
            with self.spray_lock:
                self.pump_on()
                time.sleep(duration)
                self.pump_off()
                self.last_spray_event = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "duration_sec": duration
                }
        t = threading.Thread(target=_burst, daemon=True)
        t.start()
        return {"status": "burst_triggered", "duration": duration}

    def set_auto_spray(self, enabled):
        self.auto_spray_enabled = bool(enabled)
        print(f"[Rover] Smart Auto-Spray Mode set to: {self.auto_spray_enabled}")
        return self.get_telemetry()

    def handle_ai_disease_trigger(self, ai_result):
        """
        Called when AI detects a leaf. If auto-spray is ON and leaf is diseased,
        triggers precision spray burst.
        """
        if not self.auto_spray_enabled:
            return {"triggered": False, "reason": "auto_spray_disabled"}

        if not ai_result.get("is_leaf"):
            return {"triggered": False, "reason": "non_leaf"}

        if ai_result.get("is_healthy"):
            return {"triggered": False, "reason": "healthy_leaf"}

        conf = ai_result.get("confidence", 0)
        disease = ai_result.get("prediction", "Unknown")

        if conf >= AUTO_SPRAY_CONFIDENCE_THRESHOLD:
            # Trigger precision spray burst!
            print(f"[Smart Rover Action] Detected {disease} ({conf}%). Activating Sprayer Pump!")
            self.spray_burst(AUTO_SPRAY_BURST_SECONDS)
            return {
                "triggered": True,
                "disease": disease,
                "confidence": conf,
                "duration": AUTO_SPRAY_BURST_SECONDS
            }

        return {"triggered": False, "reason": f"confidence_too_low ({conf}%)"}

    def get_telemetry(self):
        """Returns the real-time rover and pump telemetry."""
        return {
            "platform": self.platform_name,
            "direction": self.direction,
            "speed": self.speed,
            "pump_active": self.pump_active,
            "auto_spray_enabled": self.auto_spray_enabled,
            "gpio_active": self.gpio_available,
            "last_spray": self.last_spray_event
        }

    def cleanup(self):
        self.stop()
        self.pump_off()
        if self.gpio_available:
            self.GPIO.cleanup()
