import os

# Gunicorn configuration for Render deployment
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"
workers = 1
threads = 1
timeout = 180  # 3 minutes for AI inference operations
graceful_timeout = 30
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
preload_app = False
accesslog = "-"
errorlog = "-"
loglevel = "info"
