import os

# Ensure required env defaults for production
os.environ.setdefault('FLASK_DEBUG', '0')

from app import app as application  # WSGI entrypoint

# Optional: eager-load model at import time in some hosts
try:
    # Access attribute to trigger before_first_request registration
    _ = application.name
except Exception:
    pass


