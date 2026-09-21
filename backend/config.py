import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIGNATURE_UPLOAD_DIR = os.path.join(BASE_DIR, "signatures", "uploads")
TEMP_VERIFY_DIR = os.path.join(BASE_DIR, "signatures", "temp")

os.makedirs(SIGNATURE_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_VERIFY_DIR, exist_ok=True)