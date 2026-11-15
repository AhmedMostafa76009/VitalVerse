"""Application configuration and filesystem paths for Vitalverse."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
UPLOADS_DIR = PROJECT_ROOT / "uploads_private"

DB_PATH = Path(os.environ.get("ASTRO_HEALTH_DB") or (DATA_DIR / "astronaut_health_app.db"))
MODEL_PATH = MODELS_DIR / "health_model.joblib"
SCHEMA_PATH = PROJECT_ROOT / "data_raw" / "team_supplied" / "07_health_indicators_schema.json"
DEMO_CREDS_PATH = CONFIG_DIR / "demo_credentials.json"

SESSION_COOKIE = "vv_session"
CSRF_HEADER = "X-CSRF-Token"
COOKIE_SECURE = os.environ.get("VITALVERSE_COOKIE_SECURE", "0") == "1"


def load_demo_credentials() -> dict[str, Any]:
    """Load demo auth configuration and Argon2id parameters."""
    if not DEMO_CREDS_PATH.is_file():
        return {}
    return json.loads(DEMO_CREDS_PATH.read_text(encoding="utf-8"))
