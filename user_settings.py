"""
user_settings.py — Persist user-level settings across app restarts.

Currently stores the user's API key for the Neural Copilot. Mirrors the
watchlist pattern: a single JSON file in the project root, exempted from
the `*.json` gitignore rule so it ships with the user's local install.

On Streamlit Community Cloud the file lives in the container's ephemeral
disk, same caveat as `watchlist.json` — survives session, resets on
container recycle. That's acceptable: a recycled container asking for the
key once is no worse than a fresh clone.
"""
import json
import os
from threading import Lock
from typing import Optional

SETTINGS_FILE = "user_settings.json"
_lock = Lock()

# Whitelisted keys we persist. Anything else is silently dropped on save.
KNOWN_KEYS = {"api_key", "api_provider"}


def _load() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, SETTINGS_FILE)


def get_api_key() -> Optional[str]:
    """Return the persisted API key, or None if not set."""
    with _lock:
        data = _load()
    return data.get("api_key") or None


def set_api_key(key: str) -> None:
    """Persist the API key. Strips whitespace; refuses empty strings."""
    key = (key or "").strip()
    with _lock:
        data = _load()
        if key:
            data["api_key"] = key
        else:
            data.pop("api_key", None)
        _save(data)


def clear_api_key() -> None:
    """Forget the persisted key (used by the 'Forget key' button)."""
    with _lock:
        data = _load()
        data.pop("api_key", None)
        _save(data)
