import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


def _get_fallback_data_path() -> Path:
    configured_path = os.getenv("PLANE_DATA_FILE")
    if configured_path:
        return Path(configured_path).expanduser()

    return Path(__file__).resolve().with_name("resposta.json")


def _load_fallback_planes():
    data_path = _get_fallback_data_path()
    if not data_path.exists():
        return []

    try:
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        print(f"Unable to read fallback plane data from {data_path}: {exc}")
        return []

    return payload if isinstance(payload, list) else []


def get_lista():
    api_url = os.getenv("SUPABASE_URL")
    if not api_url:
        return _load_fallback_planes()

    api_key = os.getenv("api_key")
    authorization = os.getenv("authorization")

    headers = {}
    if api_key:
        headers["apikey"] = api_key
    if authorization:
        headers["Authorization"] = f"Bearer {authorization}"

    try:
        response = requests.get(f"{api_url}/rest/v1/trajectory", headers=headers, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Unable to fetch remote plane data from Supabase, using fallback data: {exc}")
        return _load_fallback_planes()

    return payload if isinstance(payload, list) else []
