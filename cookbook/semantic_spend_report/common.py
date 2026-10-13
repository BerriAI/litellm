from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, cast

type JSON = None | bool | int | float | str | list[JSON] | dict[str, JSON]

ROOT: Final = Path(__file__).resolve().parent
ARTIFACTS: Final = Path(os.environ.get("POC_OUTPUT_DIR", str(ROOT / "artifacts"))).resolve()
BASE_URL: Final = os.environ.get("GATEWAY_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
MODEL: Final = os.environ.get("REPORT_MODEL", "vertex-sonnet-5-5")
JEV_MODEL: Final = os.environ.get("JEV_MODEL", "typesafe/jev-latest")


def object_value(value: JSON) -> dict[str, JSON]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def array_value(value: JSON) -> list[JSON]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return value


def string_value(value: JSON) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a JSON string")
    return value


def load(path: Path) -> JSON:
    return cast(JSON, json.loads(path.read_text()))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Final = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Gateway:
    key: str

    def request(self, path: str, body: dict[str, JSON] | None = None, session: str = "") -> dict[str, JSON]:
        headers: Final = {
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
            **({"x-litellm-session-id": session, "x-litellm-tags": "report-generation"} if session else {}),
        }
        request: Final = urllib.request.Request(
            BASE_URL + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data: Final = cast(JSON, json.load(response))
                return {
                    "data": data,
                    "http_status": response.status,
                    "call_id": response.headers.get("x-litellm-call-id", ""),
                    "response_cost": response.headers.get("x-litellm-response-cost", ""),
                }
        except urllib.error.HTTPError as error:
            message: Final = error.read().decode(errors="replace").replace(self.key, "[REDACTED]")[:1500]
            raise RuntimeError(f"Gateway HTTP {error.code} at {path.split('?')[0]} {message}") from None

    def get(self, path: str) -> dict[str, JSON]:
        return object_value(self.request(path)["data"])


def gateway() -> Gateway:
    return Gateway(os.environ["LITELLM_API_KEY"])
