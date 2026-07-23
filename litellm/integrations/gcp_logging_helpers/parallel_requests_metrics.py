"""
Helpers for the stdout-backed parallel request metrics consumed from GCP logs.

These log lines are parsed back as metrics, so user-controlled fields must be
kept on one physical line and away from the comma delimiters used by the parser.
"""

import re
from datetime import datetime
from typing import Any, Dict, Literal, Optional
from urllib.parse import quote, unquote

_METRIC_FIELD_SAFE_CHARS = "-._~"

METRICS_LOG_PATTERN = re.compile(
    r"^\[METRICS\] Emitting parallel_requests metric: "
    r"token=(?P<token>[^,\r\n]+), "
    r"key_alias=(?P<key_alias>[^,\r\n]+), "
    r"previous_count=(?P<previous_count>-?\d+), "
    r"current_count=(?P<current_count>-?\d+), "
    r"operation=(?P<operation>increment|decrement), "
    r"timestamp=(?P<timestamp>[^,\r\n]+)$"
)


def encode_metric_field(value: Optional[Any]) -> str:
    if value is None:
        return "None"
    return quote(str(value), safe=_METRIC_FIELD_SAFE_CHARS)


def decode_metric_field(value: str) -> str:
    return unquote(value)


def decode_optional_metric_field(value: str) -> Optional[str]:
    if value == "None":
        return None
    return unquote(value)


def build_parallel_requests_metric_log_line(
    *,
    token: Optional[Any],
    key_alias: Optional[Any],
    previous_count: int,
    current_count: int,
    operation: Literal["increment", "decrement"],
    timestamp: Any,
) -> str:
    return (
        "[METRICS] Emitting parallel_requests metric: "
        f"token={encode_metric_field(token)}, "
        f"key_alias={encode_metric_field(key_alias)}, "
        f"previous_count={int(previous_count)}, "
        f"current_count={int(current_count)}, "
        f"operation={operation}, "
        f"timestamp={encode_metric_field(timestamp)}"
    )


def _parse_metric_timestamp(value: str) -> float:
    decoded_value = unquote(value)
    try:
        return float(decoded_value)
    except ValueError:
        return datetime.fromisoformat(decoded_value.replace("Z", "+00:00")).timestamp()


def parse_parallel_requests_metric_log_line(text_payload: str) -> Optional[Dict[str, Any]]:
    match = METRICS_LOG_PATTERN.fullmatch(text_payload.strip())
    if not match:
        return None

    try:
        return {
            "token": decode_metric_field(match.group("token")),
            "key_alias": decode_optional_metric_field(match.group("key_alias")),
            "previous_count": int(match.group("previous_count")),
            "current_count": int(match.group("current_count")),
            "operation": match.group("operation"),
            "timestamp": _parse_metric_timestamp(match.group("timestamp")),
        }
    except (ValueError, IndexError):
        return None
