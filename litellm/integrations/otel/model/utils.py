"""Shared, OpenTelemetry-free helpers for the otel integration.

Generic value coercion (for reading heterogeneous logging-payload dicts), time
conversion, and header parsing — pulled out of the individual modules so they
live in one place. Deliberately free of any ``opentelemetry`` import so the
OTel-free sources of truth (payloads, semconv, spans, config) can use it too.
"""

from datetime import datetime


def as_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def as_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def as_str_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return None


def to_ns(value: datetime | float | int | None) -> int | None:
    """Coerce a datetime / epoch value to integer nanoseconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1e9)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(float(value) * 1e9)
    return None


def to_seconds(value: datetime | float | int | str | None) -> float | None:
    """Coerce a datetime / epoch / formatted-string value to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
    return None


def parse_headers(raw: str | None) -> dict[str, str]:
    """Parse an OTLP ``"k=v,k=v"`` header string into a dict."""
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for pair in raw.split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            headers[key.strip()] = value.strip()
    return headers
