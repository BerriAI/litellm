import configparser
import os
import time
import uuid
from typing import Any, Dict, Final, List, Optional, Tuple

CONFIG_FILE_PATH_DEFAULT: Final[str] = "~/.opik.config"


def create_uuid7() -> str:
    """Generate an RFC 9562 conformant UUIDv7 string.

    The top 48 bits encode the Unix timestamp in milliseconds. Opik's backend
    validates this embedded timestamp on ingestion (it must fall within a window
    around "now"), so the encoding has to be correct or trace/span batches are
    rejected with HTTP 400. Implemented with the standard library only, so no
    extra dependency is added to litellm. See ``opik.id_helpers`` for the
    reference implementation.
    """
    unix_ts_ms = int(time.time() * 1000)

    # Fill the 16-byte buffer with random data, then overwrite the structured
    # parts (timestamp, version, variant) defined by the UUIDv7 layout.
    uuid_bytes = bytearray(os.urandom(16))

    # First 48 bits (6 bytes): Unix timestamp in milliseconds.
    uuid_bytes[0:6] = unix_ts_ms.to_bytes(6, byteorder="big")

    # Version 7 in the top 4 bits of byte 6.
    uuid_bytes[6] = 0x70 | (uuid_bytes[6] & 0x0F)

    # Variant 0b10 in the top 2 bits of byte 8.
    uuid_bytes[8] = 0x80 | (uuid_bytes[8] & 0x3F)

    return str(uuid.UUID(bytes=bytes(uuid_bytes)))


def _read_opik_config_file() -> Dict[str, str]:
    config_path = os.path.expanduser(CONFIG_FILE_PATH_DEFAULT)

    config = configparser.ConfigParser()
    config.read(config_path)

    config_values = {section: dict(config.items(section)) for section in config.sections()}

    if "opik" in config_values:
        return config_values["opik"]

    return {}


def _get_env_variable(key: str) -> Optional[str]:
    env_prefix = "opik_"
    return os.getenv((env_prefix + key).upper(), None)


def get_opik_config_variable(
    key: str, user_value: Optional[str] = None, default_value: Optional[str] = None
) -> Optional[str]:
    """
    Get the configuration value of a variable, order priority is:
    1. user provided value
    2. environment variable
    3. Opik configuration file
    4. default value
    """
    # Return user provided value if it is not None
    if user_value is not None:
        return user_value

    # Return environment variable if it is not None
    env_value = _get_env_variable(key)
    if env_value is not None:
        return env_value

    # Return value from Opik configuration file if it is not None
    config_values = _read_opik_config_file()

    if key in config_values:
        return config_values[key]

    # Return default value if it is not None
    return default_value


def create_usage_object(usage):
    usage_dict = {}

    if usage.completion_tokens is not None:
        usage_dict["completion_tokens"] = usage.completion_tokens
    if usage.prompt_tokens is not None:
        usage_dict["prompt_tokens"] = usage.prompt_tokens
    if usage.total_tokens is not None:
        usage_dict["total_tokens"] = usage.total_tokens
    return usage_dict


def _remove_nulls(x: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from dict."""
    return {k: v for k, v in x.items() if v is not None}


def get_traces_and_spans_from_payload(
    payload: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate traces and spans from payload.

    Traces are identified by not having a "type" field.
    Spans are identified by having a "type" field.

    Args:
        payload: List of dicts containing trace and span data

    Returns:
        Tuple of (traces, spans) where both are lists of dicts with null values removed
    """
    traces = [_remove_nulls(x) for x in payload if "type" not in x]
    spans = [_remove_nulls(x) for x in payload if "type" in x]
    return traces, spans
