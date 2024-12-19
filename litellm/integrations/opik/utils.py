import configparser
import os
import time
from typing import Dict, Final, List, Optional

CONFIG_FILE_PATH_DEFAULT: Final[str] = "~/.opik.config"


def create_uuid7():
    ns = time.time_ns()
    last = [0, 0, 0, 0]

    # Simple uuid7 implementation
    sixteen_secs = 16_000_000_000
    t1, rest1 = divmod(ns, sixteen_secs)
    t2, rest2 = divmod(rest1 << 16, sixteen_secs)
    t3, _ = divmod(rest2 << 12, sixteen_secs)
    t3 |= 7 << 12  # Put uuid version in top 4 bits, which are 0 in t3

    # The next two bytes are an int (t4) with two bits for
    # the variant 2 and a 14 bit sequence counter which increments
    # if the time is unchanged.
    if t1 == last[0] and t2 == last[1] and t3 == last[2]:
        # Stop the seq counter wrapping past 0x3FFF.
        # This won't happen in practice, but if it does,
        # uuids after the 16383rd with that same timestamp
        # will not longer be correctly ordered but
        # are still unique due to the 6 random bytes.
        if last[3] < 0x3FFF:
            last[3] += 1
    else:
        last[:] = (t1, t2, t3, 0)
    t4 = (2 << 14) | last[3]  # Put variant 0b10 in top two bits

    # Six random bytes for the lower part of the uuid
    rand = os.urandom(6)
    return f"{t1:>08x}-{t2:>04x}-{t3:>04x}-{t4:>04x}-{rand.hex()}"


def _read_opik_config_file() -> Dict[str, str]:
    config_path = os.path.expanduser(CONFIG_FILE_PATH_DEFAULT)

    config = configparser.ConfigParser()
    config.read(config_path)

    config_values = {
        section: dict(config.items(section)) for section in config.sections()
    }

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


def _remove_nulls(x):
    x_ = {k: v for k, v in x.items() if v is not None}
    return x_


def get_traces_and_spans_from_payload(payload: List):
    traces = [_remove_nulls(x) for x in payload if "type" not in x]
    spans = [_remove_nulls(x) for x in payload if "type" in x]
    return traces, spans
