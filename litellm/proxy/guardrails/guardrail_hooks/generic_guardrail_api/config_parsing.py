import re
from collections.abc import Sequence


def config_values(raw: Sequence[str] | None, *, option_name: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        raise ValueError(f"{option_name} must be a list of strings, got the single string {raw!r}")
    return tuple(raw or ())


def compile_patterns(raw: Sequence[str] | None, *, option_name: str) -> tuple[re.Pattern[str], ...]:
    try:
        return tuple(re.compile(pattern) for pattern in config_values(raw, option_name=option_name))
    except re.error as e:
        raise ValueError(f"{option_name} contains an invalid regex: {e}") from e
