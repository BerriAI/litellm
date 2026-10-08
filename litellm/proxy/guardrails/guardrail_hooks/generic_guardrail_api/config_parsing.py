import re
from typing import Final

from pydantic import StrictStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger

_STRINGS: Final[TypeAdapter[tuple[str, ...]]] = TypeAdapter(tuple[StrictStr, ...])


def config_strings(raw: object, *, option_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    try:
        return _STRINGS.validate_python(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring %s=%r, expected a list of strings. Nothing is skipped for this option", option_name, raw
        )
        return ()


def config_patterns(raw: object, *, option_name: str) -> tuple[re.Pattern[str], ...]:
    sources: Final = config_strings(raw, option_name=option_name)
    try:
        patterns: Final = tuple(re.compile(source) for source in sources)
    except re.error as e:
        verbose_proxy_logger.warning(
            "Ignoring %s=%r, it contains an invalid regex (%s). Nothing is skipped for this option",
            option_name,
            raw,
            e,
        )
        return ()
    if any(pattern.search("") is not None for pattern in patterns):
        verbose_proxy_logger.warning(
            "Ignoring %s=%r, a pattern matches an empty string, so it would also match text that was never searched. "
            "Nothing is skipped for this option",
            option_name,
            raw,
        )
        return ()
    return patterns
