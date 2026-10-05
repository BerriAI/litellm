from typing import Final

from pydantic import StrictStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger

_STRINGS: Final[TypeAdapter[tuple[str, ...]]] = TypeAdapter(tuple[StrictStr, ...])


def config_strings(
    raw: object, *, option_name: str, fallback: str = "Nothing is skipped for this option"
) -> tuple[str, ...]:
    if raw is None:
        return ()
    try:
        return _STRINGS.validate_python(raw)
    except ValidationError:
        verbose_proxy_logger.warning("Ignoring %s=%r, expected a list of strings. %s", option_name, raw, fallback)
        return ()
