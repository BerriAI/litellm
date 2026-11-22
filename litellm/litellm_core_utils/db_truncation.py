import re
from typing import Final

from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE

_SKIPPED_CHARS: Final = rf"\.\.\. \({re.escape(LITELLM_TRUNCATED_PAYLOAD_FIELD)} skipped \d+ chars"
_SAFEGUARD_NOTE: Final = rf"\. {re.escape(LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE)}"
TRUNCATION_MARKER: Final = re.compile(rf"{_SKIPPED_CHARS}{_SAFEGUARD_NOTE}\) \.\.\.")
_MARKER_OF_ANY_RELEASE: Final = re.compile(rf"{_SKIPPED_CHARS}(?:{_SAFEGUARD_NOTE})?\) \.\.\.")


def _kept_chars(max_length: int) -> tuple[int, int]:
    start_chars: Final = int(max_length * 0.35)
    return start_chars, min(int(max_length * 0.65), max_length - start_chars)


def truncated_for_db(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    start_chars, end_chars = _kept_chars(max_length)
    return (
        f"{value[:start_chars]}"
        f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {len(value) - start_chars - end_chars} chars. "
        f"{LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE}) ..."
        f"{value[-end_chars:]}"
    )


def _is_where_the_value_was_cut(marker: re.Match[str]) -> bool:
    kept: Final = (marker.start(), len(marker.string) - marker.end())
    return any(_kept_chars(max_length) == kept for max_length in (sum(kept), sum(kept) + 1))


def _ends_kept_by_a_cut(value: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (value[: marker.start()], value[marker.end() :])
        for marker in _MARKER_OF_ANY_RELEASE.finditer(value)
        if _is_where_the_value_was_cut(marker)
    )


def was_truncated_for_db(value: str) -> bool:
    return bool(_ends_kept_by_a_cut(value))


def is_truncation_of(stored: str, whole: str) -> bool:
    return any(whole.startswith(start) and whole.endswith(end) for start, end in _ends_kept_by_a_cut(stored))
