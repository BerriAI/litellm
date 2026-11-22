import re
from typing import Final

from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE

_SKIPPED: Final = rf"\.\.\. \({re.escape(LITELLM_TRUNCATED_PAYLOAD_FIELD)} skipped "
_SAFEGUARD_NOTE: Final = rf"\. {re.escape(LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE)}"
TRUNCATION_MARKER: Final = re.compile(rf"{_SKIPPED}\d+ chars{_SAFEGUARD_NOTE}\) \.\.\.")
_MARKER_OF_ANY_RELEASE: Final = re.compile(rf"{_SKIPPED}(?P<skipped>\d+) chars(?:{_SAFEGUARD_NOTE})?\) \.\.\.")


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
    return _kept_chars(sum(kept) + 1) == kept


def _cuts(value: str) -> tuple[re.Match[str], ...]:
    return tuple(filter(_is_where_the_value_was_cut, _MARKER_OF_ANY_RELEASE.finditer(value)))


def was_truncated_for_db(value: str) -> bool:
    return bool(_cuts(value))


def _is_cut_from(cut: re.Match[str], whole: str) -> bool:
    start: Final = cut.string[: cut.start()]
    end: Final = cut.string[cut.end() :]
    skipped: Final = str(len(whole) - len(start) - len(end))
    return cut["skipped"] == skipped and whole.startswith(start) and whole.endswith(end)


def is_truncation_of(stored: str, whole: str) -> bool:
    return any(_is_cut_from(cut, whole) for cut in _cuts(stored))
