import re
import unicodedata
from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.bedrock import AwsSessionTag

_SESSION_NAME: Final = re.compile(r"[\w+=,.@-]{2,64}", re.ASCII)
_CALLER_SUPPLIED_FIELDS: Final = frozenset({"user_api_key_end_user_id"})


def _identity_value(params: Mapping[str, object], field: str) -> str | None:
    if not field.startswith("user_api_key_") or field in _CALLER_SUPPLIED_FIELDS:
        return None
    sources: Final = tuple(
        source for name in ("metadata", "litellm_metadata") if (source := as_str_mapping(params.get(name))) is not None
    )
    values: Final = (source.get(field) for source in sources)
    return next((value for value in values if isinstance(value, str) and value), None)


def _tag_text(value: str, limit: int) -> bool:
    return 0 < len(value) <= limit and all(
        char in "_.:/=+-@" or unicodedata.category(char)[0] in "LZN" for char in value
    )


def _derived_tag(params: Mapping[str, object], field: str) -> AwsSessionTag | None:
    value: Final = _identity_value(params, field)
    if value is None or not _tag_text(field, 128) or not _tag_text(value, 256):
        return None
    return AwsSessionTag(Key=field, Value=value)


def resolve_bedrock_session_identity(
    params: Mapping[str, object] | None,
    static_tags: tuple[AwsSessionTag, ...] | None,
    session_name: str | None,
) -> tuple[tuple[AwsSessionTag, ...] | None, str | None]:
    identity: Final = params or {}
    configured: Final = tuple(litellm.bedrock_session_tag_fields or ())
    folded: Final = tuple(field.lower() for field in configured)
    fields: Final = tuple(field for index, field in enumerate(configured) if folded[index] not in folded[:index])
    static_keys: Final = frozenset(tag["Key"].lower() for tag in static_tags or ())
    candidates: Final = (_derived_tag(identity, field) for field in fields if field.lower() not in static_keys)
    derived: Final = tuple(tag for tag in candidates if tag is not None)[: max(0, 50 - len(static_tags or ()))]
    tags: Final = (*static_tags, *derived) if static_tags is not None else derived or None
    names: Final = (_identity_value(identity, field) for field in litellm.bedrock_session_name_fields or ())
    name: Final = next((value for value in names if value is not None and _SESSION_NAME.fullmatch(value)), None)
    return tags, session_name if session_name is not None else name
