"""Opt-in copy of the ``x-litellm-call-id`` response header inside JSON error bodies.

``general_settings.include_call_id_in_error_body: true`` turns it on. Callers hand in the id they already
set on the response header, so the body value and the header value are byte-identical. ``JSON_OBJECT``
types the untyped error dicts the emission sites already hold before they reach ``with_call_id``.
"""

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

INCLUDE_CALL_ID_IN_ERROR_BODY_SETTING: Final = "include_call_id_in_error_body"
LITELLM_CALL_ID_BODY_KEY: Final = "litellm_call_id"
JSON_OBJECT: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])  # mutable-ok: JSONResponse input


def error_body_call_id(general_settings: Mapping[str, object], call_id: str | None) -> str | None:
    if general_settings.get(INCLUDE_CALL_ID_IN_ERROR_BODY_SETTING) is not True:
        return None
    return call_id if call_id else None


def with_call_id(error: dict[str, object], call_id: str | None) -> dict[str, object]:  # mutable-ok: JSONResponse input
    if call_id is None:
        return error
    return {**error, LITELLM_CALL_ID_BODY_KEY: call_id}  # mutable-ok: JSONResponse input
