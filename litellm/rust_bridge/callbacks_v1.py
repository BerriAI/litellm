"""The v1 contract functions the native call reaches for, by name.

`python_contract.json` pins these four on both sides. The typed authoring surface — the
callback protocols, the envelope and patch types, and the registry — is
`litellm.callbacks_v1`; `snapshot` is re-exported from there because the registry has one
home, and the rest here is marshalling no callback author calls.
"""

import json
import uuid
from typing import Final, TypeGuard  # noqa: TID251  # recursive JSON validation

from litellm.callbacks_v1 import JSONValue, snapshot

__all__ = ("new_call_id", "project_response", "report", "snapshot")


def _is_json_value(
    value: object,
) -> TypeGuard[JSONValue]:  # guard-ok: recursively validates the complete JSON value
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(
            _is_json_value(item)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # list items are validated recursively
            for item in value  # pyright: ignore[reportUnknownVariableType]  # list items are validated recursively
        )
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # mapping entries are validated recursively
            for key, item in value.items()  # pyright: ignore[reportUnknownVariableType]  # mapping entries are validated recursively
        )
    return False


def project_response(response: object) -> JSONValue:
    model_dump: Final = getattr(response, "model_dump", None)
    projected: Final = model_dump(mode="json") if callable(model_dump) else response
    json.dumps(projected, allow_nan=False)
    if _is_json_value(projected):
        return projected
    raise TypeError(f"unsupported response type: {type(response).__qualname__}")


def new_call_id() -> str:
    return str(uuid.uuid4())


def report(name: str, event: str, error: BaseException) -> None:
    from litellm._logging import verbose_logger

    exception: Final = (type(error), error, error.__traceback__)
    verbose_logger.exception("callback %s failed while handling %s", name, event, exc_info=exception)
