from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError

_STREAM_FLAG: Final = TypeAdapter(bool)


def _asks_to_stream(value: object) -> bool:
    if value is None:
        return False
    try:
        return _STREAM_FLAG.validate_python(value)
    except ValidationError:
        return True


def image_params_without_stream(params: Mapping[str, object], drop_params: object) -> Mapping[str, object]:
    if _asks_to_stream(params.get("stream")) and litellm.drop_params is not True and drop_params is not True:
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                "Setting `stream` is not supported by MiniMax image endpoints. "
                "To drop it from the call, set `drop_params: true`."
            ),
        )
    return {key: value for key, value in params.items() if key != "stream"}
