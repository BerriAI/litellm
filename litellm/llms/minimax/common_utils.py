from collections.abc import Mapping

import litellm
from litellm.exceptions import UnsupportedParamsError


def image_params_without_stream(params: Mapping[str, object], drop_params: object) -> Mapping[str, object]:
    if params.get("stream") and litellm.drop_params is not True and drop_params is not True:
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                "Setting `stream` is not supported by MiniMax image endpoints. "
                "To drop it from the call, set `drop_params: true`."
            ),
        )
    return {key: value for key, value in params.items() if key != "stream"}
