from typing import Any, Optional, Union

from pydantic import BaseModel

from litellm.types.utils import HiddenParams


def add_retry_headers_to_response(
    response: Any,
    attempted_retries: int,
    max_retries: Optional[int] = None,
) -> Any:
    """
    Add retry headers to the request
    """

    if response is None or not isinstance(response, BaseModel):
        return response

    retry_headers = {
        "x-litellm-attempted-retries": attempted_retries,
    }
    if max_retries is not None:
        retry_headers["x-litellm-max-retries"] = max_retries

    hidden_params: Optional[Union[dict, HiddenParams]] = getattr(
        response, "_hidden_params", {}
    )

    if hidden_params is None:
        hidden_params = {}
    elif isinstance(hidden_params, HiddenParams):
        hidden_params = hidden_params.model_dump()

    hidden_params.setdefault("additional_headers", {})
    hidden_params["additional_headers"].update(retry_headers)

    setattr(response, "_hidden_params", hidden_params)

    return response
