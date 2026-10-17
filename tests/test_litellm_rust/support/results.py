from collections.abc import AsyncIterator, Iterator
from typing import Final

from pydantic import BaseModel, TypeAdapter

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.types.llms.openai import ResponsesAPIResponse


def response_id(response: object) -> object:
    if isinstance(response, ResponsesAPIResponse):
        return ResponsesAPIRequestUtils.decode_responses_api_response_id(response.id).get("response_id") or response.id
    return response.get("id") if isinstance(response, dict) else None


def with_id(response: object, new_id: str) -> object:
    if isinstance(response, BaseModel):
        return response.model_copy(update={"id": new_id})
    assert isinstance(response, dict)
    return {**response, "id": new_id}


async def consume(stream: object) -> tuple[object, ...]:
    if isinstance(stream, AsyncIterator):
        return tuple([chunk async for chunk in stream])
    assert isinstance(stream, Iterator)
    return tuple(stream)


def cache_key(response: object) -> object:
    headers: Final = TypeAdapter(dict[str, object]).validate_python(
        get_hidden_params_dict(response).get("additional_headers", {})
    )
    return headers.get("x-litellm-cache-key")
