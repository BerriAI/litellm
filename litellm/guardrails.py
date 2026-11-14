"""Invoke configured gateway guardrails without generating a completion."""

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared factory has untyped optional parameters
)
from litellm.types.guardrails import NativeGuardrailResponse

_NATIVE_RESPONSE: Final[TypeAdapter[NativeGuardrailResponse]] = TypeAdapter(NativeGuardrailResponse)


class _InputType(Enum):
    REQUEST = "request"
    RESPONSE = "response"


async def apply(
    guardrail_name: str,
    *,
    input: str | None = None,
    response: str | None = None,
    messages: Sequence[Mapping[str, object]] | None = None,
    metadata: Mapping[str, object] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> NativeGuardrailResponse:
    """Return a configured guardrail's native structured findings.

    Supply exactly one of ``input`` or ``response``. Grounding guardrails can
    use ``messages`` as evidence. Connection settings default to
    ``litellm.api_base`` and ``litellm.api_key``.
    """
    if (input is None) == (response is None):
        raise ValueError("Supply exactly one of input or response.")
    resolved_base: Final = api_base if api_base is not None else litellm.api_base
    if not resolved_base:
        raise ValueError("Set api_base or litellm.api_base to your LiteLLM gateway URL.")
    resolved_key: Final = api_key if api_key is not None else litellm.api_key
    input_type: Final = _InputType.RESPONSE if response is not None else _InputType.REQUEST
    payload: Final = dict[str, object](
        guardrail_name=guardrail_name,
        text=response if response is not None else input,
        input_type=input_type.value,
        response_format="native",
    )
    if messages is not None:
        payload["messages"] = [dict(message) for message in messages]
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    gateway_root: Final = resolved_base.rstrip("/").removesuffix("/v1")
    client: Final = get_async_httpx_client(llm_provider="litellm_guardrails")
    result: Final = await client.post(  # pyright: ignore[reportUnknownMemberType]  # shared handler has untyped optional parameters
        url=f"{gateway_root}/guardrails/apply_guardrail",
        json=payload,
        headers={"Authorization": f"Bearer {resolved_key}"} if resolved_key else {},
        timeout=timeout,
    )
    return _NATIVE_RESPONSE.validate_json(result.content)
