from collections.abc import Mapping
from typing import Literal

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.passthrough.main import (
    aclassification_passthrough_request,
    classification_passthrough_request,
)
from litellm.types.classification import (
    ClassificationInput,
    ClassificationRequest,
    ClassificationResponse,
)


def _classification_request_body(
    *,
    model: str,
    input: ClassificationInput,
    user: str | None,
    truncate_prompt_tokens: int | None,
    truncation_side: Literal["left", "right"] | None,
    priority: int,
    add_special_tokens: bool,
    request_id: str | None,
    use_activation: bool | None,
    extra_body: Mapping[str, object] | None,
) -> dict[str, object]:
    request = ClassificationRequest(
        model=model,
        input=input,
        user=user,
        truncate_prompt_tokens=truncate_prompt_tokens,
        truncation_side=truncation_side,
        priority=priority,
        add_special_tokens=add_special_tokens,
        request_id=request_id,
        use_activation=use_activation,
    ).model_dump(exclude_none=True)
    return {**request, **dict(extra_body or {})}


def _classification_response(response: httpx.Response) -> ClassificationResponse:
    return ClassificationResponse.model_validate(response.json())


def classify(
    *,
    model: str,
    input: ClassificationInput,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    user: str | None = None,
    truncate_prompt_tokens: int | None = None,
    truncation_side: Literal["left", "right"] | None = None,
    priority: int = 0,
    add_special_tokens: bool = True,
    request_id: str | None = None,
    use_activation: bool | None = None,
    extra_body: Mapping[str, object] | None = None,
    request_headers: dict[str, str] | None = None,
    client: HTTPHandler | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> ClassificationResponse:
    response = classification_passthrough_request(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
        request_headers=request_headers,
        json=_classification_request_body(
            model=model,
            input=input,
            user=user,
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
            priority=priority,
            add_special_tokens=add_special_tokens,
            request_id=request_id,
            use_activation=use_activation,
            extra_body=extra_body,
        ),
        client=client,
        timeout=timeout,
    )
    return _classification_response(response)


async def aclassify(
    *,
    model: str,
    input: ClassificationInput,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    user: str | None = None,
    truncate_prompt_tokens: int | None = None,
    truncation_side: Literal["left", "right"] | None = None,
    priority: int = 0,
    add_special_tokens: bool = True,
    request_id: str | None = None,
    use_activation: bool | None = None,
    extra_body: Mapping[str, object] | None = None,
    request_headers: dict[str, str] | None = None,
    client: AsyncHTTPHandler | None = None,
    timeout: float | httpx.Timeout | None = None,
) -> ClassificationResponse:
    response = await aclassification_passthrough_request(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
        request_headers=request_headers,
        json=_classification_request_body(
            model=model,
            input=input,
            user=user,
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
            priority=priority,
            add_special_tokens=add_special_tokens,
            request_id=request_id,
            use_activation=use_activation,
            extra_body=extra_body,
        ),
        client=client,
        timeout=timeout,
    )
    return _classification_response(response)
