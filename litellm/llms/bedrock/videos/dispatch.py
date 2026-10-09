"""Bedrock dispatch shims for the main-layer video functions.

The generic video routes in ``litellm.videos.main`` delegate their bedrock
branches here (AWS SigV4 signing and the async-invoke API need the bedrock
handler instead of the shared HTTP handler). Each function lazy-imports
``BedrockVideoGeneration`` so importing this module never pulls the boto3
signing stack, and forwards its arguments verbatim.
"""

from __future__ import annotations

from collections.abc import Coroutine, Mapping
from typing import TYPE_CHECKING, Final

import httpx

from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.types.videos.main import VideoObject


def dispatch_bedrock_video_generation(
    *,
    model: str,
    prompt: str,
    video_generation_request_params: Mapping[str, object],
    litellm_params: GenericLiteLLMParams,
    logging_obj: LiteLLMLogging | None,
    timeout: float | httpx.Timeout | None,
    is_async: bool,
    client: object | None = None,
    extra_headers: dict[str, object] | None = None,
    api_key: str | None = None,
) -> VideoObject | Coroutine[object, object, VideoObject]:
    """Create (StartAsyncInvoke) through the bedrock handler.

    Merges the ``aws_*`` auth params riding on litellm_params into the optional
    params for the handler (mirrors how images/main.py merges non_default_params
    for bedrock) and threads the real litellm_params through so
    metadata.request_id reaches the clientRequestToken fallback.
    """
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration

    bedrock_optional_params: Final[dict[str, object]] = (
        dict(  # mutable-ok: aws_* params are merged in before the handler call
            video_generation_request_params
        )
    )
    bedrock_optional_params.update(
        {  # mutable-ok: aws_* auth params merged into the bedrock params
            k: v for k, v in litellm_params.model_dump(exclude_none=True).items() if k.startswith("aws_")
        }
    )
    return BedrockVideoGeneration().video_generation(
        model=model,
        prompt=prompt,
        optional_params=bedrock_optional_params,
        logging_obj=logging_obj,
        timeout=timeout,
        avideo_generation=is_async,
        client=client,
        api_base=litellm_params.get("api_base"),
        extra_headers=extra_headers,
        api_key=api_key,
        # Real litellm_params so metadata.request_id reaches the
        # clientRequestToken fallback (idempotent retries); aws_* keys are
        # already merged into bedrock_optional_params above and are never
        # consumed from this object by the handler.
        litellm_params=litellm_params,
    )


def dispatch_bedrock_video_status(
    *,
    video_id: str,
    litellm_params: GenericLiteLLMParams,
    logging_obj: LiteLLMLogging | None,
    api_base: str | None,
    api_key: str | None,
    astatus: bool,
    timeout: float | httpx.Timeout | None,
) -> VideoObject | Coroutine[object, object, VideoObject]:
    """Status (GetAsyncInvoke) through the bedrock handler."""
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration

    return BedrockVideoGeneration().video_status(
        video_id=video_id,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        api_base=api_base,
        api_key=api_key,
        astatus=astatus,
        timeout=timeout,
    )


def dispatch_bedrock_video_content(
    *,
    video_id: str,
    litellm_params: GenericLiteLLMParams,
    logging_obj: LiteLLMLogging | None,
    api_base: str | None,
    api_key: str | None,
    timeout: float | httpx.Timeout | None,
) -> bytes:
    """Content download (S3 output object) through the bedrock handler."""
    from litellm.llms.bedrock.videos.handler import BedrockVideoGeneration

    return BedrockVideoGeneration().video_content(
        video_id=video_id,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
    )
