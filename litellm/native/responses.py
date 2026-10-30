"""Sends a Responses API request to a provider that speaks Responses natively."""

from collections.abc import Collection, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.litellm_core_utils.completion_timeout import CompletionTimeout, StreamKind
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.data_residency import infer_openai_data_residency
from litellm.responses.litellm_completion_transformation.compaction_items import (
    with_compaction_items_as_summaries,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    with_compaction_items_as_params,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CostPerToken

if TYPE_CHECKING:
    from aiohttp import ClientSession

__all__ = ("execute_responses",)


class _DeploymentModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: object = None


_DEPLOYMENT_MODEL_INFO: Final[TypeAdapter[_DeploymentModelInfo | object]] = TypeAdapter(
    Annotated[_DeploymentModelInfo | object, Field(union_mode="left_to_right")]
)


def _is_router_deployment(deployment_model_info: object) -> bool:
    """A Router stamps the deployment's model_info (with its id) onto every call it routes."""
    parsed: Final = _DEPLOYMENT_MODEL_INFO.validate_python(deployment_model_info)
    return isinstance(parsed, _DeploymentModelInfo) and parsed.id is not None


_COST_PER_TOKEN: Final = TypeAdapter(CostPerToken)


def _per_call_cost_per_token(litellm_params: GenericLiteLLMParams) -> CostPerToken | None:
    if litellm_params.input_cost_per_token is None or litellm_params.output_cost_per_token is None:
        return None
    return _COST_PER_TOKEN.validate_python(
        litellm_params.model_dump(
            include={
                "input_cost_per_token",
                "output_cost_per_token",
                "cache_read_input_token_cost",
                "cache_creation_input_token_cost",
            },
            exclude_none=True,
        )
    )


def execute_responses(
    *,
    http_handler: BaseLLMHTTPHandler,
    model: str,
    input: str | ResponseInputParam,
    provider_config: BaseResponsesAPIConfig,
    requested_params: ResponsesAPIOptionalRequestParams,
    allowed_openai_params: Sequence[str] | None,
    declared_reasoning_efforts: Collection[str],
    deployment_model_info: object,
    custom_llm_provider: str,
    litellm_params: GenericLiteLLMParams,
    logging_obj: LiteLLMLoggingObj,
    user: str | None,
    extra_headers: dict[str, object] | None,
    extra_body: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    is_async: bool,
    stream: bool | None,
    client: HTTPHandler | AsyncHTTPHandler | None,
    shared_session: "ClientSession | None",
    litellm_metadata: dict[str, object] | None,
    request_kwargs: dict[str, object],
) -> (
    ResponsesAPIResponse
    | BaseResponsesAPIStreamingIterator
    | Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
):
    request_params: Final = ResponsesAPIRequestUtils.get_optional_params_responses_api(
        model=model,
        responses_api_provider_config=provider_config,
        response_api_optional_params=requested_params,
        allowed_openai_params=allowed_openai_params,
        drop_params=normalize_drop_params(request_kwargs.get("drop_params")),
        declared_reasoning_efforts=declared_reasoning_efforts,
    )

    logging_obj.update_from_kwargs(
        kwargs=request_kwargs,
        model=model,
        user=user,
        optional_params=dict(request_params),
        litellm_params={
            **request_params,
            **litellm_params.custom_pricing_overrides(),
            "aresponses": is_async,
            "litellm_call_id": request_kwargs.get("litellm_call_id"),
            "model_info": request_kwargs.get("model_info"),
            "data_residency": infer_openai_data_residency(custom_llm_provider, litellm_params.api_base),
            "metadata": (
                request_kwargs["litellm_metadata"]
                if "litellm_metadata" in request_kwargs
                else request_kwargs.get("metadata")
            ),
            "provider_affinity_header": litellm_params.provider_affinity_header,
        },
        custom_llm_provider=custom_llm_provider,
    )
    if not _is_router_deployment(deployment_model_info):
        logging_obj.custom_cost_per_token = _per_call_cost_per_token(litellm_params)  # rebind-ok: prices the call
        logging_obj.custom_cost_per_second = litellm_params.cost_per_second  # rebind-ok: prices the call

    fake_stream: Final = provider_config.should_fake_stream(
        model=model, stream=stream, custom_llm_provider=custom_llm_provider
    )
    applied_timeout: Final = CompletionTimeout.for_stream(
        (StreamKind.FAKED if fake_stream else StreamKind.LIVE) if stream is True else StreamKind.NOT_STREAMED,
        CompletionTimeout.as_limit(timeout),
        None if stream is not True or litellm_params.stream_timeout is None else float(litellm_params.stream_timeout),
    )
    logging_obj.record_applied_timeout(applied_timeout)
    response: Final = http_handler.response_api_handler(
        model=model,
        input=with_compaction_items_as_summaries(
            with_compaction_items_as_params(ResponsesAPIRequestUtils.restore_encrypted_content_item_ids_in_input(input))
        ),
        responses_api_provider_config=provider_config,
        response_api_optional_request_params=request_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
        extra_headers=extra_headers,
        extra_body=extra_body,
        timeout=applied_timeout,
        _is_async=is_async,
        client=client,
        fake_stream=fake_stream,
        litellm_metadata=litellm_metadata,
        shared_session=shared_session,
    )
    if isinstance(response, Coroutine):
        return _stamped_when_done(response, litellm_metadata, custom_llm_provider)
    return _stamped(response, litellm_metadata, custom_llm_provider)


def _stamped(
    response: ResponsesAPIResponse | BaseResponsesAPIStreamingIterator,
    litellm_metadata: Mapping[str, object] | None,
    custom_llm_provider: str,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    if not isinstance(response, ResponsesAPIResponse):
        return response
    stamped: Final = ResponsesAPIRequestUtils.update_responses_api_response_id_with_model_id(
        responses_api_response=response,
        litellm_metadata=litellm_metadata,
        custom_llm_provider=custom_llm_provider,
    )
    stamped.hidden_params["custom_llm_provider"] = custom_llm_provider
    return stamped


async def _stamped_when_done(
    response: Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator],
    litellm_metadata: Mapping[str, object] | None,
    custom_llm_provider: str,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    return _stamped(await response, litellm_metadata, custom_llm_provider)
