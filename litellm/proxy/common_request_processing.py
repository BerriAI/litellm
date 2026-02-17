import asyncio
import json
import logging
import traceback
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    Optional,
    Tuple,
    Union,
)

import httpx
import orjson
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE,
    STREAM_SSE_DATA_PREFIX,
)
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import check_response_size_is_safe
from litellm.proxy.common_utils.callback_utils import (
    get_logging_caching_headers,
    get_remaining_tokens_and_requests_from_request_data,
)
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.utils import ServerToolUse

# Type alias for streaming chunk serializer (chunk after hooks + cost injection -> wire format)
StreamChunkSerializer = Callable[[Any], str]
# Type alias for streaming error serializer (ProxyException -> wire format)
StreamErrorSerializer = Callable[[ProxyException], str]

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig as _ProxyConfig

    ProxyConfig = _ProxyConfig
else:
    ProxyConfig = Any
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage


async def _parse_event_data_for_error(event_line: Union[str, bytes]) -> Optional[int]:
    """Parses an event line and returns an error code if present, else None."""
    event_line = (
        event_line.decode("utf-8") if isinstance(event_line, bytes) else event_line
    )
    if event_line.startswith("data: "):
        json_str = event_line[len("data: ") :].strip()
        if not json_str or json_str == "[DONE]":  # handle empty data or [DONE] message
            return None
        try:
            data = orjson.loads(json_str)
            if (
                isinstance(data, dict)
                and "error" in data
                and isinstance(data["error"], dict)
            ):
                error_code_raw = data["error"].get("code")
                error_code: Optional[int] = None

                if isinstance(error_code_raw, int):
                    error_code = error_code_raw
                elif isinstance(error_code_raw, str):
                    try:
                        error_code = int(error_code_raw)
                    except ValueError:
                        verbose_proxy_logger.warning(
                            f"Error code is a string but not a valid integer: {error_code_raw}"
                        )
                        # Not a valid integer string, treat as if no valid code was found for this check
                        pass

                # Ensure error_code is a valid HTTP status code
                if error_code is not None and 100 <= error_code <= 599:
                    return error_code
                elif (
                    error_code_raw is not None
                ):  # Log if original code was present but not valid
                    verbose_proxy_logger.warning(
                        f"Error has invalid or non-convertible code: {error_code_raw}"
                    )
        except (orjson.JSONDecodeError, json.JSONDecodeError):
            # not a known error chunk
            pass
    return None


def _extract_error_from_sse_chunk(event_line: Union[str, bytes]) -> dict:
    """
    Extract error dictionary from SSE format chunk.

    Args:
        event_line: SSE format event line, e.g. "data: {"error": {...}}\n\n"

    Returns:
        Error dictionary in OpenAI API format
    """
    event_line = (
        event_line.decode("utf-8") if isinstance(event_line, bytes) else event_line
    )

    # Default error format
    default_error = {
        "message": "Unknown error",
        "type": "internal_server_error",
        "param": None,
        "code": "500",
    }

    if event_line.startswith("data: "):
        json_str = event_line[len("data: ") :].strip()
        if not json_str or json_str == "[DONE]":
            return default_error

        try:
            data = orjson.loads(json_str)
            if isinstance(data, dict) and "error" in data:
                error_obj = data["error"]
                if isinstance(error_obj, dict):
                    return error_obj
        except (orjson.JSONDecodeError, json.JSONDecodeError):
            pass

    return default_error


async def create_response(
    generator: AsyncGenerator[str, None],
    media_type: str,
    headers: dict,
    default_status_code: int = status.HTTP_200_OK,
) -> Union[StreamingResponse, JSONResponse]:
    """
    Create streaming response, checking if the first chunk is an error.
    If the first chunk is an error, return a standard JSON error response.
    Otherwise, return StreamingResponse and stream all content.
    """
    first_chunk_value: Optional[str] = None
    final_status_code = default_status_code

    try:
        # Handle coroutine that returns a generator
        if asyncio.iscoroutine(generator):
            generator = await generator

        # Now get the first chunk from the actual generator
        first_chunk_value = await generator.__anext__()

        if first_chunk_value is not None:
            try:
                error_code_from_chunk = await _parse_event_data_for_error(
                    first_chunk_value
                )
                if error_code_from_chunk is not None:
                    # First chunk is an error, stream hasn't really started yet
                    # Should return standard JSON error response instead of SSE format
                    final_status_code = error_code_from_chunk
                    verbose_proxy_logger.debug(
                        f"Error detected in first stream chunk. Returning JSON error response with status code: {final_status_code}"
                    )

                    # Parse error content
                    error_dict = _extract_error_from_sse_chunk(first_chunk_value)

                    # Consume and close generator (avoid resource leak)
                    try:
                        await generator.aclose()
                    except Exception:
                        pass

                    # Return JSON format error response
                    return JSONResponse(
                        status_code=final_status_code,
                        content={"error": error_dict},
                        headers=headers,
                    )
            except Exception as e:
                verbose_proxy_logger.debug(f"Error parsing first chunk value: {e}")

    except StopAsyncIteration:
        # Generator was empty. Default status
        async def empty_gen() -> AsyncGenerator[str, None]:
            if False:
                yield  # type: ignore

        return StreamingResponse(
            empty_gen(),
            media_type=media_type,
            headers=headers,
            status_code=default_status_code,
        )
    except Exception as e:
        # Unexpected error consuming first chunk.
        verbose_proxy_logger.exception(
            f"Error consuming first chunk from generator: {e}"
        )

        # Fallback to a generic error stream
        async def error_gen_message() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'error': {'message': 'Error processing stream start', 'code': status.HTTP_500_INTERNAL_SERVER_ERROR}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            error_gen_message(),
            media_type=media_type,
            headers=headers,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    async def combined_generator() -> AsyncGenerator[str, None]:
        if first_chunk_value is not None:
            with tracer.trace(DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE):
                yield first_chunk_value
        async for chunk in generator:
            with tracer.trace(DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE):
                yield chunk

    return StreamingResponse(
        combined_generator(),
        media_type=media_type,
        headers=headers,
        status_code=final_status_code,
    )


def _override_openai_response_model(
    *,
    response_obj: Any,
    requested_model: str,
    log_context: str,
) -> None:
    """
    Force the OpenAI-compatible `model` field in the response to match what the client requested.

    LiteLLM internally prefixes some provider/deployment model identifiers (e.g. `hosted_vllm/...`).
    That internal identifier should not be returned to clients in the OpenAI `model` field.

    Note: This is intentionally verbose. A model mismatch is a useful signal that an internal
    model identifier is being stamped/preserved somewhere in the request/response pipeline.
    We log mismatches as warnings (and then restamp to the client-requested value) so these
    paths stay observable for maintainers/operators without breaking client compatibility.

    Errors are reserved for cases where the proxy cannot read/override the response model field.

    Exception: If a fallback occurred (indicated by x-litellm-attempted-fallbacks header),
    we should preserve the actual model that was used (the fallback model) rather than
    overriding it with the originally requested model.
    """
    if not requested_model:
        return

    # Check if a fallback occurred - if so, preserve the actual model used
    hidden_params = getattr(response_obj, "_hidden_params", {}) or {}
    if isinstance(hidden_params, dict):
        fallback_headers = hidden_params.get("additional_headers", {}) or {}
        attempted_fallbacks = fallback_headers.get(
            "x-litellm-attempted-fallbacks", None
        )
        if attempted_fallbacks is not None and attempted_fallbacks > 0:
            # A fallback occurred - preserve the actual model that was used
            verbose_proxy_logger.debug(
                "%s: fallback detected (attempted_fallbacks=%d), preserving actual model used instead of overriding to requested model.",
                log_context,
                attempted_fallbacks,
            )
            return

    if isinstance(response_obj, dict):
        downstream_model = response_obj.get("model")
        if downstream_model != requested_model:
            verbose_proxy_logger.debug(
                "%s: response model mismatch - requested=%r downstream=%r. Overriding response['model'] to requested model.",
                log_context,
                requested_model,
                downstream_model,
            )
        response_obj["model"] = requested_model
        return

    if not hasattr(response_obj, "model"):
        verbose_proxy_logger.error(
            "%s: cannot override response model; missing `model` attribute. response_type=%s",
            log_context,
            type(response_obj),
        )
        return

    downstream_model = getattr(response_obj, "model", None)
    if downstream_model != requested_model:
        verbose_proxy_logger.debug(
            "%s: response model mismatch - requested=%r downstream=%r. Overriding response.model to requested model.",
            log_context,
            requested_model,
            downstream_model,
        )

    try:
        setattr(response_obj, "model", requested_model)
    except Exception as e:
        verbose_proxy_logger.error(
            "%s: failed to override response.model=%r on response_type=%s. error=%s",
            log_context,
            requested_model,
            type(response_obj),
            str(e),
            exc_info=True,
        )


def _get_cost_breakdown_from_logging_obj(
    litellm_logging_obj: Optional[LiteLLMLoggingObj],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract discount and margin information from logging object's cost breakdown.

    Returns:
        Tuple of (original_cost, discount_amount, margin_total_amount, margin_percent)
    """
    if not litellm_logging_obj or not hasattr(litellm_logging_obj, "cost_breakdown"):
        return None, None, None, None

    cost_breakdown = litellm_logging_obj.cost_breakdown
    if not cost_breakdown:
        return None, None, None, None

    original_cost = cost_breakdown.get("original_cost")
    discount_amount = cost_breakdown.get("discount_amount")
    margin_total_amount = cost_breakdown.get("margin_total_amount")
    margin_percent = cost_breakdown.get("margin_percent")

    return original_cost, discount_amount, margin_total_amount, margin_percent


class ProxyBaseLLMRequestProcessing:
    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def get_custom_headers(
        *,
        user_api_key_dict: UserAPIKeyAuth,
        call_id: Optional[str] = None,
        model_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        api_base: Optional[str] = None,
        version: Optional[str] = None,
        model_region: Optional[str] = None,
        response_cost: Optional[Union[float, str]] = None,
        hidden_params: Optional[dict] = None,
        fastest_response_batch_completion: Optional[bool] = None,
        request_data: Optional[dict] = {},
        timeout: Optional[Union[float, int, httpx.Timeout]] = None,
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = None,
        **kwargs,
    ) -> dict:
        exclude_values = {"", None, "None"}
        hidden_params = hidden_params or {}

        # Extract discount and margin info from cost_breakdown if available
        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(
            litellm_logging_obj=litellm_logging_obj
        )

        # Calculate updated spend for header (include current response_cost)
        current_spend = user_api_key_dict.spend or 0.0
        updated_spend = current_spend
        if response_cost is not None:
            try:
                # Convert response_cost to float if it's a string
                cost_value = (
                    float(response_cost)
                    if isinstance(response_cost, str)
                    else response_cost
                )
                if cost_value > 0:
                    updated_spend = current_spend + cost_value
            except (ValueError, TypeError):
                # If conversion fails, use original spend
                pass

        headers = {
            "x-litellm-call-id": call_id,
            "x-litellm-model-id": model_id,
            "x-litellm-cache-key": cache_key,
            "x-litellm-model-api-base": (
                api_base.split("?")[0] if api_base else None
            ),  # don't include query params, risk of leaking sensitive info
            "x-litellm-version": version,
            "x-litellm-model-region": model_region,
            "x-litellm-response-cost": str(response_cost),
            "x-litellm-response-cost-original": (
                str(original_cost) if original_cost is not None else None
            ),
            "x-litellm-response-cost-discount-amount": (
                str(discount_amount) if discount_amount is not None else None
            ),
            "x-litellm-response-cost-margin-amount": (
                str(margin_total_amount) if margin_total_amount is not None else None
            ),
            "x-litellm-response-cost-margin-percent": (
                str(margin_percent) if margin_percent is not None else None
            ),
            "x-litellm-key-tpm-limit": str(user_api_key_dict.tpm_limit),
            "x-litellm-key-rpm-limit": str(user_api_key_dict.rpm_limit),
            "x-litellm-key-max-budget": str(user_api_key_dict.max_budget),
            "x-litellm-key-spend": str(updated_spend),
            "x-litellm-response-duration-ms": str(
                hidden_params.get("_response_ms", None)
            ),
            "x-litellm-overhead-duration-ms": str(
                hidden_params.get("litellm_overhead_time_ms", None)
            ),
            "x-litellm-fastest_response_batch_completion": (
                str(fastest_response_batch_completion)
                if fastest_response_batch_completion is not None
                else None
            ),
            "x-litellm-timeout": str(timeout) if timeout is not None else None,
            **{k: str(v) for k, v in kwargs.items()},
        }
        if request_data:
            remaining_tokens_header = (
                get_remaining_tokens_and_requests_from_request_data(request_data)
            )
            headers.update(remaining_tokens_header)

            logging_caching_headers = get_logging_caching_headers(request_data)
            if logging_caching_headers:
                headers.update(logging_caching_headers)

        try:
            return {
                key: str(value)
                for key, value in headers.items()
                if value not in exclude_values
            }
        except Exception as e:
            verbose_proxy_logger.error(f"Error setting custom headers: {e}")
            return {}

    async def common_processing_pre_call_logic(
        self,
        request: Request,
        general_settings: dict,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: ProxyLogging,
        proxy_config: ProxyConfig,
        route_type: Literal[
            "acompletion",
            "aembedding",
            "aresponses",
            "_arealtime",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
            "acompact_responses",
            "acreate_batch",
            "aretrieve_batch",
            "alist_batches",
            "acancel_batch",
            "afile_content",
            "afile_retrieve",
            "afile_delete",
            "atext_completion",
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "alist_input_items",
            "aimage_edit",
            "agenerate_content",
            "agenerate_content_stream",
            "allm_passthrough_route",
            "avector_store_search",
            "avector_store_create",
            "avector_store_file_create",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_update",
            "avector_store_file_delete",
            "aocr",
            "asearch",
            "avideo_generation",
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "acreate_container",
            "alist_containers",
            "aingest",
            "aretrieve_container",
            "adelete_container",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "anthropic_messages",
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
            "asend_message",
            "call_mcp_tool",
            "acreate_eval",
            "alist_evals",
            "aget_eval",
            "aupdate_eval",
            "adelete_eval",
            "acancel_eval",
            "acreate_run",
            "alist_runs",
            "aget_run",
            "acancel_run",
            "adelete_run",
        ],
        version: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        model: Optional[str] = None,
        llm_router: Optional[Router] = None,
    ) -> Tuple[dict, LiteLLMLoggingObj]:
        start_time = datetime.now()  # start before calling guardrail hooks

        # Calculate request queue time if arrival_time is available
        # Use start_time.timestamp() to avoid extra time.time() call for better performance
        proxy_server_request = self.data.get("proxy_server_request", {})
        arrival_time = proxy_server_request.get("arrival_time")
        queue_time_seconds = None
        if arrival_time is not None:
            # Convert start_time (datetime) to timestamp for calculation
            processing_start_time = start_time.timestamp()
            queue_time_seconds = processing_start_time - arrival_time

        self.data = await add_litellm_data_to_request(
            data=self.data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # Store queue time in metadata after add_litellm_data_to_request to ensure it's preserved
        if queue_time_seconds is not None:
            from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

            _metadata_variable_name = _get_metadata_variable_name(request)
            if _metadata_variable_name not in self.data:
                self.data[_metadata_variable_name] = {}
            if not isinstance(self.data[_metadata_variable_name], dict):
                self.data[_metadata_variable_name] = {}
            self.data[_metadata_variable_name][
                "queue_time_seconds"
            ] = queue_time_seconds

        self.data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or self.data.get("model", None)  # default passed in http request
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            self.data["temperature"] = user_temperature
        if user_request_timeout:
            self.data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            self.data["max_tokens"] = user_max_tokens
        if user_api_base:
            self.data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if (
            isinstance(self.data["model"], str)
            and self.data["model"] in litellm.model_alias_map
        ):
            self.data["model"] = litellm.model_alias_map[self.data["model"]]

        # Check key-specific aliases
        if (
            isinstance(self.data["model"], str)
            and user_api_key_dict.aliases
            and isinstance(user_api_key_dict.aliases, dict)
            and self.data["model"] in user_api_key_dict.aliases
        ):
            self.data["model"] = user_api_key_dict.aliases[self.data["model"]]

        self.data["litellm_call_id"] = request.headers.get(
            "x-litellm-call-id", str(uuid.uuid4())
        )
        ### CALL HOOKS ### - modify/reject incoming data before calling the model

        ## LOGGING OBJECT ## - initialize logging object for logging success/failure events for call
        ## IMPORTANT Note: - initialize this before running pre-call checks. Ensures we log rejected requests to langfuse.
        logging_obj, self.data = litellm.utils.function_setup(
            original_function=route_type,
            rules_obj=litellm.utils.Rules(),
            start_time=start_time,
            **self.data,
        )

        self.data["litellm_logging_obj"] = logging_obj

        self.data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=self.data, call_type=route_type  # type: ignore
        )

        # Apply hierarchical router_settings (Key > Team)
        # Global router_settings are already on the Router object itself.
        if llm_router is not None and proxy_config is not None:
            from litellm.proxy.proxy_server import prisma_client

            router_settings = await proxy_config._get_hierarchical_router_settings(
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
            )

            # If router_settings found (from key or team), apply them
            # Pass settings as per-request overrides instead of creating a new Router
            # This avoids expensive Router instantiation on each request
            if router_settings is not None:
                self.data["router_settings_override"] = router_settings

        if "messages" in self.data and self.data["messages"]:
            logging_obj.update_messages(self.data["messages"])

        return self.data, logging_obj

    @staticmethod
    def _get_model_id_from_response(hidden_params: dict, data: dict) -> str:
        """Extract model_id from hidden_params with fallback to litellm_metadata."""
        model_id = hidden_params.get("model_id", None) or ""
        if not model_id:
            litellm_metadata = data.get("litellm_metadata", {}) or {}
            model_info = litellm_metadata.get("model_info", {}) or {}
            model_id = model_info.get("id", "") or ""
        return model_id

    async def base_process_llm_request(
        self,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        route_type: Literal[
            "acompletion",
            "aembedding",
            "aresponses",
            "_arealtime",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
            "acompact_responses",
            "atext_completion",
            "aimage_edit",
            "alist_input_items",
            "agenerate_content",
            "agenerate_content_stream",
            "allm_passthrough_route",
            "avector_store_search",
            "avector_store_create",
            "avector_store_file_create",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_update",
            "avector_store_file_delete",
            "aocr",
            "asearch",
            "avideo_generation",
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "acreate_container",
            "alist_containers",
            "aingest",
            "aretrieve_container",
            "adelete_container",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "anthropic_messages",
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
            "acancel_batch",
            "afile_delete",
            "acreate_eval",
            "alist_evals",
            "aget_eval",
            "aupdate_eval",
            "adelete_eval",
            "acancel_eval",
            "acreate_run",
            "alist_runs",
            "aget_run",
            "acancel_run",
            "adelete_run",
        ],
        proxy_logging_obj: ProxyLogging,
        general_settings: dict,
        proxy_config: ProxyConfig,
        select_data_generator: Optional[Callable] = None,
        llm_router: Optional[Router] = None,
        model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        version: Optional[str] = None,
        is_streaming_request: Optional[bool] = False,
        contents: Optional[list] = None,  # Add contents parameter
    ) -> Any:
        """
        Common request processing logic for both chat completions and responses API endpoints
        """
        requested_model_from_client: Optional[str] = (
            self.data.get("model") if isinstance(self.data.get("model"), str) else None
        )
        if verbose_proxy_logger.isEnabledFor(logging.DEBUG):
            verbose_proxy_logger.debug(
                "Request received by LiteLLM:\n{}".format(
                    json.dumps(self.data, indent=4, default=str)
                ),
            )

        self.data, logging_obj = await self.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            model=model,
            route_type=route_type,
            llm_router=llm_router,
        )

        tasks = []
        # Start the moderation check (during_call_hook) as early as possible
        # This gives it a head start to mask/validate input while the proxy handles routing
        tasks.append(
            asyncio.create_task(
                proxy_logging_obj.during_call_hook(
                    data=self.data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=route_type,  # type: ignore
                )
            )
        )

        # Pass contents if provided
        if contents:
            self.data["contents"] = contents

        ### ROUTE THE REQUEST ###
        # Do not change this - it should be a constant time fetch - ALWAYS
        llm_call = await route_request(
            data=self.data,
            route_type=route_type,
            llm_router=llm_router,
            user_model=user_model,
        )
        tasks.append(llm_call)

        # wait for call to end
        llm_responses = asyncio.gather(
            *tasks
        )  # run the moderation check in parallel to the actual llm api call

        responses = await llm_responses

        response = responses[1]

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = self._get_model_id_from_response(hidden_params, self.data)

        cache_key, api_base, response_cost = (
            hidden_params.get("cache_key", None) or "",
            hidden_params.get("api_base", None) or "",
            hidden_params.get("response_cost", None) or "",
        )
        fastest_response_batch_completion, additional_headers = (
            hidden_params.get("fastest_response_batch_completion", None),
            hidden_params.get("additional_headers", {}) or {},
        )

        # Post Call Processing
        if llm_router is not None:
            self.data["deployment"] = llm_router.get_deployment(model_id=model_id)
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=self.data.get("litellm_call_id", ""), status="success"
            )
        )
        if self._is_streaming_request(
            data=self.data, is_streaming_request=is_streaming_request
        ) or self._is_streaming_response(
            response
        ):  # use generate_responses to stream responses
            custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
                request_data=self.data,
                hidden_params=hidden_params,
                litellm_logging_obj=logging_obj,
                **additional_headers,
            )

            # Call response headers hook for streaming success
            callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
                data=self.data,
                user_api_key_dict=user_api_key_dict,
                response=response,
            )
            if callback_headers:
                custom_headers.update(callback_headers)

            # Preserve the original client-requested model (pre-alias mapping) for downstream
            # streaming generators. Pre-call processing can rewrite `self.data["model"]` for
            # aliasing/routing, but the OpenAI-compatible response `model` field should reflect
            # what the client sent.
            if requested_model_from_client:
                self.data["_litellm_client_requested_model"] = (
                    requested_model_from_client
                )
            if route_type == "allm_passthrough_route":
                # Check if response is an async generator
                if self._is_streaming_response(response):
                    if asyncio.iscoroutine(response):
                        generator = await response
                    else:
                        generator = response

                    # For passthrough routes, stream directly without error parsing
                    # since we're dealing with raw binary data (e.g., AWS event streams)
                    return StreamingResponse(
                        content=generator,
                        status_code=status.HTTP_200_OK,
                        headers=custom_headers,
                    )
                else:
                    # Traditional HTTP response with aiter_bytes
                    return StreamingResponse(
                        content=response.aiter_bytes(),
                        status_code=response.status_code,
                        headers=custom_headers,
                    )
            elif route_type == "anthropic_messages":
                # Check if response is actually a streaming response (async generator)
                # Non-streaming responses (dict) should be returned directly
                # This handles cases like websearch_interception agentic loop
                # which returns a non-streaming dict even for streaming requests
                if self._is_streaming_response(response):
                    selected_data_generator = (
                        ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                            response=response,
                            user_api_key_dict=user_api_key_dict,
                            request_data=self.data,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                    )
                    return await create_response(
                        generator=selected_data_generator,
                        media_type="text/event-stream",
                        headers=custom_headers,
                    )
                # Non-streaming response - fall through to normal response handling
            elif select_data_generator:
                selected_data_generator = select_data_generator(
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                    request_data=self.data,
                )
                return await create_response(
                    generator=selected_data_generator,
                    media_type="text/event-stream",
                    headers=custom_headers,
                )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=self.data, user_api_key_dict=user_api_key_dict, response=response
        )

        # Always return the client-requested model name (not provider-prefixed internal identifiers)
        # for OpenAI-compatible responses.
        if requested_model_from_client:
            _override_openai_response_model(
                response_obj=response,
                requested_model=requested_model_from_client,
                log_context=f"litellm_call_id={logging_obj.litellm_call_id}",
            )

        hidden_params = (
            getattr(response, "_hidden_params", {}) or {}
        )  # get any updated response headers
        additional_headers = hidden_params.get("additional_headers", {}) or {}

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
                request_data=self.data,
                hidden_params=hidden_params,
                litellm_logging_obj=logging_obj,
                **additional_headers,
            )
        )

        # Call response headers hook for non-streaming success
        callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
            data=self.data,
            user_api_key_dict=user_api_key_dict,
            response=response,
        )
        if callback_headers:
            fastapi_response.headers.update(callback_headers)

        await check_response_size_is_safe(response=response)

        return response

    async def base_passthrough_process_llm_request(
        self,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: ProxyLogging,
        general_settings: dict,
        proxy_config: ProxyConfig,
        select_data_generator: Callable,
        llm_router: Optional[Router] = None,
        model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        version: Optional[str] = None,
    ):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            HttpPassThroughEndpointHelpers,
        )

        result = await self.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="allm_passthrough_route",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        # Check if result is actually a streaming response by inspecting its type
        if isinstance(result, StreamingResponse):
            return result

        content = await result.aread()
        return Response(
            content=content,
            status_code=result.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                headers=result.headers,
                custom_headers=None,
            ),
        )

    def _is_streaming_response(self, response: Any) -> bool:
        """
        Check if the response object is actually a streaming response by inspecting its type.

        This uses standard Python inspection to detect streaming/async iterator objects
        rather than relying on specific wrapper classes.
        """
        import inspect
        from collections.abc import AsyncGenerator, AsyncIterator

        # Check if it's an async generator (most reliable)
        if inspect.isasyncgen(response):
            return True

        # Check if it implements the async iterator protocol
        if isinstance(response, (AsyncIterator, AsyncGenerator)):
            return True

        return False

    def _is_streaming_request(
        self, data: dict, is_streaming_request: Optional[bool] = False
    ) -> bool:
        """
        Check if the request is a streaming request.

        1. is_streaming_request is a dynamic param passed in
        2. if "stream" in data and data["stream"] is True
        """
        if is_streaming_request is True:
            return True
        if "stream" in data and data["stream"] is True:
            return True
        return False

    async def _handle_llm_api_exception(
        self,
        e: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: ProxyLogging,
        version: Optional[str] = None,
    ):
        """Raises ProxyException (OpenAI API compatible) if an exception is raised"""
        verbose_proxy_logger.exception(
            f"litellm.proxy.proxy_server._handle_llm_api_exception(): Exception occured - {str(e)}"
        )
        # Allow callbacks to transform the error response
        transformed_exception = await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=self.data,
        )
        # Use transformed exception if callback returned one, otherwise use original
        if transformed_exception is not None:
            e = transformed_exception
        litellm_debug_info = getattr(e, "litellm_debug_info", "")
        verbose_proxy_logger.debug(
            "\033[1;31mAn error occurred: %s %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
            litellm_debug_info,
        )

        timeout = getattr(
            e, "timeout", None
        )  # returns the timeout set by the wrapper. Used for testing if model-specific timeout are set correctly
        _litellm_logging_obj: Optional[LiteLLMLoggingObj] = self.data.get(
            "litellm_logging_obj", None
        )

        # Attempt to get model_id from logging object
        #
        # Note: We check the direct model_info path first (not nested in metadata) because that's where the router sets it.
        # The nested metadata path is only a fallback for cases where model_info wasn't set at the top level.
        model_id = self.maybe_get_model_id(_litellm_logging_obj)

        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=(
                _litellm_logging_obj.litellm_call_id if _litellm_logging_obj else None
            ),
            model_id=model_id,
            version=version,
            response_cost=0,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            request_data=self.data,
            timeout=timeout,
            litellm_logging_obj=_litellm_logging_obj,
        )
        # Extract headers from exception - check both e.headers and e.response.headers
        headers = getattr(e, "headers", None) or {}
        if not headers:
            # Try to get headers from e.response.headers (httpx.Response)
            _response = getattr(e, "response", None)
            if _response is not None:
                _response_headers = getattr(_response, "headers", None)
                if _response_headers:
                    headers = get_response_headers(dict(_response_headers))
        headers.update(custom_headers)

        # Call response headers hook for failure
        try:
            callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
                data=self.data,
                user_api_key_dict=user_api_key_dict,
                response=None,
            )
            if callback_headers:
                headers.update(callback_headers)
        except Exception:
            pass

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
                provider_specific_fields=getattr(e, "provider_specific_fields", None),
                headers=headers,
            )
        elif isinstance(e, httpx.HTTPStatusError):
            # Handle httpx.HTTPStatusError - extract actual error from response
            # This matches the original behavior before the refactor in commit 511d435f6f
            error_body = await e.response.aread()
            error_text = error_body.decode("utf-8")

            raise HTTPException(
                status_code=e.response.status_code,
                detail={"error": error_text},
            )
        error_msg = f"{str(e)}"
        # Check for AttributeError in various places:
        # 1. Direct AttributeError (already handled above)
        # 2. In underlying exception (__cause__, __context__, original_exception)
        has_attribute_error = (
            (
                isinstance(e, Exception)
                and isinstance(getattr(e, "__cause__", None), AttributeError)
            )
            or (
                isinstance(e, Exception)
                and isinstance(getattr(e, "__context__", None), AttributeError)
            )
            or (
                isinstance(e, Exception)
                and isinstance(getattr(e, "original_exception", None), AttributeError)
            )
        )

        if has_attribute_error:
            raise ProxyException(
                message=f"Invalid request format: {error_msg}",
                type="invalid_request_error",
                param=None,
                code=status.HTTP_400_BAD_REQUEST,
                headers=headers,
            )
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            openai_code=getattr(e, "code", None),
            code=getattr(e, "status_code", 500),
            provider_specific_fields=getattr(e, "provider_specific_fields", None),
            headers=headers,
        )

    #########################################################
    # Proxy Level Streaming Data Generator
    #########################################################

    @staticmethod
    def return_sse_chunk(chunk: Any) -> str:
        """
        Helper function to format streaming chunks for Anthropic API format

        Args:
            chunk: A string or dictionary to be returned in SSE format

        Returns:
            str: A properly formatted SSE chunk string
        """
        if isinstance(chunk, dict):
            # Use safe_dumps for proper JSON serialization with circular reference detection
            chunk_str = safe_dumps(chunk)
            return f"{STREAM_SSE_DATA_PREFIX}{chunk_str}\n\n"
        else:
            return chunk

    @staticmethod
    async def async_streaming_data_generator(
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,
        proxy_logging_obj: ProxyLogging,
        *,
        serialize_chunk: StreamChunkSerializer,
        serialize_error: StreamErrorSerializer,
    ) -> AsyncGenerator[str, None]:
        """
        Shared streaming data generator: runs proxy iterator hook, per-chunk hook,
        cost injection, then yields chunks via serialize_chunk; on exception runs
        failure hook and yields via serialize_error. Use for SSE or NDJSON.
        """
        verbose_proxy_logger.debug("inside generator")
        try:
            str_so_far = ""
            async for chunk in proxy_logging_obj.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=response,
                request_data=request_data,
            ):
                verbose_proxy_logger.debug(
                    "async_data_generator: received streaming chunk - {}".format(chunk)
                )
                chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=chunk,
                    data=request_data,
                    str_so_far=str_so_far,
                )

                if isinstance(chunk, (ModelResponse, ModelResponseStream)):
                    response_str = litellm.get_response_string(response_obj=chunk)
                    str_so_far += response_str
                elif hasattr(chunk, "model_dump"):
                    try:
                        d = chunk.model_dump(mode="json", exclude_none=True)
                        if isinstance(d, dict):
                            str_so_far += str(d.get("content", ""))
                    except Exception:
                        pass
                elif isinstance(chunk, dict):
                    str_so_far += str(chunk.get("content", ""))

                model_name = request_data.get("model", "")
                chunk = (
                    ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(
                        chunk, model_name
                    )
                )
                yield serialize_chunk(chunk)
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.async_data_generator(): Exception occured - {}".format(
                    str(e)
                )
            )
            transformed_exception = await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data=request_data,
            )
            if transformed_exception is not None:
                e = transformed_exception
            verbose_proxy_logger.debug(
                f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
            )

            if isinstance(e, HTTPException):
                raise e
            error_traceback = traceback.format_exc()
            error_msg = f"{str(e)}\n\n{error_traceback}"
            proxy_exception = ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )
            yield serialize_error(proxy_exception)

    @staticmethod
    async def async_sse_data_generator(
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,
        proxy_logging_obj: ProxyLogging,
    ) -> AsyncGenerator[str, None]:
        """
        Anthropic /messages and Google /generateContent streaming data generator require SSE events.
        Delegates to async_streaming_data_generator with SSE serializers.
        """
        async for chunk in ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
            proxy_logging_obj=proxy_logging_obj,
            serialize_chunk=ProxyBaseLLMRequestProcessing.return_sse_chunk,
            serialize_error=lambda proxy_exc: f"{STREAM_SSE_DATA_PREFIX}{json.dumps({'error': proxy_exc.to_dict()})}\n\n",
        ):
            yield chunk

    @staticmethod
    def _process_chunk_with_cost_injection(chunk: Any, model_name: str) -> Any:
        """
        Process a streaming chunk and inject cost information if enabled.

        Args:
            chunk: The streaming chunk (dict, str, bytes, or bytearray)
            model_name: Model name for cost calculation

        Returns:
            The processed chunk with cost information injected if applicable
        """
        if not getattr(litellm, "include_cost_in_streaming_usage", False):
            return chunk

        try:
            if isinstance(chunk, dict):
                maybe_modified = (
                    ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(
                        chunk, model_name
                    )
                )
                if maybe_modified is not None:
                    return maybe_modified
            elif isinstance(chunk, (bytes, bytearray)):
                # Decode to str, inject, and rebuild as bytes
                try:
                    s = chunk.decode("utf-8", errors="ignore")
                    maybe_mod = (
                        ProxyBaseLLMRequestProcessing._inject_cost_into_sse_frame_str(
                            s, model_name
                        )
                    )
                    if maybe_mod is not None:
                        return (
                            maybe_mod + ("" if maybe_mod.endswith("\n\n") else "\n\n")
                        ).encode("utf-8")
                except Exception:
                    pass
            elif isinstance(chunk, str):
                # Try to parse SSE frame and inject cost into the data line
                maybe_mod = (
                    ProxyBaseLLMRequestProcessing._inject_cost_into_sse_frame_str(
                        chunk, model_name
                    )
                )
                if maybe_mod is not None:
                    # Ensure trailing frame separator
                    return (
                        maybe_mod
                        if maybe_mod.endswith("\n\n")
                        else (maybe_mod + "\n\n")
                    )
        except Exception:
            # Never break streaming on optional cost injection
            pass

        return chunk

    @staticmethod
    def _inject_cost_into_sse_frame_str(
        frame_str: str, model_name: str
    ) -> Optional[str]:
        """
        Inject cost information into an SSE frame string by modifying the JSON in the 'data:' line.

        Args:
            frame_str: SSE frame string that may contain multiple lines
            model_name: Model name for cost calculation

        Returns:
            Modified SSE frame string with cost injected, or None if no modification needed
        """
        try:
            # Split preserving lines
            lines = frame_str.split("\n")
            for idx, ln in enumerate(lines):
                stripped_ln = ln.strip()
                if stripped_ln.startswith("data:"):
                    json_part = stripped_ln.split("data:", 1)[1].strip()
                    if json_part and json_part != "[DONE]":
                        obj = json.loads(json_part)
                        maybe_modified = (
                            ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(
                                obj, model_name
                            )
                        )
                        if maybe_modified is not None:
                            # Replace just this line with updated JSON using safe_dumps
                            lines[idx] = f"data: {safe_dumps(maybe_modified)}"
                            return "\n".join(lines)
            return None
        except Exception:
            return None

    @staticmethod
    def _inject_cost_into_usage_dict(obj: dict, model_name: str) -> Optional[dict]:
        """
        Inject cost information into a usage dictionary for message_delta events.

        Args:
            obj: Dictionary containing the SSE event data
            model_name: Model name for cost calculation

        Returns:
            Modified dictionary with cost injected, or None if no modification needed
        """
        if obj.get("type") == "message_delta" and isinstance(obj.get("usage"), dict):
            _usage = obj["usage"]
            prompt_tokens = int(_usage.get("input_tokens", 0) or 0)
            completion_tokens = int(_usage.get("output_tokens", 0) or 0)
            total_tokens = int(
                _usage.get("total_tokens", prompt_tokens + completion_tokens)
                or (prompt_tokens + completion_tokens)
            )

            # Extract additional usage fields
            cache_creation_input_tokens = _usage.get("cache_creation_input_tokens")
            cache_read_input_tokens = _usage.get("cache_read_input_tokens")
            web_search_requests = _usage.get("web_search_requests")
            completion_tokens_details = _usage.get("completion_tokens_details")
            prompt_tokens_details = _usage.get("prompt_tokens_details")

            usage_kwargs: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            # Add optional named parameters
            if completion_tokens_details is not None:
                usage_kwargs["completion_tokens_details"] = completion_tokens_details
            if prompt_tokens_details is not None:
                usage_kwargs["prompt_tokens_details"] = prompt_tokens_details

            # Handle web_search_requests by wrapping in ServerToolUse
            if web_search_requests is not None:
                usage_kwargs["server_tool_use"] = ServerToolUse(
                    web_search_requests=web_search_requests
                )

            # Add cache-related fields to **params (handled by Usage.__init__)
            if cache_creation_input_tokens is not None:
                usage_kwargs["cache_creation_input_tokens"] = (
                    cache_creation_input_tokens
                )
            if cache_read_input_tokens is not None:
                usage_kwargs["cache_read_input_tokens"] = cache_read_input_tokens

            _mr = ModelResponse(usage=Usage(**usage_kwargs))

            try:
                cost_val = litellm.completion_cost(
                    completion_response=_mr,
                    model=model_name,
                )
            except Exception:
                cost_val = None

            if cost_val is not None:
                obj.setdefault("usage", {})["cost"] = cost_val
                return obj
        return None

    def maybe_get_model_id(
        self, _logging_obj: Optional[LiteLLMLoggingObj]
    ) -> Optional[str]:
        """
        Get model_id from logging object or request metadata.

        The router sets model_info.id when selecting a deployment. This tries multiple locations
        where the ID might be stored depending on the request lifecycle stage.
        """
        model_id = None
        if _logging_obj:
            # 1. Try getting from litellm_params (updated during call)
            if hasattr(_logging_obj, "litellm_params") and _logging_obj.litellm_params:
                # First check direct model_info path (set by router.py with selected deployment)
                model_info = _logging_obj.litellm_params.get("model_info") or {}
                model_id = model_info.get("id", None)

                # Fallback to nested metadata path
                if not model_id:
                    metadata = _logging_obj.litellm_params.get("metadata") or {}
                    model_info = metadata.get("model_info") or {}
                    model_id = model_info.get("id", None)

            # 2. Fallback to kwargs (initial)
            if not model_id:
                _kwargs = getattr(_logging_obj, "kwargs", None)
                if _kwargs:
                    litellm_params = _kwargs.get("litellm_params", {})
                    # First check direct model_info path
                    model_info = litellm_params.get("model_info") or {}
                    model_id = model_info.get("id", None)

                    # Fallback to nested metadata path
                    if not model_id:
                        metadata = litellm_params.get("metadata") or {}
                        model_info = metadata.get("model_info") or {}
                        model_id = model_info.get("id", None)

        # 3. Final fallback to self.data["litellm_metadata"] (for routes like /v1/responses that populate data before error)
        if not model_id:
            litellm_metadata = self.data.get("litellm_metadata", {}) or {}
            model_info = litellm_metadata.get("model_info", {}) or {}
            model_id = model_info.get("id", None)

        return model_id
