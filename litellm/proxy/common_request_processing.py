import asyncio
import json
import uuid
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
from fastapi.responses import Response, StreamingResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import check_response_size_is_safe
from litellm.proxy.common_utils.callback_utils import (
    get_logging_caching_headers,
    get_remaining_tokens_and_requests_from_request_data,
)
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig as _ProxyConfig

    ProxyConfig = _ProxyConfig
else:
    ProxyConfig = Any
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request


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


async def create_streaming_response(
    generator: AsyncGenerator[str, None],
    media_type: str,
    headers: dict,
    default_status_code: int = status.HTTP_200_OK,
) -> StreamingResponse:
    """
    Creates a StreamingResponse by inspecting the first chunk for an error code.
    The entire original generator content is streamed, but the HTTP status code
    of the response is set based on the first chunk if it's a recognized error.
    """
    first_chunk_value: Optional[str] = None
    final_status_code = default_status_code

    try:
        first_chunk_value = await generator.__anext__()
        if first_chunk_value is not None:
            error_code_from_chunk = await _parse_event_data_for_error(first_chunk_value)
            if error_code_from_chunk is not None:
                final_status_code = error_code_from_chunk
                verbose_proxy_logger.debug(
                    f"Error detected in first stream chunk. Status code set to: {final_status_code}"
                )

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
            yield first_chunk_value
        async for chunk in generator:
            yield chunk

    return StreamingResponse(
        combined_generator(),
        media_type=media_type,
        headers=headers,
        status_code=final_status_code,
    )


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
        **kwargs,
    ) -> dict:
        exclude_values = {"", None, "None"}
        hidden_params = hidden_params or {}
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
            "x-litellm-key-tpm-limit": str(user_api_key_dict.tpm_limit),
            "x-litellm-key-rpm-limit": str(user_api_key_dict.rpm_limit),
            "x-litellm-key-max-budget": str(user_api_key_dict.max_budget),
            "x-litellm-key-spend": str(user_api_key_dict.spend),
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
            "aresponses",
            "_arealtime",
            "aget_responses",
            "adelete_responses",
            "acreate_batch",
            "aretrieve_batch",
            "afile_content",
            "atext_completion",
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "aimage_edit",
        ],
        version: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[dict, LiteLLMLoggingObj]:
        self.data = await add_litellm_data_to_request(
            data=self.data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

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

        self.data["litellm_call_id"] = request.headers.get(
            "x-litellm-call-id", str(uuid.uuid4())
        )
        ### CALL HOOKS ### - modify/reject incoming data before calling the model
        self.data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=self.data, call_type=route_type  # type: ignore
        )

        ## LOGGING OBJECT ## - initialize logging object for logging success/failure events for call
        ## IMPORTANT Note: - initialize this before running pre-call checks. Ensures we log rejected requests to langfuse.
        logging_obj, self.data = litellm.utils.function_setup(
            original_function=route_type,
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.now(),
            **self.data,
        )

        self.data["litellm_logging_obj"] = logging_obj

        return self.data, logging_obj

    async def base_process_llm_request(
        self,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        route_type: Literal[
            "acompletion",
            "aresponses",
            "_arealtime",
            "aget_responses",
            "adelete_responses",
            "atext_completion",
            "aimage_edit",
        ],
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
    ) -> Any:
        """
        Common request processing logic for both chat completions and responses API endpoints
        """
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
        )

        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=self.data,
                user_api_key_dict=user_api_key_dict,
                call_type=ProxyBaseLLMRequestProcessing._get_pre_call_type(
                    route_type=route_type  # type: ignore
                ),
            )
        )

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
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        fastest_response_batch_completion = hidden_params.get(
            "fastest_response_batch_completion", None
        )
        additional_headers: dict = hidden_params.get("additional_headers", {}) or {}

        # Post Call Processing
        if llm_router is not None:
            self.data["deployment"] = llm_router.get_deployment(model_id=model_id)
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=self.data.get("litellm_call_id", ""), status="success"
            )
        )
        if (
            "stream" in self.data and self.data["stream"] is True
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
                **additional_headers,
            )
            selected_data_generator = select_data_generator(
                response=response,
                user_api_key_dict=user_api_key_dict,
                request_data=self.data,
            )
            return await create_streaming_response(
                generator=selected_data_generator,
                media_type="text/event-stream",
                headers=custom_headers,
            )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=self.data, user_api_key_dict=user_api_key_dict, response=response
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
                **additional_headers,
            )
        )
        await check_response_size_is_safe(response=response)

        return response

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
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=self.data,
        )
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
        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=(
                _litellm_logging_obj.litellm_call_id if _litellm_logging_obj else None
            ),
            version=version,
            response_cost=0,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            request_data=self.data,
            timeout=timeout,
        )
        headers = getattr(e, "headers", {}) or {}
        headers.update(custom_headers)

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
                headers=headers,
            )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            openai_code=getattr(e, "code", None),
            code=getattr(e, "status_code", 500),
            headers=headers,
        )

    @staticmethod
    def _get_pre_call_type(
        route_type: Literal["acompletion", "aresponses"],
    ) -> Literal["completion", "responses"]:
        if route_type == "acompletion":
            return "completion"
        elif route_type == "aresponses":
            return "responses"
