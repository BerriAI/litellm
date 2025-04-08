import asyncio
import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Tuple, Union

import httpx
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
        route_type: Literal["acompletion", "aresponses", "_arealtime"],
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

        ### CALL HOOKS ### - modify/reject incoming data before calling the model
        self.data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=self.data, call_type="completion"
        )

        ## LOGGING OBJECT ## - initialize logging object for logging success/failure events for call
        ## IMPORTANT Note: - initialize this before running pre-call checks. Ensures we log rejected requests to langfuse.
        self.data["litellm_call_id"] = request.headers.get(
            "x-litellm-call-id", str(uuid.uuid4())
        )
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
        route_type: Literal["acompletion", "aresponses", "_arealtime"],
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
            "Request received by LiteLLM:\n{}".format(json.dumps(self.data, indent=4)),
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
            return StreamingResponse(
                selected_data_generator,
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
        route_type: Literal["acompletion", "aresponses"]
    ) -> Literal["completion", "responses"]:
        if route_type == "acompletion":
            return "completion"
        elif route_type == "aresponses":
            return "responses"
