import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import (
    LiteLLMBatch,
    LlmProviders,
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any

from abc import ABC, abstractmethod


class BasePassthroughLoggingHandler(ABC):
    @property
    @abstractmethod
    def llm_provider_name(self) -> LlmProviders:
        pass

    @abstractmethod
    def get_provider_config(self, model: str) -> BaseConfig:
        pass

    def passthrough_chat_handler(
        self,
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Transforms LLM response to OpenAI response, generates a standard logging object so downstream logging can be handled
        """
        model = request_body.get("model", response_body.get("model", ""))
        provider_config = self.get_provider_config(model=model)
        litellm_model_response: ModelResponse = provider_config.transform_response(
            raw_response=httpx_response,
            model_response=litellm.ModelResponse(),
            model=model,
            messages=[],
            logging_obj=logging_obj,
            optional_params={},
            api_key="",
            request_data={},
            encoding=litellm.encoding,
            json_mode=False,
            litellm_params={},
        )

        kwargs = self._create_response_logging_payload(
            litellm_model_response=litellm_model_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        return {
            "result": litellm_model_response,
            "kwargs": kwargs,
        }

    def _get_user_from_metadata(
        self,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return get_end_user_id_from_request_body(request_body)
        return None

    def _create_response_logging_payload(
        self,
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
    ) -> dict:
        """
        Create the standard logging object for Generic LLM passthrough

        handles streaming and non-streaming responses
        """

        try:
            response_cost = litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model,
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            # the pass-through success path reads spend from
            # model_call_details["response_cost"], not from kwargs
            logging_obj.model_call_details["response_cost"] = response_cost
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (  # type: ignore
                kwargs.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = self._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs.setdefault("litellm_params", {})
                    kwargs["litellm_params"].update({"proxy_server_request": {"body": {"user": user}}})

            # Make standard logging object for Anthropic
            standard_logging_object = get_standard_logging_object_payload(
                kwargs=kwargs,
                init_response_obj=litellm_model_response,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                status="success",
            )

            # pretty print standard logging object
            verbose_proxy_logger.debug(
                "standard_logging_object= %s",
                json.dumps(standard_logging_object, indent=4),
            )
            kwargs["standard_logging_object"] = standard_logging_object

            # set litellm_call_id to logging response object
            litellm_model_response.id = logging_obj.litellm_call_id
            litellm_model_response.model = model
            logging_obj.model_call_details["model"] = model
            return kwargs
        except Exception as e:
            verbose_proxy_logger.exception("Error creating LLM passthrough response logging payload: %s", e)
            return kwargs

    @abstractmethod
    def _build_complete_streaming_response(
        self,
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw chunks

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        pass

    def _handle_logging_llm_collected_chunks(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Anthropic passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """

        model = request_body.get("model", "")
        complete_streaming_response = self._build_complete_streaming_response(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
        )
        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Anthropic passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": {},
            }
        kwargs = self._create_response_logging_payload(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs={},
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }


def get_actual_model_id_from_router(model_name: str) -> str:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is not None:
        model_ids = llm_router.get_model_ids(model_name=model_name)
        if model_ids:
            return model_ids[0]
    return model_name


def try_get_proxy_model_id_from_router(model_name: str) -> Optional[str]:
    """Return a proxy deployment id when ``model_name`` maps to a configured model."""
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return None

    model_ids = llm_router.get_model_ids(model_name=model_name)
    if model_ids:
        return model_ids[0]

    proxy_model_name = llm_router.resolve_model_name_from_model_id(model_name)
    if proxy_model_name:
        model_ids = llm_router.get_model_ids(model_name=proxy_model_name)
        if model_ids:
            return model_ids[0]

    return None


def resolve_proxy_model_from_batch_input_file(
    input_file_id: str,
    custom_llm_provider: str,
    litellm_params: Optional[dict] = None,
) -> Optional[str]:
    """Resolve a proxy deployment id from models named in a batch input JSONL file."""
    try:
        from litellm.batches.batch_utils import (
            _get_file_content_as_dictionary,
            _get_models_from_batch_input_file_content,
        )
        from litellm.files.main import file_content
        from litellm.proxy.proxy_server import llm_router, passthrough_endpoint_router

        if llm_router is None:
            return None

        file_content_kwargs: dict = {
            "file_id": input_file_id,
            "custom_llm_provider": custom_llm_provider,
        }

        litellm_params = litellm_params or {}
        api_key = litellm_params.get("api_key")
        if not api_key and passthrough_endpoint_router is not None:
            api_key = passthrough_endpoint_router.get_credentials(
                custom_llm_provider=custom_llm_provider,
                region_name=None,
            )
        if api_key:
            file_content_kwargs["api_key"] = api_key

        for key in ("api_base", "api_version", "organization"):
            if litellm_params.get(key):
                file_content_kwargs[key] = litellm_params[key]

        _file_response = file_content(**file_content_kwargs)
        content_bytes = _file_response.content if hasattr(_file_response, "content") else _file_response
        file_content_as_dict = _get_file_content_as_dictionary(content_bytes)
        models = _get_models_from_batch_input_file_content(file_content_as_dict)

        for model in models:
            proxy_model_id = try_get_proxy_model_id_from_router(model)
            if proxy_model_id:
                verbose_proxy_logger.info(
                    f"Resolved batch input file model {model!r} to proxy deployment {proxy_model_id!r}"
                )
                return proxy_model_id

        return None
    except Exception as e:  # noqa: BLE001 - a failed model resolution must never break batch cost tracking
        verbose_proxy_logger.warning(f"Could not resolve proxy model from batch input file {input_file_id!r}: {e}")
        return None


def store_batch_managed_object(
    unified_object_id: str,
    batch_object: "LiteLLMBatch",
    model_object_id: str,
    logging_obj: LiteLLMLoggingObj,
    **kwargs,
) -> None:
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj

        managed_files_hook = proxy_logging_obj.get_proxy_hook("managed_files")
        if managed_files_hook is None or not hasattr(managed_files_hook, "store_unified_object_id"):
            verbose_proxy_logger.warning(
                "Managed files hook not available, cannot store batch object for cost tracking"
            )
            return

        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        _request_metadata = (kwargs.get("litellm_params", {}) or {}).get("metadata", {}) or {}

        user_api_key_dict = UserAPIKeyAuth(
            user_id=_request_metadata.get("user_api_key_user_id", "default-user"),
            api_key="",
            team_id=_request_metadata.get("user_api_key_team_id"),
            team_alias=None,
            user_role=LitellmUserRoles.CUSTOMER,
            user_email=None,
            max_budget=None,
            spend=0.0,
            models=[],
            tpm_limit=None,
            rpm_limit=None,
            budget_duration=None,
            budget_reset_at=None,
            max_parallel_requests=None,
            allowed_model_region=None,
            metadata={},
            key_alias=None,
            permissions={},
            model_max_budget={},
            model_spend={},
        )

        import asyncio

        asyncio.create_task(
            managed_files_hook.store_unified_object_id(  # type: ignore
                unified_object_id=unified_object_id,
                file_object=batch_object,
                litellm_parent_otel_span=None,
                model_object_id=model_object_id,
                file_purpose="batch",
                user_api_key_dict=user_api_key_dict,
            )
        )

        verbose_proxy_logger.info(
            f"Stored batch managed object: unified_object_id={unified_object_id}, batch_id={model_object_id}"
        )
    except Exception as e:  # noqa: BLE001 - a failed managed-object store must never break batch creation
        verbose_proxy_logger.error(f"Error storing batch managed object: {e}")
