import json
import urllib
from typing import Any, Callable, Optional, Union, List, Dict
import time
import httpx
from functools import cached_property

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper, get_secret
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from ...base import BaseLLM
from .transformation import GenAIHubOrchestrationConfig


from typing import Callable

from typing import Optional, TYPE_CHECKING, Union
from types import ModuleType

# Type checking block for optional imports
if TYPE_CHECKING:
    from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient, temporary_headers_addition
    from gen_ai_hub.orchestration.exceptions import OrchestrationError

# Try to import the optional module
try:
    from gen_ai_hub.proxy.gen_ai_hub_proxy import GenAIHubProxyClient, temporary_headers_addition
    from gen_ai_hub.orchestration.exceptions import OrchestrationError
    _gen_ai_hub_import_error = None
except ImportError as err:
    GenAIHubProxyClient = None  # type: ignore
    _gen_ai_hub_import_error = err

class OptionalDependencyError(ImportError):
    """Custom error for missing optional dependencies."""
    pass


class KeepWaiting(Exception):
    pass


class GenAIHubOrchestrationError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self.message = message
        if request is not None:
            self.request = request
        else:
            self.request = httpx.Request(
                method="POST",
                url="https://docs.predibase.com/user-guide/inference/rest_api",
            )
        if response is not None:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )



class GenAIHubOrchestration(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self._client: Optional['GenAIHubProxyClient'] = None
        self._orchestration_client = None


    def _ensure_gen_ai_hub_installed(self) -> None:
        """Ensure the gen-ai-hub package is available."""
        if _gen_ai_hub_import_error is not None:
            raise OptionalDependencyError(
                "The gen-ai-hub package is required for this functionality. "
                "Please install it with: pip install gen-ai-hub"
            ) from _gen_ai_hub_import_error

    @property
    def orchestration_client(self):
        """Initialize and get the orchestration client."""
        self._ensure_gen_ai_hub_installed()
        from gen_ai_hub.orchestration.service import OrchestrationService

        if GenAIHubProxyClient is None:  # This should never happen due to _ensure_dependency
            raise RuntimeError("GenAIHubProxyClient is None despite passing dependency check")
        if not self._orchestration_client:
            self._orchestration_client = OrchestrationService(proxy_client=self._client)
        return self._orchestration_client


    def _validate_environment(
        self, api_key: Optional[str], user_headers: dict, tenant_id: Optional[str]
    ) -> dict:
        if api_key is None:
            raise ValueError(
                "Missing Predibase API Key - A call is being made to predibase but no key is set either in the environment variables or via params"
            )
        if tenant_id is None:
            raise ValueError(
                "Missing Predibase Tenant ID - Required for making the request. Set dynamically (e.g. `completion(..tenant_id=<MY-ID>)`) or in env - `PREDIBASE_TENANT_ID`."
            )
        headers = {
            "content-type": "application/json",
            "Authorization": "Bearer {}".format(api_key),
        }
        if user_headers is not None and isinstance(user_headers, dict):
            headers = {**headers, **user_headers}
        return headers


    def encode_model_id(self, model_id: str) -> str:
        """
        Double encode the model ID to ensure it matches the expected double-encoded format.
        Args:
            model_id (str): The model ID to encode.
        Returns:
            str: The double-encoded model ID.
        """
        return urllib.parse.quote(model_id, safe="")  # type: ignore


    async def _async_streaming(
        self,
        config,
        model: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            with temporary_headers_addition(extra_headers or {}):
                response = await self.orchestration_client.astream(
                    config=config,
                    # timeout=int(timeout) if isinstance(timeout, (float)) else timeout
                )
        except OrchestrationError as err:
            raise GenAIHubOrchestrationError(status_code=err.code, message=err.message)
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(status_code=408, message="Timeout error occurred.")
        return CustomStreamWrapper(
                completion_stream=response,
                model=model,
                logging_obj=logging_obj,
                custom_llm_provider='sap',
                stream_options={},
                make_call=None,
            )

    async def _async_completion(
        self,
        config,
        model: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            with temporary_headers_addition(extra_headers or {}):
                response = await self.orchestration_client.arun(
                    config=config,
                    # timeout=int(timeout) if isinstance(timeout, (float)) else timeout
                )
        except OrchestrationError as err:
            raise GenAIHubOrchestrationError(status_code=err.code, message=err.message)
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(status_code=408, message="Timeout error occurred.")
        return litellm.GenAIHubOrchestrationConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            config=config,
            logging_obj=logging_obj,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    def _streaming(
        self,
        config,
        model: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            with temporary_headers_addition(extra_headers or {}):
                response = self.orchestration_client.stream(
                    config=config,
                    # timeout=int(timeout) if isinstance(timeout, (float)) else timeout
                )
        except OrchestrationError as err:
            raise GenAIHubOrchestrationError(status_code=err.code, message=err.message)
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(status_code=408, message="Timeout error occurred.")
        return CustomStreamWrapper(
                completion_stream=response,
                model=model,
                logging_obj=logging_obj,
                custom_llm_provider='sap',
                stream_options={},
                make_call=None,
            )

    def _complete(
        self,
        config,
        model: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        optional_params: dict,
        extra_headers: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        try:
            with temporary_headers_addition(extra_headers or {}):
                response = self.orchestration_client.run(
                    config=config,
                    # timeout=int(timeout) if isinstance(timeout, (float)) else timeout
                )
        except OrchestrationError as err:
            raise GenAIHubOrchestrationError(status_code=err.code, message=err.message)
        except httpx.TimeoutException:
            raise GenAIHubOrchestrationError(status_code=408, message="Timeout error occurred.")
        return litellm.GenAIHubOrchestrationConfig()._transform_response(
            model=model,
            response=response,
            model_response=model_response,
            config=config,
            logging_obj=logging_obj,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    def completion(self,
                   model: str,
                   messages: List[Dict[str, str]],
                   model_response: ModelResponse,
                   print_verbose: Callable,
                   encoding,
                   logging_obj: LiteLLMLoggingObject,
                   optional_params: dict,
                   acompletion: bool,
                   headers: Optional[Dict] = None,
                   timeout: Optional[Union[float, httpx.Timeout]] = None,
                   litellm_params: Optional[Dict]=None,
                   logger_fn=None,
                   extra_headers: Optional[dict] = None,
                   client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
                   **kwargs):

        headers = {} or headers
        stream = optional_params.pop("stream", None)

        config = litellm.GenAIHubOrchestrationConfig()._transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        ## COMPLETION CALL
            ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": config.to_dict(),
                "api_base": self.orchestration_client.api_url,
                "headers": self.orchestration_client.proxy_client.request_header,
            },
        )


        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion and not stream:
            return self._async_completion(
                config=config,
                model=model,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )
        elif acompletion and stream is not None and stream is True:
            return self._async_streaming(
                config=config,
                model=model,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )

        elif stream is not None and stream is True:
            return self._streaming(
                config=config,
                model=model,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )
        else:
            return self._complete(
                config=config,
                model=model,
                model_response=model_response,
                print_verbose=print_verbose,
                timeout=timeout,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                extra_headers=extra_headers,
                client=(
                    client
                    if client is not None and isinstance(client, AsyncHTTPHandler)
                    else None
                ),
            )