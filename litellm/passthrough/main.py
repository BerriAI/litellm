"""
This module is used to pass through requests to the LLM APIs.
"""

from __future__ import annotations

import asyncio
import contextvars
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Coroutine,
    Generator,
    List,
    cast,
)

import httpx
from httpx._types import CookieTypes, QueryParamTypes, RequestFiles

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.passthrough.utils import CommonUtils
from litellm.utils import client

base_llm_http_handler = BaseLLMHTTPHandler()
from .utils import BasePassthroughUtils

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig


class AsyncPassthroughStreamingResponse(AsyncGenerator[Any, Any]):
    def __init__(
        self,
        response: Coroutine[Any, Any, httpx.Response],
        litellm_logging_obj: "LiteLLMLoggingObj",
        provider_config: "BasePassthroughConfig",
    ) -> None:
        self._initialized = False
        self._status_code: int = 0
        self._headers = httpx.Headers()
        self._response_coro = response
        self._response: httpx.Response
        self._iterator: AsyncGenerator[bytes, Any]
        self._litellm_logging_obj = litellm_logging_obj
        self._provider_config = provider_config
        self._raw_bytes: list[bytes] = []
        self._flush_scheduled = False
        self._background_tasks: set[asyncio.Task] = set()

    @property
    def status_code(self) -> int:
        if not self._initialized:
            raise RuntimeError("AsyncPassthroughStreamingResponse must be awaited before accessing status_code")
        return self._status_code

    @status_code.setter
    def status_code(self, value: int) -> None:
        self._status_code = value

    @property
    def headers(self) -> httpx.Headers:
        if not self._initialized:
            raise RuntimeError("AsyncPassthroughStreamingResponse must be awaited before accessing headers")
        return self._headers

    @headers.setter
    def headers(self, value: httpx.Headers) -> None:
        self._headers = value

    def __await__(self):
        async def _init():
            if not self._initialized:
                self._response = await self._response_coro
                self.headers = self._response.headers
                self.status_code = self._response.status_code
                self._initialized = True
                try:
                    self._response.raise_for_status()
                    self._iterator = cast(AsyncGenerator[bytes, Any], self._response.aiter_bytes())
                except Exception:  # noqa: BLE001
                    try:
                        await self._response.aclose()
                    except Exception:  # noqa: BLE001
                        pass
                    raise
            return self

        return _init().__await__()

    def _start_flush(self) -> None:
        if self._flush_scheduled or not self._raw_bytes:
            return
        self._flush_scheduled = True

        try:
            task = asyncio.create_task(
                self._litellm_logging_obj.async_flush_passthrough_collected_chunks(
                    raw_bytes=self._raw_bytes,
                    provider_config=self._provider_config,
                )
            )

            # Compliant: Save a strong reference to prevent GC
            self._background_tasks.add(task)

            # Remove the task from the set when it finishes to avoid memory leaks
            task.add_done_callback(self._background_tasks.discard)
        except Exception as e:  # noqa: BLE001
            verbose_logger.exception(
                "Failed to schedule passthrough spend-tracking flush; %d buffered chunks dropped: %s",
                len(self._raw_bytes),
                e,
            )

    def __aiter__(self) -> "AsyncPassthroughStreamingResponse":
        return self

    async def __anext__(self) -> bytes:
        if not self._initialized:
            await self
        try:
            chunk = await self._iterator.__anext__()
            self._raw_bytes.append(chunk)
            return chunk
        except Exception:  # noqa: BLE001
            self._start_flush()
            try:
                await self._response.aclose()
            except Exception:  # noqa: BLE001
                pass
            raise

    async def asend(self, value: Any) -> bytes:
        if not self._initialized:
            await self
        return await self._iterator.asend(value)

    async def athrow(self, typ: Any, val: Any = None, tb: Any = None) -> bytes:
        if not self._initialized:
            await self
        return await self._iterator.athrow(typ, val, tb)

    async def aclose(self) -> None:
        self._start_flush()
        try:
            if self._initialized:
                await self._response.aclose()
        except Exception:  # noqa: BLE001
            pass


class PassthroughStreamingResponse(Generator[Any, Any, Any]):
    def __init__(
        self,
        response: httpx.Response,
        litellm_logging_obj: "LiteLLMLoggingObj",
        provider_config: "BasePassthroughConfig",
    ) -> None:
        self._response = response
        self.headers = response.headers
        self.status_code = response.status_code
        self._litellm_logging_obj = litellm_logging_obj
        self._provider_config = provider_config
        self._iterator: Generator[bytes, Any, Any] = cast(Generator[bytes, Any, Any], response.iter_bytes())
        self._raw_bytes: List[bytes] = []
        self._flush_scheduled = False

    def _start_flush(self) -> None:
        if self._flush_scheduled or not self._raw_bytes:
            return
        self._flush_scheduled = True

        from litellm.utils import executor

        try:
            executor.submit(
                self._litellm_logging_obj.flush_passthrough_collected_chunks,
                raw_bytes=self._raw_bytes,
                provider_config=self._provider_config,
            )
        except Exception as e:  # noqa: BLE001
            verbose_logger.exception(
                "Failed to schedule passthrough spend-tracking flush; %d buffered chunks dropped: %s",
                len(self._raw_bytes),
                e,
            )

    def __iter__(self) -> "PassthroughStreamingResponse":
        return self

    def __next__(self) -> bytes:
        try:
            chunk = next(self._iterator)
            self._raw_bytes.append(chunk)
            return chunk
        except Exception:  # noqa: BLE001
            self._start_flush()
            try:
                self._response.close()
            except Exception:  # noqa: BLE001
                pass
            raise

    def send(self, value: Any) -> bytes:
        return self._iterator.send(value)

    def throw(self, typ: Any, val: Any = None, tb: Any = None) -> bytes:
        return self._iterator.throw(typ, val, tb)

    def close(self) -> None:
        self._start_flush()
        try:
            self._response.close()
        except Exception:  # noqa: BLE001
            pass


@client
async def allm_passthrough_route(
    *,
    method: str,
    endpoint: str,
    model: str,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    request_query_params: dict | None = None,
    request_headers: dict | None = None,
    content: Any | None = None,
    data: dict | None = None,
    files: RequestFiles | None = None,
    json: Any | None = None,
    params: QueryParamTypes | None = None,
    cookies: CookieTypes | None = None,
    client: HTTPHandler | AsyncHTTPHandler | None = None,
    **kwargs,
) -> httpx.Response | AsyncGenerator[Any, Any]:
    """
    Async: Reranks a list of documents based on their relevance to the query
    """
    try:
        loop = asyncio.get_event_loop()
        kwargs["allm_passthrough_route"] = True

        model, custom_llm_provider, api_key, api_base = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )

        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        provider_config = cast(
            "BasePassthroughConfig" | None, kwargs.get("provider_config")
        ) or ProviderConfigManager.get_provider_passthrough_config(
            provider=LlmProviders(custom_llm_provider),
            model=model,
        )

        if provider_config is None:
            raise Exception(f"Provider {custom_llm_provider} not found")

        func = partial(
            llm_passthrough_route,
            method=method,
            endpoint=endpoint,
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
            request_query_params=request_query_params,
            request_headers=request_headers,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            cookies=cookies,
            client=client,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        # Since allm_passthrough_route=True, we always get a coroutine from _async_passthrough_request
        if asyncio.iscoroutine(init_response):
            response = await init_response

            # Only call raise_for_status if it's a Response object (not a generator)
            if isinstance(response, httpx.Response):
                response.raise_for_status()

            return response
        else:
            # This shouldn't happen when allm_passthrough_route=True, but handle it for type safety
            raise Exception("Expected coroutine from async passthrough route")

    except httpx.HTTPStatusError as e:
        # For HTTP errors, re-raise as-is to preserve the original error details
        # The caller (e.g., proxy layer) can handle conversion to appropriate response format
        raise e
    except Exception as e:
        # For other exceptions, use provider-specific error handling
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        # Get the provider using the same logic as llm_passthrough_route
        _, resolved_custom_llm_provider, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )

        # Get provider config if available
        provider_config = None
        if resolved_custom_llm_provider:
            try:
                provider_config = cast(
                    "BasePassthroughConfig" | None, kwargs.get("provider_config")
                ) or ProviderConfigManager.get_provider_passthrough_config(
                    provider=LlmProviders(resolved_custom_llm_provider),
                    model=model,
                )
            except Exception:
                # If we can't get provider config, pass None
                pass

        if provider_config is None:
            # If no provider config available, raise the original exception
            raise e

        raise base_llm_http_handler._handle_error(
            e=e,
            provider_config=provider_config,
        )


@client
def llm_passthrough_route(
    *,
    method: str,
    endpoint: str,
    model: str,
    custom_llm_provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    request_query_params: dict | None = None,
    request_headers: dict | None = None,
    allm_passthrough_route: bool = False,
    content: Any | None = None,
    data: dict | None = None,
    files: RequestFiles | None = None,
    json: Any | None = None,
    params: QueryParamTypes | None = None,
    cookies: CookieTypes | None = None,
    client: HTTPHandler | AsyncHTTPHandler | None = None,
    **kwargs,
) -> (
    httpx.Response
    | Coroutine[Any, Any, httpx.Response]
    | Coroutine[Any, Any, httpx.Response | AsyncGenerator[Any, Any]]
    | Generator[Any, Any, Any]
    | AsyncGenerator[Any, Any]
):
    """
    Pass through requests to the LLM APIs.

    Step 1. Build the request
    Step 2. Send the request
    Step 3. Return the response
    """
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    _is_async = allm_passthrough_route

    litellm_logging_obj = cast("LiteLLMLoggingObj", kwargs.get("litellm_logging_obj"))

    model, custom_llm_provider, api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )

    litellm_params_dict = get_litellm_params(**kwargs)

    if client is None:
        from litellm.llms.custom_httpx.http_handler import (
            _get_httpx_client,
            get_async_httpx_client,
        )
        from litellm.passthrough.timeout_utils import resolve_llm_passthrough_timeout
        from litellm.types.llms.custom_http import httpxSpecialProvider

        resolved_timeout = resolve_llm_passthrough_timeout(
            kwargs=kwargs,
            litellm_params=litellm_params_dict,
        )
        if _is_async:
            client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.PassThroughEndpoint,
                params={"timeout": resolved_timeout},
            )
        else:
            client = _get_httpx_client(params={"timeout": resolved_timeout})

    # Add model_id to litellm_params if present in kwargs (for Bedrock Application Inference Profiles)
    if "model_id" in kwargs:
        litellm_params_dict["model_id"] = kwargs["model_id"]

    litellm_logging_obj.update_environment_variables(
        model=model,
        litellm_params=litellm_params_dict,
        optional_params={},
        endpoint=endpoint,
        custom_llm_provider=custom_llm_provider,
        request_data=data if data else json,
    )

    provider_config = cast(
        "BasePassthroughConfig" | None, kwargs.get("provider_config")
    ) or ProviderConfigManager.get_provider_passthrough_config(
        provider=LlmProviders(custom_llm_provider),
        model=model,
    )
    if provider_config is None:
        raise Exception(f"Provider {custom_llm_provider} not found")

    updated_url, base_target_url = provider_config.get_complete_url(
        api_base=api_base,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        request_query_params=request_query_params,
        litellm_params=litellm_params_dict,
    )

    # [TODO: Refactor to bedrockpassthroughconfig] need to encode the id of application-inference-profile for bedrock
    if custom_llm_provider == "bedrock" and "application-inference-profile" in endpoint:
        encoded_url_str = CommonUtils.encode_bedrock_runtime_modelid_arn(str(updated_url))
        updated_url = httpx.URL(encoded_url_str)

    # Add or update query parameters
    provider_api_key = provider_config.get_api_key(api_key)

    auth_headers = provider_config.validate_environment(
        headers={},
        model=model,
        messages=[],
        optional_params={},
        litellm_params=litellm_params_dict,
        api_key=provider_api_key,
        api_base=base_target_url,
    )

    headers = BasePassthroughUtils.forward_headers_from_request(
        request_headers=request_headers or {},
        headers=auth_headers,
        forward_headers=False,
    )

    headers, signed_json_body = provider_config.sign_request(
        headers=headers,
        litellm_params=litellm_params_dict,
        request_data=data if data else json,
        api_base=str(updated_url),
        model=model,
    )

    ## SWAP MODEL IN JSON BODY [TODO: REFACTOR TO A provider_config.transform_request method]
    if json and isinstance(json, dict) and "model" in json:
        json["model"] = model

    request = client.client.build_request(
        method=method,
        url=updated_url,
        content=signed_json_body if signed_json_body is not None else content,
        data=data if (signed_json_body is None and content is None) else None,
        files=files,
        json=json if (signed_json_body is None and content is None) else None,
        params=params,
        headers=headers,
        cookies=cookies,
    )

    ## IS STREAMING REQUEST
    is_streaming_request = provider_config.is_streaming_request(
        endpoint=endpoint,
        request_data=data or json or {},
    )

    # Update logging object with streaming status
    litellm_logging_obj.stream = is_streaming_request

    ## LOGGING PRE-CALL
    request_data = data if data else json
    litellm_logging_obj.pre_call(
        input=request_data,
        api_key=provider_api_key,
        additional_args={
            "complete_input_dict": request_data,
            "api_base": str(updated_url),
            "headers": headers,
        },
    )

    try:
        if _is_async:
            # Return the coroutine to be awaited by the caller
            return _async_passthrough_request(
                client=client,
                request=request,
                is_streaming_request=is_streaming_request,
                litellm_logging_obj=litellm_logging_obj,
                provider_config=provider_config,
            )
        else:
            # Sync path - client.client.send returns Response directly
            response: httpx.Response = client.client.send(request=request, stream=is_streaming_request)  # type: ignore
            response.raise_for_status()

            if hasattr(response, "iter_bytes") and is_streaming_request:
                return PassthroughStreamingResponse(response, litellm_logging_obj, provider_config)
            else:
                return response
    except Exception as e:
        if provider_config is None:
            raise e
        raise base_llm_http_handler._handle_error(
            e=e,
            provider_config=provider_config,
        )


async def _async_passthrough_request(
    client: HTTPHandler | AsyncHTTPHandler,
    request: httpx.Request,
    is_streaming_request: bool,
    litellm_logging_obj: "LiteLLMLoggingObj",
    provider_config: "BasePassthroughConfig",
) -> httpx.Response | AsyncGenerator[Any, Any]:
    """
    Handle async passthrough requests.
    Uses async client to send request and properly handles streaming.
    """
    # client.client.send returns a coroutine for async clients
    response_result = client.client.send(request=request, stream=is_streaming_request)

    # Check if it's a coroutine and await it
    if asyncio.iscoroutine(response_result):
        if is_streaming_request:
            return await AsyncPassthroughStreamingResponse(
                response=response_result,
                litellm_logging_obj=litellm_logging_obj,
                provider_config=provider_config,
            )
        else:
            response = await response_result
            await response.aread()
            response.raise_for_status()
            return response
    else:
        # Fallback for sync-like behavior (shouldn't happen in async path)
        raise Exception("Expected coroutine from async client")
