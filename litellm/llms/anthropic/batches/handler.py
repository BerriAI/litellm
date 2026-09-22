"""
Anthropic Batches API Handler
"""

import asyncio
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
)
from litellm.types.utils import LiteLLMBatch, LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_EMPTY_PARAMS: Final[
    dict
] = {}  # mutable-ok: shared empty mapping for signatures that declare dict; every callee here only reads it
_EMPTY_MESSAGES: Final[list] = []  # mutable-ok: shared empty list for the messages signature; callees only read it

from ..common_utils import AnthropicModelInfo
from .transformation import AnthropicBatchesConfig


class AnthropicBatchesHandler:
    """
    Handler for Anthropic Message Batches API.

    Supports:
    - retrieve_batch() - Retrieve batch status and information
    - list_batches() - List batches
    - cancel_batch() - Cancel a batch
    """

    def __init__(self):
        self.anthropic_model_info = AnthropicModelInfo()
        self.provider_config = AnthropicBatchesConfig()

    @staticmethod
    def _default_logging_obj(call_type: str, call_id: str) -> LiteLLMLoggingObj:
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObjClass,
        )

        return LiteLLMLoggingObjClass(
            model="anthropic/unknown",
            messages=_EMPTY_MESSAGES,
            stream=False,
            call_type=call_type,
            start_time=None,
            litellm_call_id=call_id,
            function_id=call_type,
        )

    async def aretrieve_batch(
        self,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> LiteLLMBatch:
        """
        Async: Retrieve a batch from Anthropic.

        Args:
            batch_id: The batch ID to retrieve
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            logging_obj: Optional logging object

        Returns:
            LiteLLMBatch: Batch information in OpenAI format
        """
        # Resolve API credentials
        api_base = api_base or self.anthropic_model_info.get_api_base(api_base)
        api_key = api_key or self.anthropic_model_info.get_api_key()

        if not api_key:
            raise ValueError("Missing Anthropic API Key")

        # Create a minimal logging object if not provided
        if logging_obj is None:
            from litellm.litellm_core_utils.litellm_logging import (
                Logging as LiteLLMLoggingObjClass,
            )

            logging_obj = LiteLLMLoggingObjClass(
                model="anthropic/unknown",
                messages=[],
                stream=False,
                call_type="batch_retrieve",
                start_time=None,
                litellm_call_id=f"batch_retrieve_{batch_id}",
                function_id="batch_retrieve",
            )

        # Get the complete URL for batch retrieval
        retrieve_url: Final = self.provider_config.get_retrieve_batch_url(
            api_base=api_base,
            batch_id=batch_id,
            optional_params={},
            litellm_params={},
        )

        # Validate environment and get headers
        headers: Final = self.provider_config.validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=api_base,
        )

        logging_obj.pre_call(
            input=batch_id,
            api_key=api_key,
            additional_args={
                "api_base": retrieve_url,
                "headers": headers,
                "complete_input_dict": {},
            },
        )
        # Make the request
        async_client: Final = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC)
        response: Final = await async_client.get(url=retrieve_url, headers=headers)
        response.raise_for_status()

        # Transform response to LiteLLM format
        return self.provider_config.transform_retrieve_batch_response(
            model=None,
            raw_response=response,
            logging_obj=logging_obj,
            litellm_params={},
        )

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> LiteLLMBatch | Coroutine[Any, Any, LiteLLMBatch]:
        """
        Retrieve a batch from Anthropic.

        Args:
            _is_async: Whether to run asynchronously
            batch_id: The batch ID to retrieve
            api_base: Anthropic API base URL
            api_key: Anthropic API key
            timeout: Request timeout
            max_retries: Max retry attempts (unused for now)
            logging_obj: Optional logging object

        Returns:
            LiteLLMBatch or Coroutine: Batch information in OpenAI format
        """
        if _is_async:
            return self.aretrieve_batch(
                batch_id=batch_id,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )
        else:
            return asyncio.run(
                self.aretrieve_batch(
                    batch_id=batch_id,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    logging_obj=logging_obj,
                )
            )

    async def alist_batches(
        self,
        after: str | None,
        limit: int | None,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> dict:
        """
        Async: List batches from Anthropic.
        """
        resolved_api_base: Final = api_base or self.anthropic_model_info.get_api_base(api_base)
        resolved_api_key: Final = api_key or self.anthropic_model_info.get_api_key()

        if not resolved_api_key:
            raise ValueError("Missing Anthropic API Key")

        resolved_logging_obj: Final = logging_obj or self._default_logging_obj(
            call_type="batch_list", call_id="batch_list"
        )

        list_url: Final = self.provider_config.get_list_batches_url(
            api_base=resolved_api_base,
            after=after,
            limit=limit,
            optional_params=_EMPTY_PARAMS,
            litellm_params=_EMPTY_PARAMS,
        )

        headers: Final = self.provider_config.validate_environment(
            headers={},  # mutable-ok: validate_environment fills this mapping in place
            model="",
            messages=_EMPTY_MESSAGES,
            optional_params=_EMPTY_PARAMS,
            litellm_params=_EMPTY_PARAMS,
            api_key=resolved_api_key,
            api_base=resolved_api_base,
        )

        resolved_logging_obj.pre_call(
            input="",
            api_key=resolved_api_key,
            additional_args={  # mutable-ok: pre_call logging contract takes a plain dict
                "api_base": list_url,
                "headers": headers,
                "complete_input_dict": _EMPTY_PARAMS,
            },
        )
        async_client: Final = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC)
        response: Final = await async_client.get(url=list_url, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise self.provider_config.get_error_class(
                error_message=response.text,
                status_code=response.status_code,
                headers=response.headers,
            )

        return self.provider_config.transform_list_batches_response(raw_response=response)

    def list_batches(
        self,
        _is_async: bool,
        after: str | None,
        limit: int | None,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> dict | Coroutine[Any, Any, dict]:
        """
        List batches from Anthropic.
        """
        if _is_async:
            return self.alist_batches(
                after=after,
                limit=limit,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )
        else:
            return asyncio.run(
                self.alist_batches(
                    after=after,
                    limit=limit,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    logging_obj=logging_obj,
                )
            )

    async def acancel_batch(
        self,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> LiteLLMBatch:
        """
        Async: Cancel a batch on Anthropic.
        """
        resolved_api_base: Final = api_base or self.anthropic_model_info.get_api_base(api_base)
        resolved_api_key: Final = api_key or self.anthropic_model_info.get_api_key()

        if not resolved_api_key:
            raise ValueError("Missing Anthropic API Key")

        resolved_logging_obj: Final = logging_obj or self._default_logging_obj(
            call_type="batch_cancel", call_id=f"batch_cancel_{batch_id}"
        )

        cancel_url: Final = self.provider_config.get_cancel_batch_url(
            api_base=resolved_api_base,
            batch_id=batch_id,
            optional_params=_EMPTY_PARAMS,
            litellm_params=_EMPTY_PARAMS,
        )

        headers: Final = self.provider_config.validate_environment(
            headers={},  # mutable-ok: validate_environment fills this mapping in place
            model="",
            messages=_EMPTY_MESSAGES,
            optional_params=_EMPTY_PARAMS,
            litellm_params=_EMPTY_PARAMS,
            api_key=resolved_api_key,
            api_base=resolved_api_base,
        )

        resolved_logging_obj.pre_call(
            input=batch_id,
            api_key=resolved_api_key,
            additional_args={  # mutable-ok: pre_call logging contract takes a plain dict
                "api_base": cancel_url,
                "headers": headers,
                "complete_input_dict": _EMPTY_PARAMS,
            },
        )
        async_client: Final = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC)
        response: Final = await async_client.post(url=cancel_url, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise self.provider_config.get_error_class(
                error_message=response.text,
                status_code=response.status_code,
                headers=response.headers,
            )

        return self.provider_config.transform_cancel_batch_response(
            model=None,
            raw_response=response,
            logging_obj=resolved_logging_obj,
            litellm_params=_EMPTY_PARAMS,
        )

    def cancel_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        api_key: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> LiteLLMBatch | Coroutine[Any, Any, LiteLLMBatch]:
        """
        Cancel a batch on Anthropic.
        """
        if _is_async:
            return self.acancel_batch(
                batch_id=batch_id,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                logging_obj=logging_obj,
            )
        else:
            return asyncio.run(
                self.acancel_batch(
                    batch_id=batch_id,
                    api_base=api_base,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    logging_obj=logging_obj,
                )
            )
