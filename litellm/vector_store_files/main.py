"""LiteLLM SDK functions for managing vector store files."""

import asyncio
import contextvars
from collections.abc import Coroutine
from functools import partial
from typing import Any, Final

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.vector_store_files import (
    VectorStoreFileContentResponse,
    VectorStoreFileCreateRequest,
    VectorStoreFileDeleteResponse,
    VectorStoreFileListQueryParams,
    VectorStoreFileListResponse,
    VectorStoreFileObject,
    VectorStoreFileUpdateRequest,
)
from litellm.utils import ProviderConfigManager, client
from litellm.vector_store_files.utils import VectorStoreFileRequestUtils

base_llm_http_handler = BaseLLMHTTPHandler()

VectorStoreFileAttributeValue = str | int | float | bool
VectorStoreFileAttributes = dict[str, VectorStoreFileAttributeValue]


def _ensure_provider(custom_llm_provider: str | None) -> str:
    return custom_llm_provider or "openai"


def _prepare_registry_credentials(
    *,
    vector_store_id: str,
    kwargs: dict[str, Any],
) -> None:
    if litellm.vector_store_registry is None:
        return
    try:
        registry_credentials: Final = litellm.vector_store_registry.get_credentials_for_vector_store(vector_store_id)
        if registry_credentials:
            kwargs.update(registry_credentials)
    except Exception:
        pass


@client
async def acreate(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: VectorStoreFileAttributes | None = None,
    chunking_strategy: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileObject:
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate"] = True

        func: Final = partial(
            create,
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: VectorStoreFileAttributes | None = None,
    chunking_strategy: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileObject | Coroutine[Any, Any, VectorStoreFileObject]:
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id")
        _is_async: Final = kwargs.pop("acreate", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Vector store file create is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        create_request: VectorStoreFileCreateRequest = VectorStoreFileRequestUtils.get_create_request_params(local_vars)
        create_request["file_id"] = file_id

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                **create_request,
            },
            litellm_params={
                "vector_store_id": vector_store_id,
                "litellm_call_id": litellm_call_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_file_create_handler(
            vector_store_id=vector_store_id,
            create_request=create_request,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def alist(
    *,
    vector_store_id: str,
    after: str | None = None,
    before: str | None = None,
    filter: str | None = None,
    limit: int | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileListResponse:
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist"] = True

        func: Final = partial(
            list,
            vector_store_id=vector_store_id,
            after=after,
            before=before,
            filter=filter,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list(
    *,
    vector_store_id: str,
    after: str | None = None,
    before: str | None = None,
    filter: str | None = None,
    limit: int | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileListResponse | Coroutine[Any, Any, VectorStoreFileListResponse]:
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id")
        _is_async: Final = kwargs.pop("alist", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Vector store file list is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        list_query: VectorStoreFileListQueryParams = VectorStoreFileRequestUtils.get_list_query_params(local_vars)

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"vector_store_id": vector_store_id, **list_query},
            litellm_params={
                "vector_store_id": vector_store_id,
                "litellm_call_id": litellm_call_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_file_list_handler(
            vector_store_id=vector_store_id,
            query_params=list_query,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aretrieve(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileObject:
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aretrieve"] = True

        func: Final = partial(
            retrieve,
            vector_store_id=vector_store_id,
            file_id=file_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def retrieve(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileObject | Coroutine[Any, Any, VectorStoreFileObject]:
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id")
        _is_async: Final = kwargs.pop("aretrieve", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Vector store file retrieve is not supported for {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
            },
            litellm_params={
                "vector_store_id": vector_store_id,
                "litellm_call_id": litellm_call_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_file_retrieve_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aretrieve_content(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileContentResponse:
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aretrieve_content"] = True

        func: Final = partial(
            retrieve_content,
            vector_store_id=vector_store_id,
            file_id=file_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def retrieve_content(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileContentResponse | Coroutine[Any, Any, VectorStoreFileContentResponse]:
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id")
        _is_async: Final = kwargs.pop("aretrieve_content", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Vector store file content retrieve is not supported for {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
            },
            litellm_params={
                "vector_store_id": vector_store_id,
                "litellm_call_id": litellm_call_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_file_content_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aupdate(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: VectorStoreFileAttributes,
    extra_headers: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileObject:
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aupdate"] = True

        func: Final = partial(
            update,
            vector_store_id=vector_store_id,
            file_id=file_id,
            attributes=attributes,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def update(
    *,
    vector_store_id: str,
    file_id: str,
    attributes: VectorStoreFileAttributes,
    extra_headers: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileObject | Coroutine[Any, Any, VectorStoreFileObject]:
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id")
        _is_async: Final = kwargs.pop("aupdate", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Vector store file update is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        update_request: VectorStoreFileUpdateRequest = VectorStoreFileRequestUtils.get_update_request_params(local_vars)
        update_request["attributes"] = attributes

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
                **update_request,
            },
            litellm_params={
                "vector_store_id": vector_store_id,
                "litellm_call_id": litellm_call_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_file_update_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            update_request=update_request,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def adelete(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileDeleteResponse:
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete"] = True

        func: Final = partial(
            delete,
            vector_store_id=vector_store_id,
            file_id=file_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)
        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete(
    *,
    vector_store_id: str,
    file_id: str,
    extra_headers: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> VectorStoreFileDeleteResponse | Coroutine[Any, Any, VectorStoreFileDeleteResponse]:
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id")
        _is_async: Final = kwargs.pop("adelete", False) is True

        custom_llm_provider = _ensure_provider(custom_llm_provider)

        _prepare_registry_credentials(vector_store_id=vector_store_id, kwargs=kwargs)

        litellm_params: Final = GenericLiteLLMParams(vector_store_id=vector_store_id, **kwargs)

        provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(
            provider=LlmProviders(custom_llm_provider)
        )
        if provider_config is None:
            raise ValueError(f"Vector store file delete is not supported for {custom_llm_provider}")

        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={
                "vector_store_id": vector_store_id,
                "file_id": file_id,
            },
            litellm_params={
                "vector_store_id": vector_store_id,
                "litellm_call_id": litellm_call_id,
                **litellm_params.model_dump(exclude_none=True),
            },
            custom_llm_provider=custom_llm_provider,
        )

        response: Final = base_llm_http_handler.vector_store_file_delete_handler(
            vector_store_id=vector_store_id,
            file_id=file_id,
            vector_store_files_provider_config=provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            client=kwargs.get("client"),
            _is_async=_is_async,
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
