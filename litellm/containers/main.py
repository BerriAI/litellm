import asyncio
import contextvars
import json
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union, overload

import litellm
from litellm.constants import request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.containers.utils import ContainerRequestUtils
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.main import base_llm_http_handler
from litellm.types.containers.main import (
    ContainerCreateOptionalRequestParams,
    ContainerFileListResponse,
    ContainerListOptionalRequestParams,
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes
from litellm.utils import ProviderConfigManager, client

__all__ = [
    "acreate_container",
    "adelete_container",
    "alist_container_files",
    "alist_containers",
    "aretrieve_container",
    "create_container",
    "delete_container",
    "list_container_files",
    "list_containers",
    "retrieve_container",
]

##### Container Create #######################
@client
async def acreate_container(
    name: str,
    expires_after: Optional[Dict[str, Any]] = None,
    file_ids: Optional[List[str]] = None,
    timeout=600,  # default to 10 minutes
    # LiteLLM specific params,
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ContainerObject:
    """Asynchronously calls the `create_container` function with the given arguments and keyword arguments.

    Parameters:
    - `name` (str): Name of the container to create
    - `expires_after` (Optional[Dict[str, Any]]): Container expiration time settings
    - `file_ids` (Optional[List[str]]): IDs of files to copy to the container
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Optional[Literal["openai"]]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (ContainerObject): The created container object
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        func = partial(
            create_container,
            name=name,
            expires_after=expires_after,
            file_ids=file_ids,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

# Overload for when acreate_container=True (returns Coroutine)
@overload
def create_container(
    name: str,
    expires_after: Optional[Dict[str, Any]] = None,
    file_ids: Optional[List[str]] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    acreate_container: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, ContainerObject]:
    ...


@overload
def create_container(
    name: str,
    expires_after: Optional[Dict[str, Any]] = None,
    file_ids: Optional[List[str]] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    acreate_container: Literal[False] = False,
    **kwargs,
) -> ContainerObject:
    ...

# fmt: on


@client
def create_container(
    name: str,
    expires_after: Optional[Dict[str, Any]] = None,
    file_ids: Optional[List[str]] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    ContainerObject,
    Coroutine[Any, Any, ContainerObject],
]:
    """Create a container using the OpenAI Container API.

    Currently supports OpenAI

    Example:
    ```python
    import litellm
    
    response = litellm.create_container(
        name="My Container",
        custom_llm_provider="openai",
    )
    print(response)
    ```
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id")
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response")
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = ContainerObject(**mock_response)
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        # get provider config
        container_provider_config: Optional[BaseContainerConfig] = (
            ProviderConfigManager.get_provider_container_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if container_provider_config is None:
            raise ValueError(f"container operations are not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # Get ContainerCreateOptionalRequestParams with only valid parameters
        container_create_optional_params: ContainerCreateOptionalRequestParams = (
            ContainerRequestUtils.get_requested_container_create_optional_param(local_vars)
        )

        # Get optional parameters for the container API
        container_create_request_params: Dict = (
            ContainerRequestUtils.get_optional_params_container_create(
                container_provider_config=container_provider_config,
                container_create_optional_params=container_create_optional_params,
            )
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            optional_params=dict(container_create_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **container_create_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type for container creation
        litellm_logging_obj.call_type = CallTypes.create_container.value

        return base_llm_http_handler.container_create_handler(
            name=name,
            container_create_request_params=container_create_request_params,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Container List #######################
@client
async def alist_containers(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ContainerListResponse:
    """Asynchronously list containers.

    Parameters:
    - `after` (Optional[str]): A cursor for pagination
    - `limit` (Optional[int]): Number of items to return (1-100, default 20)
    - `order` (Optional[str]): Sort order ('asc' or 'desc', default 'desc')
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Literal["openai"]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (ContainerListResponse): The list of containers
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        func = partial(
            list_containers,
            after=after,
            limit=limit,
            order=order,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

@overload
def list_containers(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    alist_containers: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, ContainerListResponse]:
    ...


@overload
def list_containers(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    alist_containers: Literal[False] = False,
    **kwargs,
) -> ContainerListResponse:
    ...

# fmt: on


@client
def list_containers(
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    ContainerListResponse,
    Coroutine[Any, Any, ContainerListResponse],
]:
    """List containers using the OpenAI Container API.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id")
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response")
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = ContainerListResponse(**mock_response)
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        # get provider config
        container_provider_config: Optional[BaseContainerConfig] = (
            ProviderConfigManager.get_provider_container_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if container_provider_config is None:
            raise ValueError(f"Container provider config not found for provider: {custom_llm_provider}")

        # Get container list request parameters
        container_list_optional_params: ContainerListOptionalRequestParams = (
            ContainerRequestUtils.get_requested_container_list_optional_param(local_vars)
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            optional_params=dict(container_list_optional_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **container_list_optional_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type
        litellm_logging_obj.call_type = CallTypes.list_containers.value

        return base_llm_http_handler.container_list_handler(
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            after=after,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Container Retrieve #######################
@client
async def aretrieve_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ContainerObject:
    """Asynchronously retrieve a container.

    Parameters:
    - `container_id` (str): The ID of the container to retrieve
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Literal["openai"]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (ContainerObject): The container object
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        func = partial(
            retrieve_container,
            container_id=container_id,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

@overload
def retrieve_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    aretrieve_container: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, ContainerObject]:
    ...


@overload
def retrieve_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    aretrieve_container: Literal[False] = False,
    **kwargs,
) -> ContainerObject:
    ...

# fmt: on


@client
def retrieve_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    ContainerObject,
    Coroutine[Any, Any, ContainerObject],
]:
    """Retrieve a container using the OpenAI Container API.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id")
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response")
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = ContainerObject(**mock_response)
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        # get provider config
        container_provider_config: Optional[BaseContainerConfig] = (
            ProviderConfigManager.get_provider_container_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if container_provider_config is None:
            raise ValueError(f"Container provider config not found for provider: {custom_llm_provider}")

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            optional_params={},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type
        litellm_logging_obj.call_type = CallTypes.retrieve_container.value

        return base_llm_http_handler.container_retrieve_handler(
            container_id=container_id,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Container Delete #######################
@client
async def adelete_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> DeleteContainerResult:
    """Asynchronously delete a container.

    Parameters:
    - `container_id` (str): The ID of the container to delete
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Literal["openai"]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (DeleteContainerResult): The deletion result
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        func = partial(
            delete_container,
            container_id=container_id,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

@overload
def delete_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    adelete_container: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, DeleteContainerResult]:
    ...


@overload
def delete_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    adelete_container: Literal[False] = False,
    **kwargs,
) -> DeleteContainerResult:
    ...

# fmt: on


@client
def delete_container(
    container_id: str,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    DeleteContainerResult,
    Coroutine[Any, Any, DeleteContainerResult],
]:
    """Delete a container using the OpenAI Container API.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id")
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response")
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = DeleteContainerResult(**mock_response)
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        # get provider config
        container_provider_config: Optional[BaseContainerConfig] = (
            ProviderConfigManager.get_provider_container_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if container_provider_config is None:
            raise ValueError(f"Container provider config not found for provider: {custom_llm_provider}")

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            optional_params={},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type
        litellm_logging_obj.call_type = CallTypes.delete_container.value

        return base_llm_http_handler.container_delete_handler(
            container_id=container_id,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


##### Container Files List #######################
@client
async def alist_container_files(
    container_id: str,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ContainerFileListResponse:
    """Asynchronously list files in a container.

    Parameters:
    - `container_id` (str): The ID of the container
    - `after` (Optional[str]): A cursor for pagination
    - `limit` (Optional[int]): Number of items to return (1-100, default 20)
    - `order` (Optional[str]): Sort order ('asc' or 'desc', default 'desc')
    - `timeout` (int): Request timeout in seconds
    - `custom_llm_provider` (Literal["openai"]): The LLM provider to use
    - `extra_headers` (Optional[Dict[str, Any]]): Additional headers
    - `extra_query` (Optional[Dict[str, Any]]): Additional query parameters
    - `extra_body` (Optional[Dict[str, Any]]): Additional body parameters
    - `kwargs` (dict): Additional keyword arguments

    Returns:
    - `response` (ContainerFileListResponse): The list of container files
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        func = partial(
            list_container_files,
            container_id=container_id,
            after=after,
            limit=limit,
            order=order,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# fmt: off

@overload
def list_container_files(
    container_id: str,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    alist_container_files: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, ContainerFileListResponse]:
    ...


@overload
def list_container_files(
    container_id: str,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    *,
    alist_container_files: Literal[False] = False,
    **kwargs,
) -> ContainerFileListResponse:
    ...

# fmt: on


@client
def list_container_files(
    container_id: str,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Literal["openai"] = "openai",
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[
    ContainerFileListResponse,
    Coroutine[Any, Any, ContainerFileListResponse],
]:
    """List files in a container using the OpenAI Container API.

    Currently supports OpenAI
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id")
        _is_async = kwargs.pop("async_call", False) is True

        # Check for mock response first
        mock_response = kwargs.get("mock_response")
        if mock_response is not None:
            if isinstance(mock_response, str):
                mock_response = json.loads(mock_response)

            response = ContainerFileListResponse(**mock_response)
            return response

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        # get provider config
        container_provider_config: Optional[BaseContainerConfig] = (
            ProviderConfigManager.get_provider_container_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if container_provider_config is None:
            raise ValueError(f"Container provider config not found for provider: {custom_llm_provider}")

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model="",
            optional_params={"container_id": container_id, "after": after, "limit": limit, "order": order},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Set the correct call type
        litellm_logging_obj.call_type = CallTypes.list_container_files.value

        return base_llm_http_handler.container_file_list_handler(
            container_id=container_id,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            after=after,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
        )

    except Exception as e:
        raise litellm.exception_type(
            model="",
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )

