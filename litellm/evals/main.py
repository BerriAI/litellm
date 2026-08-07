"""
Main entry point for Evals API operations
Provides create, list, get, update, delete, and cancel operations for evals
"""

import asyncio
import contextvars
from collections.abc import Coroutine
from functools import partial
from typing import Any, Final

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.evals.transformation import BaseEvalsAPIConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.openai_evals import (
    CancelEvalResponse,
    CancelRunResponse,
    CreateEvalRequest,
    CreateRunRequest,
    DeleteEvalResponse,
    Eval,
    ListEvalsParams,
    ListEvalsResponse,
    ListRunsParams,
    ListRunsResponse,
    Run,
    RunDeleteResponse,
    UpdateEvalRequest,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

# Initialize HTTP handler
base_llm_http_handler = BaseLLMHTTPHandler()
DEFAULT_OPENAI_API_BASE: Final = "https://api.openai.com"


@client
async def acreate_eval(
    data_source_config: dict[str, Any],
    testing_criteria: list[dict[str, Any]],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Eval:
    """
    Async: Create a new evaluation

    Args:
        data_source_config: Configuration for the data source
        testing_criteria: List of graders for all eval runs
        name: Optional name for the evaluation
        metadata: Optional additional metadata (max 16 key-value pairs)
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Eval object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate_eval"] = True

        func: Final = partial(
            create_eval,
            data_source_config=data_source_config,
            testing_criteria=testing_criteria,
            name=name,
            metadata=metadata,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create_eval(
    data_source_config: dict[str, Any],
    testing_criteria: list[dict[str, Any]],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Eval | Coroutine[Any, Any, Eval]:
    """
    Create a new evaluation

    Args:
        data_source_config: Configuration for the data source
        testing_criteria: List of graders for all eval runs
        name: Optional name for the evaluation
        metadata: Optional additional metadata (max 16 key-value pairs)
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Eval object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acreate_eval", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"CREATE eval is not supported for {custom_llm_provider}")

        # Build create request
        create_request: Final[CreateEvalRequest] = {
            "data_source_config": data_source_config,
            "testing_criteria": testing_criteria,
        }
        if name is not None:
            create_request["name"] = name

        # Merge extra_body if provided
        if extra_body:
            create_request.update(extra_body)

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        request_body: Final = evals_api_provider_config.transform_create_eval_request(
            create_request=create_request,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Get API base and URL
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        url: Final = evals_api_provider_config.get_complete_url(api_base=api_base, endpoint="evals")

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params=request_body,
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.create_eval_handler(
            url=url,
            request_body=request_body,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def alist_evals(
    limit: int | None = None,
    after: str | None = None,
    before: str | None = None,
    order: str | None = None,
    order_by: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> ListEvalsResponse:
    """
    Async: List all evaluations

    Args:
        limit: Number of results to return per page (max 100, default 20)
        after: Cursor for pagination - returns evals after this ID
        before: Cursor for pagination - returns evals before this ID
        order: Sort order ('asc' or 'desc', default 'desc')
        order_by: Field to sort by ('created_at' or 'updated_at', default 'created_at')
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        ListEvalsResponse object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist_evals"] = True

        func: Final = partial(
            list_evals,
            limit=limit,
            after=after,
            before=before,
            order=order,
            order_by=order_by,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list_evals(
    limit: int | None = None,
    after: str | None = None,
    before: str | None = None,
    order: str | None = None,
    order_by: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> ListEvalsResponse | Coroutine[Any, Any, ListEvalsResponse]:
    """
    List all evaluations

    Args:
        limit: Number of results to return per page (max 100, default 20)
        after: Cursor for pagination - returns evals after this ID
        before: Cursor for pagination - returns evals before this ID
        order: Sort order ('asc' or 'desc', default 'desc')
        order_by: Field to sort by ('created_at' or 'updated_at', default 'created_at')
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        ListEvalsResponse object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("alist_evals", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"LIST evals is not supported for {custom_llm_provider}")

        # Build list parameters
        list_params: Final[ListEvalsParams] = {}
        if limit is not None:
            list_params["limit"] = limit
        if after is not None:
            list_params["after"] = after
        if before is not None:
            list_params["before"] = before
        if order is not None:
            list_params["order"] = order
        if order_by is not None:
            list_params["order_by"] = order_by

        # Merge extra_query if provided
        if extra_query:
            list_params.update(extra_query)

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        url, query_params = evals_api_provider_config.transform_list_evals_request(
            list_params=list_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params=query_params,
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.list_evals_handler(
            url=url,
            query_params=query_params,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aget_eval(
    eval_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Eval:
    """
    Async: Get an evaluation by ID

    Args:
        eval_id: The ID of the evaluation to fetch
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Eval object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aget_eval"] = True

        func: Final = partial(
            get_eval,
            eval_id=eval_id,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get_eval(
    eval_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Eval | Coroutine[Any, Any, Eval]:
    """
    Get an evaluation by ID

    Args:
        eval_id: The ID of the evaluation to fetch
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Eval object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aget_eval", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"GET eval is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        url, headers = evals_api_provider_config.transform_get_eval_request(
            eval_id=eval_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.get_eval_handler(
            url=url,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aupdate_eval(
    eval_id: str,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Eval:
    """
    Async: Update an evaluation

    Args:
        eval_id: The ID of the evaluation to update
        name: Updated name
        metadata: Updated metadata
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Eval object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aupdate_eval"] = True

        func: Final = partial(
            update_eval,
            eval_id=eval_id,
            name=name,
            metadata=metadata,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def update_eval(
    eval_id: str,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Eval | Coroutine[Any, Any, Eval]:
    """
    Update an evaluation

    Args:
        eval_id: The ID of the evaluation to update
        name: Updated name
        metadata: Updated metadata
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Eval object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aupdate_eval", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"UPDATE eval is not supported for {custom_llm_provider}")

        # Build update request
        update_request: Final[UpdateEvalRequest] = {}
        if name is not None:
            update_request["name"] = name

        # Filter metadata to exclude internal LiteLLM fields
        if metadata is not None:
            # List of internal LiteLLM metadata keys that should NOT be sent to OpenAI
            internal_keys: Final = {
                "headers",
                "requester_metadata",
                "user_api_key_hash",
                "user_api_key_alias",
                "user_api_key_spend",
                "user_api_key_max_budget",
                "user_api_key_team_id",
                "user_api_key_user_id",
                "user_api_key_org_id",
                "user_api_key_team_alias",
                "user_api_key_end_user_id",
                "user_api_key_user_email",
                "user_api_key_request_route",
                "user_api_key_budget_reset_at",
                "user_api_key_auth_metadata",
                "user_api_key",
                "user_api_end_user_max_budget",
                "user_api_key_auth",
                "litellm_api_version",
                "global_max_parallel_requests",
                "user_api_key_team_max_budget",
                "user_api_key_team_spend",
                "user_api_key_model_max_budget",
                "user_api_key_user_spend",
                "user_api_key_user_max_budget",
                "user_api_key_metadata",
                "endpoint",
                "litellm_parent_otel_span",
                "requester_ip_address",
                "user_agent",
            }
            # Only include user-provided metadata keys
            filtered_metadata: Final = {k: v for k, v in metadata.items() if k not in internal_keys}
            if filtered_metadata:  # Only add if there's user metadata
                update_request["metadata"] = filtered_metadata

        # Merge extra_body if provided
        if extra_body:
            update_request.update(extra_body)

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        (
            url,
            headers,
            request_body,
        ) = evals_api_provider_config.transform_update_eval_request(
            eval_id=eval_id,
            update_request=update_request,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params=request_body,
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.update_eval_handler(
            url=url,
            request_body=request_body,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def adelete_eval(
    eval_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> DeleteEvalResponse:
    """
    Async: Delete an evaluation

    Args:
        eval_id: The ID of the evaluation to delete
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        DeleteEvalResponse object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete_eval"] = True

        func: Final = partial(
            delete_eval,
            eval_id=eval_id,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete_eval(
    eval_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> DeleteEvalResponse | Coroutine[Any, Any, DeleteEvalResponse]:
    """
    Delete an evaluation

    Args:
        eval_id: The ID of the evaluation to delete
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        DeleteEvalResponse object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("adelete_eval", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"DELETE eval is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        url, headers = evals_api_provider_config.transform_delete_eval_request(
            eval_id=eval_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.delete_eval_handler(
            url=url,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def acancel_eval(
    eval_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> CancelEvalResponse:
    """
    Async: Cancel a running evaluation

    Args:
        eval_id: The ID of the evaluation to cancel
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        CancelEvalResponse object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acancel_eval"] = True

        func: Final = partial(
            cancel_eval,
            eval_id=eval_id,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def cancel_eval(
    eval_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> CancelEvalResponse | Coroutine[Any, Any, CancelEvalResponse]:
    """
    Cancel a running evaluation

    Args:
        eval_id: The ID of the evaluation to cancel
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        CancelEvalResponse object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acancel_eval", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"CANCEL eval is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        (
            url,
            headers,
            request_body,
        ) = evals_api_provider_config.transform_cancel_eval_request(
            eval_id=eval_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.cancel_eval_handler(
            url=url,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ===================================
# Run API Functions
# ===================================


@client
async def acreate_run(
    eval_id: str,
    data_source: dict[str, Any],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Run:
    """
    Async: Create a new run for an evaluation

    Args:
        eval_id: The ID of the evaluation to run
        data_source: Data source configuration for the run (can be jsonl, completions, or responses type)
        name: Optional name for the run
        metadata: Optional additional metadata
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Run object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate_run"] = True

        func: Final = partial(
            create_run,
            eval_id=eval_id,
            data_source=data_source,
            name=name,
            metadata=metadata,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create_run(
    eval_id: str,
    data_source: dict[str, Any],
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Run | Coroutine[Any, Any, Run]:
    """
    Create a new run for an evaluation

    Args:
        eval_id: The ID of the evaluation to run
        data_source: Data source configuration for the run (can be jsonl, completions, or responses type)
        name: Optional name for the run
        metadata: Optional additional metadata
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout (default 600s for long-running operations)
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Run object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acreate_run", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"CREATE run is not supported for {custom_llm_provider}")

        # Build create request
        create_request: Final[CreateRunRequest] = {
            "data_source": data_source,
        }
        if name is not None:
            create_request["name"] = name
        # if metadata is not None:
        #     create_request["metadata"] = metadata

        # Merge extra_body if provided
        if extra_body:
            create_request.update(extra_body)

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        url, request_body = evals_api_provider_config.transform_create_run_request(
            eval_id=eval_id,
            create_request=create_request,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params=request_body,
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request (default 600s timeout for long-running operations)
        response: Final = base_llm_http_handler.create_run_handler(
            url=url,
            request_body=request_body,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or httpx.Timeout(timeout=600.0, connect=5.0),
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def alist_runs(
    eval_id: str,
    limit: int | None = None,
    after: str | None = None,
    before: str | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> ListRunsResponse:
    """
    Async: List all runs for an evaluation

    Args:
        eval_id: The ID of the evaluation
        limit: Number of results to return per page (max 100, default 20)
        after: Cursor for pagination - returns runs after this ID
        before: Cursor for pagination - returns runs before this ID
        order: Sort order ('asc' or 'desc', default 'desc')
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        ListRunsResponse object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist_runs"] = True

        func: Final = partial(
            list_runs,
            eval_id=eval_id,
            limit=limit,
            after=after,
            before=before,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list_runs(
    eval_id: str,
    limit: int | None = None,
    after: str | None = None,
    before: str | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> ListRunsResponse | Coroutine[Any, Any, ListRunsResponse]:
    """
    List all runs for an evaluation

    Args:
        eval_id: The ID of the evaluation
        limit: Number of results to return per page (max 100, default 20)
        after: Cursor for pagination - returns runs after this ID
        before: Cursor for pagination - returns runs before this ID
        order: Sort order ('asc' or 'desc', default 'desc')
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        ListRunsResponse object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("alist_runs", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"LIST runs is not supported for {custom_llm_provider}")

        # Build list parameters
        list_params: Final[ListRunsParams] = {}
        if limit is not None:
            list_params["limit"] = limit
        if after is not None:
            list_params["after"] = after
        if before is not None:
            list_params["before"] = before
        if order is not None:
            list_params["order"] = order

        # Merge extra_query if provided
        if extra_query:
            list_params.update(extra_query)

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        url, query_params = evals_api_provider_config.transform_list_runs_request(
            eval_id=eval_id,
            list_params=list_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id, **query_params},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.list_runs_handler(
            url=url,
            query_params=query_params,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aget_run(
    eval_id: str,
    run_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Run:
    """
    Async: Get a specific run

    Args:
        eval_id: The ID of the evaluation
        run_id: The ID of the run to retrieve
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Run object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aget_run"] = True

        func: Final = partial(
            get_run,
            eval_id=eval_id,
            run_id=run_id,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get_run(
    eval_id: str,
    run_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Run | Coroutine[Any, Any, Run]:
    """
    Get a specific run

    Args:
        eval_id: The ID of the evaluation
        run_id: The ID of the run to retrieve
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        Run object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aget_run", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"GET run is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        url, headers = evals_api_provider_config.transform_get_run_request(
            eval_id=eval_id,
            run_id=run_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id, "run_id": run_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.get_run_handler(
            url=url,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def acancel_run(
    eval_id: str,
    run_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> CancelRunResponse:
    """
    Async: Cancel a running run

    Args:
        eval_id: The ID of the evaluation
        run_id: The ID of the run to cancel
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        CancelRunResponse object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acancel_run"] = True

        func: Final = partial(
            cancel_run,
            eval_id=eval_id,
            run_id=run_id,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def cancel_run(
    eval_id: str,
    run_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> CancelRunResponse | Coroutine[Any, Any, CancelRunResponse]:
    """
    Cancel a running run

    Args:
        eval_id: The ID of the evaluation
        run_id: The ID of the run to cancel
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        CancelRunResponse object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acancel_run", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"CANCEL run is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        (
            url,
            headers,
            request_body,
        ) = evals_api_provider_config.transform_cancel_run_request(
            eval_id=eval_id,
            run_id=run_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id, "run_id": run_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.cancel_run_handler(
            url=url,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ===================================
# Delete Run API Functions
# ===================================


@client
async def adelete_run(
    eval_id: str,
    run_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> RunDeleteResponse:
    """
    Async: Delete a run

    Args:
        eval_id: The ID of the evaluation
        run_id: The ID of the run to delete
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        RunDeleteResponse object
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete_run"] = True

        func: Final = partial(
            delete_run,
            eval_id=eval_id,
            run_id=run_id,
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
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete_run(
    eval_id: str,
    run_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> RunDeleteResponse | Coroutine[Any, Any, RunDeleteResponse]:
    """
    Delete a run

    Args:
        eval_id: The ID of the evaluation
        run_id: The ID of the run to delete
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'openai')
        **kwargs: Additional parameters

    Returns:
        RunDeleteResponse object
    """
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("adelete_run", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "openai"

        # Get provider config
        evals_api_provider_config: BaseEvalsAPIConfig | None = ProviderConfigManager.get_provider_evals_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if evals_api_provider_config is None:
            raise ValueError(f"DELETE run is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = evals_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        api_base: Final = litellm_params.api_base or DEFAULT_OPENAI_API_BASE
        (
            url,
            headers,
            request_body,
        ) = evals_api_provider_config.transform_delete_run_request(
            eval_id=eval_id,
            run_id=run_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"eval_id": eval_id, "run_id": run_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response: Final = base_llm_http_handler.delete_run_handler(
            url=url,
            evals_api_provider_config=evals_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
