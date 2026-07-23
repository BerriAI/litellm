"""
Main entry point for Skills API operations
Provides create, list, get, and delete operations for skills
"""

import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Optional, Union

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.skill_id_utils import (
    encode_skill_id,
    get_original_skill_id,
)
from litellm.llms.base_llm.skills.transformation import BaseSkillsAPIConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.llms.anthropic_skills import (
    CreateSkillRequest,
    DeleteSkillResponse,
    ListSkillsParams,
    ListSkillsResponse,
    Skill,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, client

# Initialize HTTP handler
base_llm_http_handler = BaseLLMHTTPHandler()
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"

# Initialize LiteLLM skills handler (lazy - only used when custom_llm_provider="litellm")
_litellm_skills_handler = None


def _get_skill_model(kwargs: dict[str, Any]) -> str | None:
    model = kwargs.get("_litellm_skill_model") or kwargs.get("model")
    return model if isinstance(model, str) and model else None


def _get_native_skill_model(custom_llm_provider: str | None, kwargs: dict[str, Any]) -> str | None:
    if custom_llm_provider not in {LlmProviders.OPENAI.value, LlmProviders.AZURE.value}:
        return None
    return _get_skill_model(kwargs)


def _wrap_skill_response(response: Any, model: str | None) -> Any:
    if model is None:
        return response
    if hasattr(response, "id") and isinstance(response.id, str):
        response.id = encode_skill_id(response.id, model)
    if hasattr(response, "data") and isinstance(response.data, list):
        for item in response.data:
            if hasattr(item, "id") and isinstance(item.id, str):
                item.id = encode_skill_id(item.id, model)
    if hasattr(response, "skill_id") and isinstance(response.skill_id, str):
        response.skill_id = encode_skill_id(response.skill_id, model)
    return response


def _get_user_api_key_auth_from_kwargs(kwargs: Dict[str, Any]) -> Optional[Any]:
    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = kwargs.get(metadata_key)
        if isinstance(metadata, dict) and metadata.get("user_api_key_auth") is not None:
            return metadata["user_api_key_auth"]
    return None


def _get_skill_request_metadata(
    kwargs: Dict[str, Any],
    extra_body: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if extra_body and isinstance(extra_body.get("metadata"), dict):
        return extra_body["metadata"]

    metadata = kwargs.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("requester_metadata"), dict):
        return metadata["requester_metadata"]
    return None


def _get_litellm_skills_handler():
    """Lazy initialization of LiteLLM skills handler to avoid import overhead."""
    global _litellm_skills_handler
    if _litellm_skills_handler is None:
        from litellm.llms.litellm_proxy.skills.transformation import (
            LiteLLMSkillsTransformationHandler,
        )

        _litellm_skills_handler = LiteLLMSkillsTransformationHandler()
    return _litellm_skills_handler


@client
async def acreate_skill(
    files: Optional[List[Any]] = None,
    display_title: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Skill:
    """
    Async: Create a new skill

    Args:
        files: Files to upload for the skill. All files must be in the same top-level directory and must include a SKILL.md file at the root.
        display_title: Optional display title for the skill
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        Skill object
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acreate_skill"] = True

        func = partial(
            create_skill,
            files=files,
            display_title=display_title,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def create_skill(
    files: Optional[List[Any]] = None,
    display_title: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[Skill, Coroutine[Any, Any, Skill]]:
    """
    Create a new skill

    Args:
        files: Files to upload for the skill. All files must be in the same top-level directory and must include a SKILL.md file at the root.
        display_title: Optional display title for the skill
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        Skill object
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acreate_skill", False) is True

        # Get LiteLLM parameters
        litellm_params = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

        # Build create request
        create_request: CreateSkillRequest = {}
        if display_title is not None:
            create_request["display_title"] = display_title
        if files is not None:
            create_request["files"] = files

        # Merge extra_body if provided
        if extra_body:
            create_request.update(extra_body)  # type: ignore

        # Route to LiteLLM DB if custom_llm_provider="litellm_proxy"
        if custom_llm_provider == LlmProviders.LITELLM_PROXY.value:
            return _get_litellm_skills_handler().create_skill_handler(
                display_title=display_title,
                files=files,
                metadata=_get_skill_request_metadata(kwargs, extra_body),
                user_id=kwargs.get("user_id"),
                user_api_key_dict=_get_user_api_key_auth_from_kwargs(kwargs),
                _is_async=_is_async,
                logging_obj=litellm_logging_obj,
                litellm_call_id=litellm_call_id,
            )

        # Get provider config for external providers (Anthropic, etc.)
        skills_api_provider_config: Optional[BaseSkillsAPIConfig] = (
            ProviderConfigManager.get_provider_skills_api_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if skills_api_provider_config is None:
            raise ValueError(f"CREATE skill is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = dict(extra_headers or {})
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        request_body = skills_api_provider_config.transform_create_skill_request(
            create_request=create_request,
            litellm_params=litellm_params,
            headers=headers,
        )

        api_base = skills_api_provider_config.get_api_base(litellm_params)
        url = skills_api_provider_config.get_complete_url(
            api_base=api_base,
            endpoint="skills",
            litellm_params=litellm_params,
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
        response = base_llm_http_handler.create_skill_handler(
            url=url,
            request_body=request_body,
            skills_api_provider_config=skills_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def alist_skills(
    limit: Optional[int] = None,
    page: Optional[str] = None,
    after: str | None = None,
    order: str | None = None,
    source: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> ListSkillsResponse:
    """
    Async: List all skills

    Args:
        limit: Number of results to return per page (max 100, default 20)
        page: Pagination token for fetching a specific page of results
        source: Filter skills by source ('custom' or 'anthropic')
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        ListSkillsResponse object
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["alist_skills"] = True

        func = partial(
            list_skills,
            limit=limit,
            page=page,
            after=after,
            order=order,
            source=source,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list_skills(
    limit: Optional[int] = None,
    page: Optional[str] = None,
    after: str | None = None,
    order: str | None = None,
    source: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[ListSkillsResponse, Coroutine[Any, Any, ListSkillsResponse]]:
    """
    List all skills

    Args:
        limit: Number of results to return per page (max 100, default 20)
        page: Pagination token for fetching a specific page of results
        source: Filter skills by source ('custom' or 'anthropic')
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        ListSkillsResponse object
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("alist_skills", False) is True

        # Get LiteLLM parameters
        litellm_params = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

        # Route to LiteLLM DB if custom_llm_provider="litellm_proxy"
        if custom_llm_provider == LlmProviders.LITELLM_PROXY.value:
            return _get_litellm_skills_handler().list_skills_handler(
                limit=limit or 20,
                offset=0,
                user_api_key_dict=_get_user_api_key_auth_from_kwargs(kwargs),
                _is_async=_is_async,
                logging_obj=litellm_logging_obj,
                litellm_call_id=litellm_call_id,
            )

        # Get provider config for external providers (Anthropic, etc.)
        skills_api_provider_config: Optional[BaseSkillsAPIConfig] = (
            ProviderConfigManager.get_provider_skills_api_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if skills_api_provider_config is None:
            raise ValueError(f"LIST skills is not supported for {custom_llm_provider}")

        # Build list parameters
        list_params: ListSkillsParams = {}
        if limit is not None:
            list_params["limit"] = limit
        if page is not None:
            list_params["page"] = page
        if after is not None:
            list_params["after"] = after
        if order is not None:
            list_params["order"] = order
        if source is not None:
            list_params["source"] = source

        # Merge extra_query if provided
        if extra_query:
            list_params.update(extra_query)  # type: ignore

        # Validate environment and get headers
        headers = dict(extra_headers or {})
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        url, query_params = skills_api_provider_config.transform_list_skills_request(
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
        response = base_llm_http_handler.list_skills_handler(
            url=url,
            query_params=query_params,
            skills_api_provider_config=skills_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aget_skill(
    skill_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Skill:
    """
    Async: Get a skill by ID

    Args:
        skill_id: The ID of the skill to fetch
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        Skill object
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aget_skill"] = True

        func = partial(
            get_skill,
            skill_id=skill_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get_skill(
    skill_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[Skill, Coroutine[Any, Any, Skill]]:
    """
    Get a skill by ID

    Args:
        skill_id: The ID of the skill to fetch
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        Skill object
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aget_skill", False) is True

        # Get LiteLLM parameters
        litellm_params = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

        skill_id = get_original_skill_id(skill_id)

        # Route to LiteLLM DB if custom_llm_provider="litellm_proxy"
        if custom_llm_provider == LlmProviders.LITELLM_PROXY.value:
            return _get_litellm_skills_handler().get_skill_handler(
                skill_id=skill_id,
                user_api_key_dict=_get_user_api_key_auth_from_kwargs(kwargs),
                _is_async=_is_async,
                logging_obj=litellm_logging_obj,
                litellm_call_id=litellm_call_id,
            )

        # Get provider config for external providers (Anthropic, etc.)
        skills_api_provider_config: Optional[BaseSkillsAPIConfig] = (
            ProviderConfigManager.get_provider_skills_api_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if skills_api_provider_config is None:
            raise ValueError(f"GET skill is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = dict(extra_headers or {})
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        api_base = skills_api_provider_config.get_api_base(litellm_params)

        # Transform request
        url, headers = skills_api_provider_config.transform_get_skill_request(
            skill_id=skill_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"skill_id": skill_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response = base_llm_http_handler.get_skill_handler(
            url=url,
            skills_api_provider_config=skills_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def adelete_skill(
    skill_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> DeleteSkillResponse:
    """
    Async: Delete a skill by ID

    Args:
        skill_id: The ID of the skill to delete
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        DeleteSkillResponse object
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["adelete_skill"] = True

        func = partial(
            delete_skill,
            skill_id=skill_id,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response
        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete_skill(
    skill_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[DeleteSkillResponse, Coroutine[Any, Any, DeleteSkillResponse]]:
    """
    Delete a skill by ID

    Args:
        skill_id: The ID of the skill to delete
        extra_headers: Additional headers for the request
        extra_query: Additional query parameters
        timeout: Request timeout
        custom_llm_provider: Provider name (e.g., 'anthropic')
        **kwargs: Additional parameters

    Returns:
        DeleteSkillResponse object
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("adelete_skill", False) is True

        # Get LiteLLM parameters
        litellm_params = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

        skill_id = get_original_skill_id(skill_id)

        # Route to LiteLLM DB if custom_llm_provider="litellm_proxy"
        if custom_llm_provider == LlmProviders.LITELLM_PROXY.value:
            return _get_litellm_skills_handler().delete_skill_handler(
                skill_id=skill_id,
                user_api_key_dict=_get_user_api_key_auth_from_kwargs(kwargs),
                _is_async=_is_async,
                logging_obj=litellm_logging_obj,
                litellm_call_id=litellm_call_id,
            )

        # Get provider config for external providers (Anthropic, etc.)
        skills_api_provider_config: Optional[BaseSkillsAPIConfig] = (
            ProviderConfigManager.get_provider_skills_api_config(
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if skills_api_provider_config is None:
            raise ValueError(f"DELETE skill is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = dict(extra_headers or {})
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        api_base = skills_api_provider_config.get_api_base(litellm_params)

        # Transform request
        url, headers = skills_api_provider_config.transform_delete_skill_request(
            skill_id=skill_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Pre-call logging
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=None,
            optional_params={"skill_id": skill_id},
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Make HTTP request
        response = base_llm_http_handler.delete_skill_handler(
            url=url,
            skills_api_provider_config=skills_api_provider_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


def _run_skill_operation(
    operation: str,
    skill_id: str,
    version: str | None = None,
    files: list[Any] | None = None,
    default: bool | None = None,
    default_version: str | None = None,
    after: str | None = None,
    limit: int | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    is_async = kwargs.pop("_skill_is_async", False) is True
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")
    litellm_params = GenericLiteLLMParams(**kwargs)
    custom_llm_provider = custom_llm_provider or kwargs.get("custom_llm_provider")
    if custom_llm_provider is None:
        raise ValueError("custom_llm_provider is required for native Skills operations")
    skill_id = get_original_skill_id(skill_id)
    skills_api_provider_config = ProviderConfigManager.get_provider_skills_api_config(
        provider=litellm.LlmProviders(custom_llm_provider)
    )
    if skills_api_provider_config is None or not getattr(skills_api_provider_config, "is_openai_native", False):
        raise ValueError(f"{operation} skill operation is not supported for {custom_llm_provider}")

    headers = skills_api_provider_config.validate_environment(
        headers=dict(extra_headers or {}), litellm_params=litellm_params
    )
    request_body: dict[str, Any] = dict(extra_body or {})
    if operation == "update":
        if default_version is None:
            raise ValueError("default_version is required")
        request_body["default_version"] = default_version
    elif operation == "create_version":
        if files is not None:
            request_body["files"] = files
        if default is not None:
            request_body["default"] = default

    query_params: dict[str, Any] = dict(extra_query or {})
    if operation == "list_versions":
        if after is not None:
            query_params["after"] = after
        if limit is not None:
            query_params["limit"] = limit
        if order is not None:
            query_params["order"] = order

    url = skills_api_provider_config.get_skill_operation_url(
        operation=operation,
        skill_id=skill_id,
        version=version,
        litellm_params=litellm_params,
    )
    response = base_llm_http_handler.skill_operation_handler(
        method={
            "update": "POST",
            "content": "GET",
            "create_version": "POST",
            "list_versions": "GET",
            "version": "GET",
            "version_content": "GET",
            "delete_version": "DELETE",
        }[operation],
        operation=operation,
        url=url,
        skills_api_provider_config=skills_api_provider_config,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=litellm_logging_obj,
        request_body=request_body or None,
        query_params=query_params or None,
        extra_headers=headers,
        timeout=timeout or request_timeout,
        _is_async=is_async,
        client=kwargs.get("client"),
        shared_session=kwargs.get("shared_session"),
    )
    if is_async:
        return response
    return _wrap_skill_response(response, _get_native_skill_model(custom_llm_provider, kwargs))


async def _call_skill_operation_async(function: Any, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(function, **kwargs))
    if asyncio.iscoroutine(result):
        result = await result
    return result


@client
def update_skill(
    skill_id: str,
    default_version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="update",
        skill_id=skill_id,
        default_version=default_version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def aupdate_skill(
    skill_id: str,
    default_version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        update_skill,
        skill_id=skill_id,
        default_version=default_version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
def create_skill_version(
    skill_id: str,
    files: list[Any] | None = None,
    default: bool | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="create_version",
        skill_id=skill_id,
        files=files,
        default=default,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def acreate_skill_version(
    skill_id: str,
    files: list[Any] | None = None,
    default: bool | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        create_skill_version,
        skill_id=skill_id,
        files=files,
        default=default,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
def list_skill_versions(
    skill_id: str,
    after: str | None = None,
    limit: int | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="list_versions",
        skill_id=skill_id,
        after=after,
        limit=limit,
        order=order,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def alist_skill_versions(
    skill_id: str,
    after: str | None = None,
    limit: int | None = None,
    order: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        list_skill_versions,
        skill_id=skill_id,
        after=after,
        limit=limit,
        order=order,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
def get_skill_version(
    skill_id: str,
    version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="version",
        skill_id=skill_id,
        version=version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def aget_skill_version(
    skill_id: str,
    version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        get_skill_version,
        skill_id=skill_id,
        version=version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
def delete_skill_version(
    skill_id: str,
    version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="delete_version",
        skill_id=skill_id,
        version=version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def adelete_skill_version(
    skill_id: str,
    version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        delete_skill_version,
        skill_id=skill_id,
        version=version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
def get_skill_content(
    skill_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="content",
        skill_id=skill_id,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def aget_skill_content(
    skill_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        get_skill_content,
        skill_id=skill_id,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
def get_skill_version_content(
    skill_id: str,
    version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    return _run_skill_operation(
        operation="version_content",
        skill_id=skill_id,
        version=version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )


@client
async def aget_skill_version_content(
    skill_id: str,
    version: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Any:
    kwargs["_skill_is_async"] = True
    return await _call_skill_operation_async(
        get_skill_version_content,
        skill_id=skill_id,
        version=version,
        extra_headers=extra_headers,
        extra_query=extra_query,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        **kwargs,
    )
