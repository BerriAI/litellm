"""
Main entry point for Skills API operations
Provides create, list, get, and delete operations for skills
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
DEFAULT_ANTHROPIC_API_BASE: Final = "https://api.anthropic.com/v1"

# Initialize LiteLLM skills handler (lazy - only used when custom_llm_provider="litellm")
_litellm_skills_handler = None


def _get_user_api_key_auth_from_kwargs(kwargs: dict[str, Any]) -> Any | None:
    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = kwargs.get(metadata_key)
        if isinstance(metadata, dict) and metadata.get("user_api_key_auth") is not None:
            return metadata["user_api_key_auth"]
    return None


def _get_skill_request_metadata(
    kwargs: dict[str, Any],
    extra_body: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if extra_body and isinstance(extra_body.get("metadata"), dict):
        return extra_body["metadata"]

    metadata: Final = kwargs.get("metadata")
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
    files: list[Any] | None = None,
    display_title: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
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
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["acreate_skill"] = True

        func: Final = partial(
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
def create_skill(
    files: list[Any] | None = None,
    display_title: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Skill | Coroutine[Any, Any, Skill]:
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
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("acreate_skill", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

        # Build create request
        create_request: Final[CreateSkillRequest] = {}
        if display_title is not None:
            create_request["display_title"] = display_title
        if files is not None:
            create_request["files"] = files

        # Merge extra_body if provided
        if extra_body:
            create_request.update(extra_body)

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
        skills_api_provider_config: BaseSkillsAPIConfig | None = ProviderConfigManager.get_provider_skills_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if skills_api_provider_config is None:
            raise ValueError(f"CREATE skill is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Transform request
        request_body: Final = skills_api_provider_config.transform_create_skill_request(
            create_request=create_request,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Get API base and URL
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        api_base: Final = AnthropicModelInfo.get_api_base(litellm_params.api_base)
        url: Final = skills_api_provider_config.get_complete_url(api_base=api_base, endpoint="skills")

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
        response: Final = base_llm_http_handler.create_skill_handler(
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
async def alist_skills(
    limit: int | None = None,
    page: str | None = None,
    source: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
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
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["alist_skills"] = True

        func: Final = partial(
            list_skills,
            limit=limit,
            page=page,
            source=source,
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
def list_skills(
    limit: int | None = None,
    page: str | None = None,
    source: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> ListSkillsResponse | Coroutine[Any, Any, ListSkillsResponse]:
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
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("alist_skills", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

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
        skills_api_provider_config: BaseSkillsAPIConfig | None = ProviderConfigManager.get_provider_skills_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if skills_api_provider_config is None:
            raise ValueError(f"LIST skills is not supported for {custom_llm_provider}")

        # Build list parameters
        list_params: Final[ListSkillsParams] = {}
        if limit is not None:
            list_params["limit"] = limit
        if page is not None:
            list_params["page"] = page
        if source is not None:
            list_params["source"] = source

        # Merge extra_query if provided
        if extra_query:
            list_params.update(extra_query)

        # Validate environment and get headers
        headers = extra_headers or {}
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
        response: Final = base_llm_http_handler.list_skills_handler(
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
async def aget_skill(
    skill_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
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
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["aget_skill"] = True

        func: Final = partial(
            get_skill,
            skill_id=skill_id,
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
def get_skill(
    skill_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> Skill | Coroutine[Any, Any, Skill]:
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
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("aget_skill", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

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
        skills_api_provider_config: BaseSkillsAPIConfig | None = ProviderConfigManager.get_provider_skills_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if skills_api_provider_config is None:
            raise ValueError(f"GET skill is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Get API base
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        api_base: Final = AnthropicModelInfo.get_api_base(litellm_params.api_base)

        # Transform request
        url, headers = skills_api_provider_config.transform_get_skill_request(
            skill_id=skill_id,
            api_base=api_base or DEFAULT_ANTHROPIC_API_BASE,
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
        response: Final = base_llm_http_handler.get_skill_handler(
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
async def adelete_skill(
    skill_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
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
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["adelete_skill"] = True

        func: Final = partial(
            delete_skill,
            skill_id=skill_id,
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
def delete_skill(
    skill_id: str,
    extra_headers: dict[str, Any] | None = None,
    extra_query: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> DeleteSkillResponse | Coroutine[Any, Any, DeleteSkillResponse]:
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
    local_vars: Final = locals()
    try:
        litellm_logging_obj: Final[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)
        _is_async: Final = kwargs.pop("adelete_skill", False) is True

        # Get LiteLLM parameters
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        # Determine provider
        if custom_llm_provider is None:
            custom_llm_provider = "anthropic"

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
        skills_api_provider_config: BaseSkillsAPIConfig | None = ProviderConfigManager.get_provider_skills_api_config(
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if skills_api_provider_config is None:
            raise ValueError(f"DELETE skill is not supported for {custom_llm_provider}")

        # Validate environment and get headers
        headers = extra_headers or {}
        headers = skills_api_provider_config.validate_environment(headers=headers, litellm_params=litellm_params)

        # Get API base
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        api_base: Final = AnthropicModelInfo.get_api_base(litellm_params.api_base)

        # Transform request
        url, headers = skills_api_provider_config.transform_delete_skill_request(
            skill_id=skill_id,
            api_base=api_base or DEFAULT_ANTHROPIC_API_BASE,
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
        response: Final = base_llm_http_handler.delete_skill_handler(
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

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
