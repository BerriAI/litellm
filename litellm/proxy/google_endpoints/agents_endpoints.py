"""
Google AI Studio Managed Agents API Proxy Endpoints.

Exposes Gemini's /v1beta/agents surface through the LiteLLM proxy so that
user curl commands transfer 1-to-1 by swapping the host + auth header.

Routes:
  POST   /v1beta/agents                     -> acreate_agent
  GET    /v1beta/agents                     -> alist_agents
  GET    /v1beta/agents/{name}              -> aget_agent
  DELETE /v1beta/agents/{name}              -> adelete_agent
  GET    /v1beta/agents/{name}/versions     -> alist_agent_versions

These are distinct from the A2A agent registry at /v1/agents.
"""

import json

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_query_params,
)

router = APIRouter(tags=["gemini managed agents"])


def _merge_query_params_into_data(data: dict, request: Request) -> dict:
    """
    For GET/DELETE endpoints that cannot carry a JSON body, read a
    JSON-encoded ``litellm_params_template`` query parameter and merge its
    contents into *data*, without overwriting keys that are already present
    (e.g. path params like ``name`` or the fixed ``custom_llm_provider``).

    This mirrors the ``litellm_params_template`` handling in
    ``create_gemini_agent`` and is the supported way for multi-tenant
    callers to supply per-request credentials on non-POST endpoints:

    .. code-block:: bash

        curl "http://localhost:4000/v1beta/agents?litellm_params_template=%7B%22api_key%22%3A%22AIza...%22%7D" \\
            -H "Authorization: Bearer sk-..."

    Credentials MUST NOT be passed as plain flat query parameters (e.g.
    ``?api_key=AIza...``) because URL query strings appear verbatim in
    web-server access logs, CDN edge logs, browser history, and Referer
    headers. Use the ``litellm_params_template`` JSON body field on POST
    requests, or the JSON-encoded query parameter above for GET/DELETE.
    """
    query_params = _safe_get_request_query_params(request)
    if not query_params:
        return data

    raw_template = query_params.get("litellm_params_template")
    if raw_template:
        try:
            template = (
                json.loads(raw_template)
                if isinstance(raw_template, str)
                else raw_template
            )
        except (json.JSONDecodeError, ValueError):
            template = {}
        if isinstance(template, dict):
            for key, value in template.items():
                data.setdefault(key, value)

    return data


def _proxy_server_imports():
    from litellm.proxy.proxy_server import (  # noqa: PLC0415
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    return dict(
        general_settings=general_settings,
        llm_router=llm_router,
        proxy_config=proxy_config,
        proxy_logging_obj=proxy_logging_obj,
        select_data_generator=select_data_generator,
        user_api_base=user_api_base,
        user_max_tokens=user_max_tokens,
        user_model=user_model,
        user_request_timeout=user_request_timeout,
        user_temperature=user_temperature,
        version=version,
    )


@router.post(
    "/v1beta/agents",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
)
async def create_gemini_agent(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a named custom agent on the Gemini side.

    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1beta/agents" \\
        -H "Authorization: Bearer sk-..." \\
        -H "Content-Type: application/json" \\
        -d '{
            "name": "my-custom-slides-agent",
            "base_agent": "waverunner",
            "instructions": "You are a helpful assistant that creates slides.",
            "base_environment": {
                "type": "remote",
                "sources": [
                    {"type": "gcs", "source": "gs://eap-templates/slides-skill",
                     "target": "/.agents/skills/slides-skill"}
                ]
            }
        }'
    ```
    """
    srv = _proxy_server_imports()
    data = await _read_request_body(request=request)
    # Merge litellm_params_template (e.g. custom_llm_provider, api_key) into the request
    litellm_params_template = data.pop("litellm_params_template", None) or {}
    if isinstance(litellm_params_template, dict):
        for key, value in litellm_params_template.items():
            if key not in data:
                data[key] = value
    data.setdefault("custom_llm_provider", "gemini")

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_agent",
            proxy_logging_obj=srv["proxy_logging_obj"],
            llm_router=srv["llm_router"],
            general_settings=srv["general_settings"],
            proxy_config=srv["proxy_config"],
            select_data_generator=srv["select_data_generator"],
            model=None,
            user_model=srv["user_model"],
            user_temperature=srv["user_temperature"],
            user_request_timeout=srv["user_request_timeout"],
            user_max_tokens=srv["user_max_tokens"],
            user_api_base=srv["user_api_base"],
            version=srv["version"],
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=srv["proxy_logging_obj"],
            version=srv["version"],
        )


@router.get(
    "/v1beta/agents",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
)
async def list_gemini_agents(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all custom agents on the Gemini side.

    Pass per-request Gemini credentials via ``litellm_params_template``
    (JSON-encoded) or as flat query parameters:

    ```bash
    curl "http://localhost:4000/v1beta/agents?api_key=AIza..." \\
        -H "Authorization: Bearer sk-..."
    ```
    """
    srv = _proxy_server_imports()
    data: dict = {"custom_llm_provider": "gemini"}
    _merge_query_params_into_data(data, request)

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_agents",
            proxy_logging_obj=srv["proxy_logging_obj"],
            llm_router=srv["llm_router"],
            general_settings=srv["general_settings"],
            proxy_config=srv["proxy_config"],
            select_data_generator=srv["select_data_generator"],
            model=None,
            user_model=srv["user_model"],
            user_temperature=srv["user_temperature"],
            user_request_timeout=srv["user_request_timeout"],
            user_max_tokens=srv["user_max_tokens"],
            user_api_base=srv["user_api_base"],
            version=srv["version"],
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=srv["proxy_logging_obj"],
            version=srv["version"],
        )


@router.get(
    "/v1beta/agents/{name}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
)
async def get_gemini_agent(
    request: Request,
    name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a specific custom agent by name.

    Pass per-request Gemini credentials via ``litellm_params_template``
    (JSON-encoded) or as flat query parameters:

    ```bash
    curl "http://localhost:4000/v1beta/agents/my-custom-slides-agent?api_key=AIza..." \\
        -H "Authorization: Bearer sk-..."
    ```
    """
    srv = _proxy_server_imports()
    data = {"name": name, "custom_llm_provider": "gemini"}
    _merge_query_params_into_data(data, request)

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_agent",
            proxy_logging_obj=srv["proxy_logging_obj"],
            llm_router=srv["llm_router"],
            general_settings=srv["general_settings"],
            proxy_config=srv["proxy_config"],
            select_data_generator=srv["select_data_generator"],
            model=None,
            user_model=srv["user_model"],
            user_temperature=srv["user_temperature"],
            user_request_timeout=srv["user_request_timeout"],
            user_max_tokens=srv["user_max_tokens"],
            user_api_base=srv["user_api_base"],
            version=srv["version"],
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=srv["proxy_logging_obj"],
            version=srv["version"],
        )


@router.delete(
    "/v1beta/agents/{name}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
)
async def delete_gemini_agent(
    request: Request,
    name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a custom agent by name.

    Pass per-request Gemini credentials via ``litellm_params_template``
    (JSON-encoded) or as flat query parameters:

    ```bash
    curl -X DELETE "http://localhost:4000/v1beta/agents/my-custom-slides-agent?api_key=AIza..." \\
        -H "Authorization: Bearer sk-..."
    ```
    """
    srv = _proxy_server_imports()
    data = {"name": name, "custom_llm_provider": "gemini"}
    _merge_query_params_into_data(data, request)

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_agent",
            proxy_logging_obj=srv["proxy_logging_obj"],
            llm_router=srv["llm_router"],
            general_settings=srv["general_settings"],
            proxy_config=srv["proxy_config"],
            select_data_generator=srv["select_data_generator"],
            model=None,
            user_model=srv["user_model"],
            user_temperature=srv["user_temperature"],
            user_request_timeout=srv["user_request_timeout"],
            user_max_tokens=srv["user_max_tokens"],
            user_api_base=srv["user_api_base"],
            version=srv["version"],
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=srv["proxy_logging_obj"],
            version=srv["version"],
        )


@router.get(
    "/v1beta/agents/{name}/versions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
)
async def list_gemini_agent_versions(
    request: Request,
    name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List versions of a custom agent.

    Pass per-request Gemini credentials via ``litellm_params_template``
    (JSON-encoded) or as flat query parameters:

    ```bash
    curl "http://localhost:4000/v1beta/agents/my-custom-slides-agent/versions?api_key=AIza..." \\
        -H "Authorization: Bearer sk-..."
    ```
    """
    srv = _proxy_server_imports()
    data = {"name": name, "custom_llm_provider": "gemini"}
    _merge_query_params_into_data(data, request)

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_agent_versions",
            proxy_logging_obj=srv["proxy_logging_obj"],
            llm_router=srv["llm_router"],
            general_settings=srv["general_settings"],
            proxy_config=srv["proxy_config"],
            select_data_generator=srv["select_data_generator"],
            model=None,
            user_model=srv["user_model"],
            user_temperature=srv["user_temperature"],
            user_request_timeout=srv["user_request_timeout"],
            user_max_tokens=srv["user_max_tokens"],
            user_api_base=srv["user_api_base"],
            version=srv["version"],
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=srv["proxy_logging_obj"],
            version=srv["version"],
        )
