"""
HTTP handler for the Agents API.

Mirrors InteractionsHTTPHandler — owns all HTTP logic so that
BaseAgentsAPIConfig stays as pure transform code.
"""

from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.agents import AgentCreateResponse
from litellm.types.router import GenericLiteLLMParams


class AgentsHTTPHandler:
    """HTTP handler for Agents API requests."""

    def _handle_error(
        self,
        e: Exception,
        provider_config: BaseAgentsAPIConfig,
    ) -> Exception:
        if isinstance(e, httpx.HTTPStatusError):
            error_message = e.response.text
            status_code = e.response.status_code
            headers = dict(e.response.headers)
            return provider_config.get_error_class(
                error_message=error_message,
                status_code=status_code,
                headers=headers,
            )
        return e

    # ------------------------------------------------------------------ #
    # CREATE AGENT                                                         #
    # ------------------------------------------------------------------ #

    def create_agent(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[AgentCreateResponse, Coroutine[Any, Any, AgentCreateResponse]]:
        """
        Create an agent — sync or async depending on _is_async.

        Follows the same dispatch pattern as InteractionsHTTPHandler.create_interaction.
        """
        if _is_async:
            return self.async_create_agent(
                agents_api_config=agents_api_config,
                name=name,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
            )

        sync_httpx_client = client or _get_httpx_client(
            params={"ssl_verify": litellm_params.get("ssl_verify", None)}
        )

        headers = agents_api_config.validate_environment(
            headers=extra_headers or {},
            litellm_params=dict(litellm_params),
        )
        url = agents_api_config.get_complete_url(
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        data = agents_api_config.transform_create_request(
            name=name,
            litellm_params=dict(litellm_params),
        )
        if extra_body:
            data.update(extra_body)

        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = sync_httpx_client.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )

        return agents_api_config.transform_create_response(
            raw_response=response,
            name=name,
        )

    async def async_create_agent(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AgentCreateResponse:
        """Async version of create_agent."""
        async_httpx_client = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.GEMINI,
            params={"ssl_verify": litellm_params.get("ssl_verify", None)},
        )

        headers = agents_api_config.validate_environment(
            headers=extra_headers or {},
            litellm_params=dict(litellm_params),
        )
        url = agents_api_config.get_complete_url(
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        data = agents_api_config.transform_create_request(
            name=name,
            litellm_params=dict(litellm_params),
        )
        if extra_body:
            data.update(extra_body)

        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = await async_httpx_client.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )

        return agents_api_config.transform_create_response(
            raw_response=response,
            name=name,
        )


agents_http_handler = AgentsHTTPHandler()
