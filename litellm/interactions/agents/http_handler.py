"""
HTTP handler for the Agents API.

Extends InteractionsHTTPHandler so that the shared HTTP infrastructure
(_handle_error, _sync_client, _async_client) is reused rather than
duplicated. BaseAgentsAPIConfig stays as pure transform code.
"""

from typing import Any, Coroutine, Dict, Optional, Union

import httpx

from litellm.constants import request_timeout
from litellm.interactions.http_handler import InteractionsHTTPHandler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.agents import (
    AgentCreateResponse,
    AgentDeleteResult,
    AgentListResponse,
    AgentVersionsResponse,
)
from litellm.types.router import GenericLiteLLMParams


class AgentsHTTPHandler(InteractionsHTTPHandler):
    """HTTP handler for Agents API CRUD requests."""

    # ------------------------------------------------------------------ #
    # CREATE                                                               #
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

        sync_httpx_client = self._sync_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url = agents_api_config.get_complete_url(
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        data = agents_api_config.transform_create_request(
            name=name, litellm_params=dict(litellm_params)
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
                url=url, headers=headers, json=data, timeout=timeout or request_timeout
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        return agents_api_config.transform_create_response(
            raw_response=response, name=name
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
        async_httpx_client = self._async_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url = agents_api_config.get_complete_url(
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        data = agents_api_config.transform_create_request(
            name=name, litellm_params=dict(litellm_params)
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
                url=url, headers=headers, json=data, timeout=timeout or request_timeout
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        return agents_api_config.transform_create_response(
            raw_response=response, name=name
        )

    # ------------------------------------------------------------------ #
    # LIST                                                                 #
    # ------------------------------------------------------------------ #

    def list_agents(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[AgentListResponse, Coroutine[Any, Any, AgentListResponse]]:
        if _is_async:
            return self.async_list_agents(
                agents_api_config=agents_api_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        sync_httpx_client = self._sync_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url, params = agents_api_config.transform_list_request(
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input="list_agents",
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = sync_httpx_client.get(url=url, headers=headers, params=params)
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_list_response(raw_response=response)

    async def async_list_agents(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AgentListResponse:
        async_httpx_client = self._async_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url, params = agents_api_config.transform_list_request(
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input="list_agents",
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=params
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_list_response(raw_response=response)

    # ------------------------------------------------------------------ #
    # GET                                                                  #
    # ------------------------------------------------------------------ #

    def get_agent(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[AgentCreateResponse, Coroutine[Any, Any, AgentCreateResponse]]:
        if _is_async:
            return self.async_get_agent(
                agents_api_config=agents_api_config,
                name=name,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        sync_httpx_client = self._sync_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url, params = agents_api_config.transform_get_request(
            name=name,
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = sync_httpx_client.get(url=url, headers=headers, params=params)
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_get_response(
            raw_response=response, name=name
        )

    async def async_get_agent(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AgentCreateResponse:
        async_httpx_client = self._async_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url, params = agents_api_config.transform_get_request(
            name=name,
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=params
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_get_response(
            raw_response=response, name=name
        )

    # ------------------------------------------------------------------ #
    # DELETE                                                               #
    # ------------------------------------------------------------------ #

    def delete_agent(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[AgentDeleteResult, Coroutine[Any, Any, AgentDeleteResult]]:
        if _is_async:
            return self.async_delete_agent(
                agents_api_config=agents_api_config,
                name=name,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        sync_httpx_client = self._sync_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url = agents_api_config.transform_delete_request(
            name=name,
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = sync_httpx_client.delete(
                url=url, headers=headers, timeout=timeout or request_timeout
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_delete_response(
            raw_response=response, name=name
        )

    async def async_delete_agent(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AgentDeleteResult:
        async_httpx_client = self._async_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url = agents_api_config.transform_delete_request(
            name=name,
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = await async_httpx_client.delete(
                url=url, headers=headers, timeout=timeout or request_timeout
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_delete_response(
            raw_response=response, name=name
        )

    # ------------------------------------------------------------------ #
    # LIST VERSIONS                                                        #
    # ------------------------------------------------------------------ #

    def list_agent_versions(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[AgentVersionsResponse, Coroutine[Any, Any, AgentVersionsResponse]]:
        if _is_async:
            return self.async_list_agent_versions(
                agents_api_config=agents_api_config,
                name=name,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        sync_httpx_client = self._sync_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url, params = agents_api_config.transform_list_versions_request(
            name=name,
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = sync_httpx_client.get(url=url, headers=headers, params=params)
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_list_versions_response(
            raw_response=response, name=name
        )

    async def async_list_agent_versions(
        self,
        agents_api_config: BaseAgentsAPIConfig,
        name: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AgentVersionsResponse:
        async_httpx_client = self._async_client(litellm_params, client)
        headers = agents_api_config.validate_environment(
            headers=extra_headers or {}, litellm_params=dict(litellm_params)
        )
        url, params = agents_api_config.transform_list_versions_request(
            name=name,
            api_base=litellm_params.get("api_base"),
            litellm_params=dict(litellm_params),
        )
        logging_obj.pre_call(
            input=name,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )
        try:
            response = await async_httpx_client.get(
                url=url, headers=headers, params=params
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=agents_api_config)

        logging_obj.post_call(original_response=response.text, additional_args={})
        return agents_api_config.transform_list_versions_response(
            raw_response=response, name=name
        )


agents_http_handler = AgentsHTTPHandler()
