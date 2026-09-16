"""Interactions over agent sessions: get reads the session object and its transcript in two requests,
and cancel names the session it interrupted.
"""

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final

import httpx

from litellm.interactions.http_handler import InteractionsHTTPHandler
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.interactions.session_transformation import BaseSessionInteractionsConfig
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.interactions import CancelInteractionResult, InteractionsAPIResponse
from litellm.types.router import GenericLiteLLMParams

_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


def _session_config(config: BaseInteractionsAPIConfig) -> BaseSessionInteractionsConfig:
    if isinstance(config, BaseSessionInteractionsConfig):
        return config
    raise TypeError(f"{type(config).__name__} is not a session-backed interactions config")


def _with_id(result: CancelInteractionResult, interaction_id: str) -> CancelInteractionResult:
    return result if result.id else result.model_copy(update=MappingProxyType({"id": interaction_id}))


class SessionInteractionsHTTPHandler(InteractionsHTTPHandler):
    def get_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: dict[str, str] | None = None,  # mutable-ok: InteractionsHTTPHandler signature
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
        _is_async: bool = False,
    ) -> InteractionsAPIResponse | Coroutine[object, object, InteractionsAPIResponse]:
        if _is_async:
            return self.async_get_interaction(
                interaction_id=interaction_id,
                interactions_api_config=interactions_api_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )
        config: Final = _session_config(interactions_api_config)
        headers: Final = config.validate_environment(
            headers=extra_headers or _NO_HEADERS, model="", litellm_params=litellm_params
        )
        api_base: Final = litellm_params.api_base or ""
        session_url, session_params = config.transform_get_interaction_request(
            interaction_id=interaction_id, api_base=api_base, litellm_params=litellm_params, headers=headers
        )
        transcript_url, transcript_params = config.transform_get_transcript_request(
            interaction_id=interaction_id, api_base=api_base, litellm_params=litellm_params
        )
        logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # Logging.pre_call is untyped
            input=interaction_id,
            api_key="",
            additional_args={"api_base": session_url, "headers": headers},  # mutable-ok: pre_call takes a dict
        )
        http_client: Final = self._sync_client(litellm_params, client)
        try:
            session_response: Final = http_client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get exposes untyped optional mappings
                url=session_url, headers=headers, params=session_params
            )
            transcript_response: Final = http_client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get exposes untyped optional mappings
                url=transcript_url,
                headers=headers,
                params=dict(transcript_params),  # mutable-ok: httpx params= needs a plain dict
            )
        except httpx.HTTPError as e:
            raise self._handle_error(e=e, provider_config=config)
        return config.assemble_get_response(
            session_response=session_response, transcript_response=transcript_response, logging_obj=logging_obj
        )

    async def async_get_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: dict[str, str] | None = None,  # mutable-ok: InteractionsHTTPHandler signature
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> InteractionsAPIResponse:
        config: Final = _session_config(interactions_api_config)
        headers: Final = config.validate_environment(
            headers=extra_headers or _NO_HEADERS, model="", litellm_params=litellm_params
        )
        api_base: Final = litellm_params.api_base or ""
        session_url, session_params = config.transform_get_interaction_request(
            interaction_id=interaction_id, api_base=api_base, litellm_params=litellm_params, headers=headers
        )
        transcript_url, transcript_params = config.transform_get_transcript_request(
            interaction_id=interaction_id, api_base=api_base, litellm_params=litellm_params
        )
        logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # Logging.pre_call is untyped
            input=interaction_id,
            api_key="",
            additional_args={"api_base": session_url, "headers": headers},  # mutable-ok: pre_call takes a dict
        )
        http_client: Final = self._async_client(litellm_params, client)
        try:
            session_response: Final = await http_client.get(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.get exposes untyped optional mappings
                url=session_url, headers=headers, params=session_params
            )
            transcript_response: Final = await http_client.get(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.get exposes untyped optional mappings
                url=transcript_url,
                headers=headers,
                params=dict(transcript_params),  # mutable-ok: httpx params= needs a plain dict
            )
        except httpx.HTTPError as e:
            raise self._handle_error(e=e, provider_config=config)
        return config.assemble_get_response(
            session_response=session_response, transcript_response=transcript_response, logging_obj=logging_obj
        )

    def cancel_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: dict[str, str] | None = None,  # mutable-ok: InteractionsHTTPHandler signature
        timeout: float | httpx.Timeout | None = None,
        client: HTTPHandler | None = None,
        _is_async: bool = False,
    ) -> CancelInteractionResult | Coroutine[object, object, CancelInteractionResult]:
        if _is_async:
            return self.async_cancel_interaction(
                interaction_id=interaction_id,
                interactions_api_config=interactions_api_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )
        result: Final = super().cancel_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            client=client,
        )
        if not isinstance(result, CancelInteractionResult):
            raise TypeError("the synchronous cancel returned a coroutine")
        return _with_id(result, interaction_id)

    async def async_cancel_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: dict[str, str] | None = None,  # mutable-ok: InteractionsHTTPHandler signature
        timeout: float | httpx.Timeout | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> CancelInteractionResult:
        result: Final = await super().async_cancel_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            client=client,
        )
        return _with_id(result, interaction_id)


session_interactions_http_handler: Final = SessionInteractionsHTTPHandler()
