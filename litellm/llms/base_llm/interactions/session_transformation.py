"""Interactions API over a provider whose unit of work is a long-lived agent session: reading an
interaction back needs the session object plus its transcript.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import httpx

from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class BaseSessionInteractionsConfig(BaseInteractionsAPIConfig):
    @abstractmethod
    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:  # mutable-ok: BaseInteractionsAPIConfig.validate_environment signature
        """Auth and beta headers for every session request."""

    @abstractmethod
    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseInteractionsAPIConfig signature
        """``(url, query_params)`` for the session object."""

    @abstractmethod
    def transform_get_transcript_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
    ) -> tuple[str, Mapping[str, object]]:
        """``(url, query_params)`` for the session transcript read next to the session object."""

    @abstractmethod
    def assemble_get_response(
        self,
        session_response: httpx.Response,
        transcript_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        """Fold the session object and its transcript into one interaction."""
