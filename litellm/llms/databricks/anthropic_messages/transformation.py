"""
Databricks Unity AI Gateway — native Anthropic Messages transformation config.

Routes ``litellm.anthropic.messages(model="databricks/<claude-endpoint>")`` to the
gateway's native Anthropic Messages surface:

    https://<workspace-host>/ai-gateway/anthropic/v1/messages

Unlike the OpenAI-chat coercion used by the default Databricks chat path, this
preserves the native Anthropic request/response wire format — prompt caching,
``thinking`` blocks, and ``tool_use`` content survive end to end. The native
Anthropic surface exists only on the AI Gateway, so this config always targets
the gateway (the workspace host is recovered from any provided ``api_base``).

Auth reuses the shared Databricks credential resolution (M2M / PAT / SDK unified)
via :class:`DatabricksBase`, emitting ``Authorization: Bearer <token>`` rather
than Anthropic's ``x-api-key``.
"""

from typing import Any, List, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    DEFAULT_ANTHROPIC_API_VERSION,
    AnthropicMessagesConfig,
)

from ..ai_gateway import build_anthropic_messages_url
from ..common_utils import DatabricksBase


class DatabricksAnthropicMessagesConfig(DatabricksBase, AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "databricks"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        resolved_base = self._get_api_base(api_base)
        return build_anthropic_messages_url(resolved_base)

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        custom_user_agent = (
            optional_params.pop("user_agent", None)
            or optional_params.pop("databricks_user_agent", None)
            or litellm_params.get("user_agent")
        )

        databricks_profile = optional_params.pop(
            "databricks_profile", None
        ) or self.resolve_databricks_profile(litellm_params)

        # Reuse the shared Databricks auth core (M2M / PAT / PROFILE / SDK unified).
        # It sets `Authorization: Bearer <token>` and never appends an endpoint path.
        _, headers = self.databricks_resolve_auth(
            api_key=api_key,
            api_base=api_base,
            custom_endpoint=False,
            headers=headers,
            custom_user_agent=custom_user_agent,
            databricks_profile=databricks_profile,
        )

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
            custom_llm_provider=self.custom_llm_provider or "databricks",
        )

        headers = self.apply_request_tags_header(
            headers, optional_params=optional_params, litellm_params=litellm_params
        )

        return headers, api_base
