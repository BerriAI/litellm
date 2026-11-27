from collections.abc import Mapping
from typing import Final

from litellm.constants import DROP_UNSUPPORTED_PARAMS_HINT
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.llms.anthropic.messages.api_versions import requested_api_version
from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams

from ....vertex_llm_base import VertexBase
from ..output_params_utils import sanitize_vertex_anthropic_output_params

_VERSION_HEADER_VERTEX_READS_FROM_BODY: Final = "anthropic-version"
_VERTEX_DEFAULT_VERSION: Final = "vertex-2023-10-16"
_VERSIONS_SERVED_AS_VERTEX_DEFAULT: Final = frozenset({"", "2023-01-01", "2023-06-01"})
_VERSIONS_VERTEX_ACCEPTS_IN_BODY: Final = frozenset({_VERTEX_DEFAULT_VERSION, "2026-10-21"})


def _vertex_version(headers: Mapping[str, object], model: str, drop_params: bool) -> str | UnsupportedParamsError:
    requested: Final = requested_api_version(headers) or ""
    if requested in _VERSIONS_SERVED_AS_VERTEX_DEFAULT:
        return _VERTEX_DEFAULT_VERSION
    if requested in _VERSIONS_VERTEX_ACCEPTS_IN_BODY:
        return requested
    if drop_params:
        return _VERTEX_DEFAULT_VERSION
    return UnsupportedParamsError(
        message=(
            f"{model} does not support {_VERSION_HEADER_VERTEX_READS_FROM_BODY}={requested}. {DROP_UNSUPPORTED_PARAMS_HINT}"
        ),
        llm_provider="vertex_ai",
        model=model,
    )


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    @property
    def custom_llm_provider(self) -> str | None:
        return "vertex_ai"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def supports_diagnostics(self) -> bool:
        return False

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict[str, object]],
        anthropic_messages_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, object],  # mutable-ok: mirrors the provider transform contract
    ) -> dict[str, object]:
        anthropic_messages_request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        self._normalize_system_role_messages(
            anthropic_messages_request, model=model, drop_params=litellm_params.drop_params
        )

        self._remove_scope_from_cache_control(anthropic_messages_request)

        version: Final = _vertex_version(
            headers, model=model, drop_params=drops_unsupported_params(litellm_params.drop_params)
        )
        if isinstance(version, UnsupportedParamsError):
            raise version
        anthropic_messages_request["anthropic_version"] = version

        anthropic_messages_request.pop("model", None)  # do not pass model in request body to vertex ai

        sanitize_vertex_anthropic_output_params(anthropic_messages_request, model)

        return anthropic_messages_request
