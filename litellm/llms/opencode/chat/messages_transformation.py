"""
OpenCode Anthropic Messages wire-format config.

Routes models in the surface's messages-model set to ``{base}/v1/messages``
with Anthropic Messages body shape.  Both Zen and Go authenticate
``/v1/messages`` with ``x-api-key`` (Anthropic default); verified live that
Bearer returns 401 "Missing API key" on both surfaces.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final  # noqa: TID251  # Anthropic Messages wire format uses Any in param/return shapes

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.opencode.common_utils import (
    OpenCodeException,
    cost_map_max_output_tokens,
    resolve_opencode_api_base,
    resolve_opencode_api_key,
)
from litellm.types.router import GenericLiteLLMParams

# ---------- surface base URL ( /v1/messages appended downstream ) ----------

ZEN_MESSAGES_BASE: Final = "https://opencode.ai/zen"

# ------------------------------------------------------------------- model sets
# Models the gateway serves via the Anthropic Messages wire format, per surface.
# Source: models.dev ``npm == @ai-sdk/anthropic`` classification.
#
# These live in code rather than being read from the cost map's ``mode`` field
# because ``litellm.model_cost`` is fetched from the published remote map at
# import, and a provider's entries only appear there once released. Routing that
# depended on them would send every model below to the chat arm — the wrong wire
# format — on any install whose cost map predates this provider.

OPENCODE_MESSAGES_MODELS: Final = MappingProxyType(
    {
        "zen": frozenset(
            {
                "claude-fable-5",
                "claude-haiku-4-5",
                "claude-opus-4-5",
                "claude-opus-4-6",
                "claude-opus-4-7",
                "claude-opus-4-8",
                "claude-opus-5",
                "claude-sonnet-4",
                "claude-sonnet-4-5",
                "claude-sonnet-4-6",
                "claude-sonnet-5",
                "qwen3.5-plus",
                "qwen3.6-plus",
            }
        ),
        # The qwen3.{5..8}-{plus,max} grid is listed in full rather than only the
        # entries the cost map carries today, so a gateway-side addition keeps
        # routing to the right wire format without waiting on a release.
        "go": frozenset(
            {
                "minimax-m2.5",
                "minimax-m2.7",
                "minimax-m3",
                "qwen3.8-flash",
                *(f"qwen3.{n}-{tier}" for n in range(5, 9) for tier in ("plus", "max")),
            }
        ),
    }
)


def is_messages_model(surface: str, model: str) -> bool:
    """Return True when *model* belongs on the messages arm of *surface*.

    Strips the ``opencode_{surface}/`` prefix when the caller passes the
    fully-qualified model name (e.g. ``opencode_zen/claude-sonnet-4``). An
    unrecognised model is not on the messages arm, so it falls through to chat
    completions.
    """
    bare: Final = model.rsplit("/", 1)[-1]
    return bare in OPENCODE_MESSAGES_MODELS.get(surface, frozenset())


# ------------------------------------------------------------------ config class


class OpenCodeMessagesConfig(AnthropicMessagesConfig):
    """Anthropic Messages config for the OpenCode gateway.

    Parameters
    ----------
    surface :
        ``"zen"`` (default) or ``"go"``.  Determines the base URL.
    """

    def __init__(self, surface: str = "zen") -> None:
        self.surface: Final = surface

    @property
    def custom_llm_provider(self) -> str:
        return f"opencode_{self.surface}"

    def should_strip_billing_metadata(self) -> bool:
        """OpenCode is a third-party gateway, not the first-party Anthropic API;
        x-anthropic-billing-header client attribution must not leak to it."""
        return True

    def _base_url(self) -> str:
        """Surface default, used only when nothing is configured.

        ``/go`` is part of the Go surface's default host layout, so it belongs
        on the default rather than being appended to whatever base the operator
        configured (which would rewrite a private gateway to ``{gateway}/go``).
        """
        return f"{ZEN_MESSAGES_BASE}/go" if self.surface == "go" else ZEN_MESSAGES_BASE

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        stream: bool | None = None,
    ) -> str:
        """Return ``{api_base}/v1/messages``."""
        base: Final = (resolve_opencode_api_base(self.surface, api_base) or self._base_url()).rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        model: str,
        messages: list[Any],  # mutable-ok: signature must match AnthropicMessagesConfig
        optional_params: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        litellm_params: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:  # mutable-ok: signature must match AnthropicMessagesConfig
        """Resolve key / base URL and let the base class inject the auth header.

        Both surfaces authenticate ``/v1/messages`` with ``x-api-key``
        (Anthropic default); Bearer returns 401 "Missing API key" on both.
        """
        # -- key resolution (same chain as chat arm) ---------------------------
        # A missing key is rejected here: the base class resolves one from
        # ANTHROPIC_API_KEY, which would send a first-party Anthropic
        # credential to opencode.ai.
        key: Final = resolve_opencode_api_key(self.surface, api_key)
        if key is None:
            raise ValueError(
                f"OpenCode API key is required. Set OPENCODE_{self.surface.upper()}_API_KEY "
                f"or OPENCODE_API_KEY, or pass api_key."
            )

        base_url: Final = resolve_opencode_api_base(self.surface, api_base) or self._base_url()

        # -- auth header per surface -------------------------------------------
        # OpenCode does not support OAuth, so we can skip the OAuth check that
        # the base class performs.  Both surfaces authenticate /v1/messages with
        # x-api-key (verified live: Bearer returns 401 "Missing API key" on both
        # Zen and Go), so leave headers empty and let the base class inject
        # x-api-key.
        # NOTE: this intentionally diverges from the chat arm, which uses Bearer.

        # -- base class handles defaults, beta headers, content-type ----------
        resolved_headers, resolved_base_url = super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=key,
            api_base=base_url,
        )
        return resolved_headers, resolved_base_url

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: signature must match AnthropicMessagesConfig
    ) -> BaseLLMException:
        return OpenCodeException(message=error_message, status_code=status_code, headers=headers)

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],  # mutable-ok: signature must match AnthropicMessagesConfig
        anthropic_messages_optional_request_params: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
    ) -> dict:  # mutable-ok: signature must match AnthropicMessagesConfig
        """Default ``max_tokens`` from the cost map before the base class runs.

        The Anthropic ``/v1/messages`` API requires ``max_tokens``, but the
        messages arm receives ``optional_params`` straight from the caller
        (e.g. a playground wildcard request with no explicit ``max_tokens``).
        The base class raises if it is absent, so default it from the model's
        cost-map ``max_output_tokens`` here.  ``model`` arrives bare (e.g.
        ``qwen3.7-plus``), so qualify it with the surface prefix for the
        ``litellm.model_cost`` lookup.
        """
        default_max_tokens: Final = (
            cost_map_max_output_tokens(surface=self.surface, model=model)
            if anthropic_messages_optional_request_params.get("max_tokens") is None
            else None
        )
        params: Final = (
            {  # mutable-ok: base config requires a mutable dict
                **anthropic_messages_optional_request_params,
                "max_tokens": default_max_tokens,
            }
            if default_max_tokens is not None
            else anthropic_messages_optional_request_params
        )

        return super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers=headers,
        )
