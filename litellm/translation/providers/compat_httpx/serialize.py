"""Serializers for the httpx-path openai-compat shim family (wave 1b).

Every provider here goes through ``base_llm_http_handler`` with its config's
``transform_request`` LIVE — but none of the ten overrides it beyond the
two raw-guard concerns (heroku's content-list flatten, minimax's preserved
cache_control; cometapi's extra_body stub is a verified no-op), so the body
is ``openai_compat.assemble_body`` after the provider's gates plus the one
mechanical delta:

- ``rename_max_completion_tokens``: the OpenAILike super arm
  (bedrock_mantle / amazon_nova / datarobot / lemonade — lemonade via the
  get_optional_params OpenAILike else-arm today, see params.py). heroku /
  minimax / compactifai / ovhcloud / cometapi are OpenAIGPT-based copies
  (verbatim); gradient_ai's own map never renames (verbatim, mct in its
  list).

``reasoning_effort`` emission is derived from ``params.ALLOWED`` exactly
like compat_sdk (amazon_nova unconditional; bedrock_mantle
capability-narrowed by its gate). No provider here serves ``user``
(gradient_ai RAISES on it; the rest silently drop — both typed fallbacks).

The response side is per-provider DATA too (response.py): NO seam model
preset on this path (the xai R4 rule) — bare wire model everywhere except
the compactifai/amazon-nova/lemonade ``{prefix}/{REQUEST model}`` rewrites,
which are parser scope exactly like azure_ai's rename (but keyed on the
request model: v1 overwrites with ``f"{prefix}/{model}"`` after the
transform, ignoring the wire model).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]
Serializer = Callable[[ChatRequest, TranslationDeps], _SerializeResult]
_GateFn = Callable[[ChatRequest, TranslationDeps], str | None]


@dataclass(frozen=True)
class HttpxProfile:
    provider: p.CompatHttpxProvider
    unsupported: _GateFn
    rename_max_completion_tokens: bool = False
    response_model_prefix: str | None = None
    """v1 ``transform_response`` overwrites the response model with
    ``f"{prefix}/{REQUEST model}"`` (compactifai/, amazon-nova/ — note the
    HYPHEN, not the provider slug — and lemonade/); None = bare wire model
    (the base/OpenAILike transforms with custom_llm_provider=None)."""


def serialize_with_profile(
    request: ChatRequest, deps: TranslationDeps, profile: HttpxProfile
) -> _SerializeResult:
    reason = profile.unsupported(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(lambda body: _with_deltas(body, request, profile))


def _with_deltas(body: Body, request: ChatRequest, profile: HttpxProfile) -> Body:
    if profile.rename_max_completion_tokens and "max_completion_tokens" in body:
        body = {
            **{k: v for k, v in body.items() if k != "max_completion_tokens"},
            "max_tokens": body["max_completion_tokens"],
        }
    if "reasoning_effort" not in p.ALLOWED[profile.provider]:
        return body
    effort = request.reasoning_effort.default_value(None)
    if effort is None:
        return body
    return {**body, "reasoning_effort": effort}


_PROFILE_ROWS: tuple[HttpxProfile, ...] = (
    HttpxProfile(provider="heroku", unsupported=p.heroku_unsupported),
    HttpxProfile(
        provider="bedrock_mantle",
        unsupported=p.bedrock_mantle_unsupported,
        rename_max_completion_tokens=True,
    ),
    HttpxProfile(provider="minimax", unsupported=p.minimax_unsupported),
    HttpxProfile(
        provider="compactifai",
        unsupported=p.compactifai_unsupported,
        response_model_prefix="compactifai",
    ),
    HttpxProfile(
        provider="amazon_nova",
        unsupported=p.amazon_nova_unsupported,
        rename_max_completion_tokens=True,
        response_model_prefix="amazon-nova",
    ),
    HttpxProfile(
        provider="datarobot",
        unsupported=p.datarobot_unsupported,
        rename_max_completion_tokens=True,
    ),
    HttpxProfile(provider="gradient_ai", unsupported=p.gradient_ai_unsupported),
    HttpxProfile(provider="ovhcloud", unsupported=p.ovhcloud_unsupported),
    HttpxProfile(
        provider="lemonade",
        unsupported=p.lemonade_unsupported,
        rename_max_completion_tokens=True,
        response_model_prefix="lemonade",
    ),
    HttpxProfile(provider="cometapi", unsupported=p.cometapi_unsupported),
)

PROFILES: Mapping[p.CompatHttpxProvider, HttpxProfile] = MappingProxyType(
    {profile.provider: profile for profile in _PROFILE_ROWS}
)


def _profile_serializer(profile: HttpxProfile) -> Serializer:
    def serialize(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
        return serialize_with_profile(request, deps, profile)

    return serialize


SERIALIZERS: Mapping[p.CompatHttpxProvider, Serializer] = MappingProxyType(
    {provider: _profile_serializer(profile) for provider, profile in PROFILES.items()}
)
