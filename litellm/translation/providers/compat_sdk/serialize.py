"""Serializers for the SDK-path openai-compat family (wave 1a).

Every provider here rides v1's big openai elif (main.py:2646-2667) into the
OpenAI SDK: ``get_optional_params`` runs the provider config's param gates,
``provider_config.transform_request`` (openai.py:727) runs the inherited
base five-touch assembly, and none of the family overrides it. The v2 body
is therefore ``openai_compat.assemble_body`` after the provider's gates,
plus at most two mechanical deltas captured per provider in a frozen
``CompatProfile``:

- ``rename_max_completion_tokens``: the configs whose map renames mct ->
  max_tokens (cerebras/nvidia_nim/nebius/wandb/featherless_ai and the
  OpenAILike-based lambda_ai/volcengine).
- ``drop_text_response_format``: together_ai's map pops a verbatim
  ``{"type": "text"}`` response_format.

``user`` and ``reasoning_effort`` emission is NOT profile data: it is
derived from ``params.ALLOWED[provider]`` — the same single source the
gates narrow — so the emission facts can never desync from the gate facts
(critic-wave1a M3).

The registration surface is data too: ``PROFILES`` is the family registry
and ``SERIALIZERS`` its derived serializer table; ``engine/pipeline.py``
splices both so adding a provider to this family costs a profile row here,
an allowed set + ALLOWED row in params.py, and one dispatch Literal line
(critic-wave1a M1).

The response side needs NO per-provider code: the live v1 normalizer is the
same ``convert_to_model_response_object`` the openai_compat parser mirrors,
and the ``{provider}/{wire_model}`` model re-prefix (openai.py:676-677 +
cdr:699-710) is the SEAM's ``_to_model_response_openai`` preset arm, pinned
per provider by the differential's preset-model rows. Streams ride the
default openai wrapper arm -> the ``"openai"`` chunk dialect (baseten is the
one would-be member that does NOT, and is dropped from the wave for it).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from expression import Error, Result

from litellm.constants import MIN_NON_ZERO_TEMPERATURE

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body
from . import params as p
from .checks import base_list_unsupported
from .json_registry import JSON_RENAME, json_registry_unsupported

_SerializeResult = Result[Body, TranslationError]
Serializer = Callable[[ChatRequest, TranslationDeps], _SerializeResult]
_GateFn = Callable[[ChatRequest, TranslationDeps], str | None]
_RewriteFn = Callable[[Body, ChatRequest, TranslationDeps], Body]


@dataclass(frozen=True)
class CompatProfile:
    provider: p.CompatSdkProvider
    unsupported: _GateFn
    rename_max_completion_tokens: bool = False
    drop_text_response_format: bool = False
    drop_non_json_schema_response_format: bool = False
    """meta_llama: v1's map POPS response_format unless type == json_schema
    (LlamaAPIConfig.map_openai_params) — json_object is silently dropped."""
    # wave-2a named deltas (append-only; defaults inert for wave-1a/1b rows):
    drop_tool_choice: bool = False
    """deepinfra: v1's tool_choice map arm never copies the value, so a
    served "auto"/"none" is silently DROPPED from the wire (every other
    value falls back at the gate; v1 raises)."""
    flatten_text_content_lists: bool = False
    """sambanova/moonshot: v1 flattens text-only content lists to a single
    concatenated string (handle_messages_with_content_list_to_str_conversion);
    moonshot skips the flatten REQUEST-WIDE when any non-text part is present
    (sambanova's gate falls back on those, so the same skip is inert there)."""
    rewrite: _RewriteFn | None = None
    """Provider VALUE rewrites (pure functions of body/request/deps): the
    deepinfra zero-temperature floor and the moonshot temperature laws."""


def serialize_with_profile(
    request: ChatRequest, deps: TranslationDeps, profile: CompatProfile
) -> _SerializeResult:
    reason = profile.unsupported(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(
        lambda body: _with_deltas(body, request, deps, profile)
    )


def _with_deltas(
    body: Body, request: ChatRequest, deps: TranslationDeps, profile: CompatProfile
) -> Body:
    if profile.rename_max_completion_tokens and "max_completion_tokens" in body:
        body = {
            **{k: v for k, v in body.items() if k != "max_completion_tokens"},
            "max_tokens": body["max_completion_tokens"],
        }
    if profile.drop_text_response_format and body.get("response_format") == {
        "type": "text"
    }:
        body = {k: v for k, v in body.items() if k != "response_format"}
    if profile.drop_non_json_schema_response_format:
        response_format = body.get("response_format")
        if (
            isinstance(response_format, dict)
            and response_format.get("type") != "json_schema"
        ):
            body = {k: v for k, v in body.items() if k != "response_format"}
    if profile.drop_tool_choice:
        body = {k: v for k, v in body.items() if k != "tool_choice"}
    if profile.flatten_text_content_lists:
        body = _flatten_text_content_lists(body)
    if profile.rewrite is not None:
        body = profile.rewrite(body, request, deps)
    emittable = p.ALLOWED[profile.provider]
    extras: dict[str, PlainJson] = {}
    user = request.user.default_value(None) if "user" in emittable else None
    if user is not None:
        extras = {**extras, "user": user}
    effort = (
        request.reasoning_effort.default_value(None)
        if "reasoning_effort" in emittable
        else None
    )
    if effort is not None:
        extras = {**extras, "reasoning_effort": effort}
    return {**body, **extras} if extras else body


def _base_list_gate(provider: str) -> _GateFn:
    def gate(request: ChatRequest, deps: TranslationDeps) -> str | None:
        return base_list_unsupported(request, deps, provider)

    return gate


def _json_registry_gate(provider: str) -> _GateFn:
    def gate(request: ChatRequest, deps: TranslationDeps) -> str | None:
        return json_registry_unsupported(request, deps, provider)

    return gate


def _flatten_text_content_lists(body: Body) -> Body:
    """v1's handle_messages_with_content_list_to_str_conversion, body-level:
    a content list becomes the concatenation of its text values — but only
    when the joined text is non-empty (v1's ``if texts:`` keeps all-empty
    lists as lists), and only when NO message in the request carries a
    non-text part (moonshot's request-wide multimodal skip; sambanova's gate
    already fell back on those shapes, so the skip never fires there).
    All verified in-process at HEAD ("ab" join, lossless skip, empty keep)."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    parts = [part for message in messages for part in _content_list_parts(message)]
    if any(not isinstance(part, dict) or part.get("type") != "text" for part in parts):
        return body  # the request-wide multimodal skip
    return {**body, "messages": [_flattened_message(message) for message in messages]}


def _content_list_parts(message: PlainJson) -> list[PlainJson]:
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def _flattened_message(message: PlainJson) -> PlainJson:
    parts = _content_list_parts(message)
    if not parts or not isinstance(message, dict):
        return message
    texts = "".join(
        text
        for part in parts
        if isinstance(part, dict) and isinstance(text := part.get("text"), str) and text
    )
    return {**message, "content": texts} if texts else message


def _deepinfra_rewrite(body: Body, request: ChatRequest, deps: TranslationDeps) -> Body:
    """The one-model zero-temperature floor (deepinfra map_openai_params:
    temperature == 0 on exactly mistralai/Mistral-7B-Instruct-v0.1 is bumped
    to MIN_NON_ZERO_TEMPERATURE; verified 0 -> 0.0001 at HEAD)."""
    if (
        request.model == "mistralai/Mistral-7B-Instruct-v0.1"
        and body.get("temperature") == 0
    ):
        return {**body, "temperature": MIN_NON_ZERO_TEMPERATURE}
    return body


def _moonshot_rewrite(body: Body, request: ChatRequest, deps: TranslationDeps) -> Body:
    """Moonshot's temperature laws (map_openai_params:135-150), verified at
    HEAD: reasoning models (model-map supports_reasoning over moonshot/{m})
    get temperature POPPED entirely; otherwise temperature > 1 is clamped to
    the literal int 1 (1.5 -> 1, 2 -> 1; 1.0 stays 1.0). v1's third arm
    (< 0.3 with n > 1 -> 0.3) is unreachable here: n is outside the IR and
    falls back at the inbound boundary."""
    if p.supports_moonshot_reasoning(request.model, deps):
        return {k: v for k, v in body.items() if k != "temperature"}
    temperature = body.get("temperature")
    if isinstance(temperature, (int, float)) and temperature > 1:
        return {**body, "temperature": 1}
    return body


_PROFILE_ROWS: tuple[CompatProfile, ...] = (
    CompatProfile(
        provider="together_ai",
        unsupported=p.together_ai_unsupported,
        drop_text_response_format=True,
    ),
    CompatProfile(
        provider="cerebras",
        unsupported=p.cerebras_unsupported,
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="nvidia_nim",
        unsupported=p.nvidia_nim_unsupported,
        rename_max_completion_tokens=True,
    ),
    CompatProfile(provider="lm_studio", unsupported=_base_list_gate("lm_studio")),
    CompatProfile(provider="llamafile", unsupported=_base_list_gate("llamafile")),
    CompatProfile(
        provider="lambda_ai",
        unsupported=_base_list_gate("lambda_ai"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="nebius",
        unsupported=_base_list_gate("nebius"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(provider="novita", unsupported=_base_list_gate("novita")),
    CompatProfile(
        provider="wandb",
        unsupported=_base_list_gate("wandb"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="featherless_ai",
        unsupported=p.featherless_ai_unsupported,
        rename_max_completion_tokens=True,
    ),
    CompatProfile(provider="nscale", unsupported=p.nscale_unsupported),
    CompatProfile(provider="hyperbolic", unsupported=p.hyperbolic_unsupported),
    CompatProfile(
        provider="volcengine",
        unsupported=p.volcengine_unsupported,
        rename_max_completion_tokens=True,
    ),
    # wave-1b SDK-path shims. mct flags re-verified in-process at HEAD:
    # rename = the OpenAILike super arm (ai21_chat/empower/friendliai/
    # galadriel/github/inception) or docker_model_runner's explicit map arm;
    # verbatim = plain OpenAIGPT copy (dashscope/meta_llama/
    # vercel_ai_gateway); morph/v0/zai raise (mct outside their lists, so the
    # OpenAILike rename arm is dead behind the gate — the nscale/hyperbolic
    # trap again).
    CompatProfile(
        provider="ai21_chat",
        unsupported=p.ai21_chat_unsupported,
        rename_max_completion_tokens=True,
    ),
    CompatProfile(provider="dashscope", unsupported=_base_list_gate("dashscope")),
    CompatProfile(
        provider="docker_model_runner",
        unsupported=_base_list_gate("docker_model_runner"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="empower",
        unsupported=_base_list_gate("empower"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="friendliai",
        unsupported=_base_list_gate("friendliai"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="galadriel",
        unsupported=_base_list_gate("galadriel"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="github",
        unsupported=_base_list_gate("github"),
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="inception",
        unsupported=p.inception_unsupported,
        rename_max_completion_tokens=True,
    ),
    CompatProfile(
        provider="meta_llama",
        unsupported=_base_list_gate("meta_llama"),
        drop_non_json_schema_response_format=True,
    ),
    CompatProfile(provider="morph", unsupported=p.morph_unsupported),
    CompatProfile(provider="v0", unsupported=p.v0_unsupported),
    CompatProfile(provider="zai", unsupported=p.zai_unsupported),
    CompatProfile(
        provider="vercel_ai_gateway",
        unsupported=_base_list_gate("vercel_ai_gateway"),
        # The dedicated main.py elif is DEAD CODE (the compat list matches at
        # the big SDK elif first); the map's always-injected ``extra_body`` is
        # empty unless the non-IR ``providerOptions`` rides (inbound fallback)
        # and the SDK spreads the empty dict to nothing on the wire.
    ),
    # wave-1b JSON-registry rows: one parameterized gate (the dynamic
    # JSONProviderConfig's function-calling capability fork) + the
    # param_mappings mct→max_tokens rename where providers.json carries it.
    *(
        CompatProfile(
            provider=provider,
            unsupported=_json_registry_gate(provider),
            rename_max_completion_tokens=provider in JSON_RENAME,
        )
        for provider in p.JSON_REGISTRY_PROVIDERS
    ),
    # wave-2a rows (researcher-4 Part 1, entries 1-5; all v1 facts re-verified
    # in-process at HEAD — see test_differential_compat_sdk_request.py).
    CompatProfile(provider="perplexity", unsupported=p.perplexity_unsupported),
    CompatProfile(
        provider="sambanova",
        unsupported=p.sambanova_unsupported,
        rename_max_completion_tokens=True,
        flatten_text_content_lists=True,
    ),
    CompatProfile(
        provider="deepinfra",
        unsupported=p.deepinfra_unsupported,
        rename_max_completion_tokens=True,
        drop_tool_choice=True,
        rewrite=_deepinfra_rewrite,
    ),
    CompatProfile(
        provider="moonshot",
        unsupported=p.moonshot_unsupported,
        rename_max_completion_tokens=True,
        flatten_text_content_lists=True,
        rewrite=_moonshot_rewrite,
    ),
)

PROFILES: Mapping[p.CompatSdkProvider, CompatProfile] = MappingProxyType(
    {profile.provider: profile for profile in _PROFILE_ROWS}
)


def _profile_serializer(profile: CompatProfile) -> Serializer:
    def serialize(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
        return serialize_with_profile(request, deps, profile)

    return serialize


SERIALIZERS: Mapping[p.CompatSdkProvider, Serializer] = MappingProxyType(
    {provider: _profile_serializer(profile) for provider, profile in PROFILES.items()}
)
