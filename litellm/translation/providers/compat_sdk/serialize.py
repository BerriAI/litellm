"""Serializers for the SDK-path openai-compat family (wave 1a).

Every provider here rides v1's big openai elif (main.py:2646-2667) into the
OpenAI SDK: ``get_optional_params`` runs the provider config's param gates,
``provider_config.transform_request`` (openai.py:727) runs the inherited
base five-touch assembly, and none of the family overrides it. The v2 body
is therefore ``openai_compat.assemble_body`` after the provider's gates,
plus at most three mechanical deltas captured per provider in a frozen
``CompatProfile``:

- ``rename_max_completion_tokens``: the configs whose map renames mct ->
  max_tokens (cerebras/nvidia_nim/nebius/wandb/featherless_ai and the
  OpenAILike-based lambda_ai/volcengine). The IR already collapses mct into
  ``max_tokens``, so the rename is emitting that collapsed key.
- ``emit_user``: providers whose own supported list carries ``user``
  unconditionally (cerebras, hyperbolic) — a typed fallback everywhere else.
- ``drop_text_response_format``: together_ai's map pops a verbatim
  ``{"type": "text"}`` response_format.
- ``emit_reasoning_effort``: cerebras, gated per model in params.py before
  emission ever happens.

The response side needs NO per-provider code: the live v1 normalizer is the
same ``convert_to_model_response_object`` the openai_compat parser mirrors,
and the ``{provider}/{wire_model}`` model re-prefix (openai.py:676-677 +
cdr:699-710) is the SEAM's ``_to_model_response_openai`` preset arm, pinned
per provider by the differential's preset-model rows. Streams ride the
default openai wrapper arm -> the ``"openai"`` chunk dialect (baseten is the
one would-be member that does NOT, and is dropped from the wave for it).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]
_GateFn = Callable[[ChatRequest, TranslationDeps], str | None]


@dataclass(frozen=True)
class CompatProfile:
    provider: str
    unsupported: _GateFn
    rename_max_completion_tokens: bool = False
    emit_user: bool = False
    drop_text_response_format: bool = False
    emit_reasoning_effort: bool = False


def serialize_with_profile(
    request: ChatRequest, deps: TranslationDeps, profile: CompatProfile
) -> _SerializeResult:
    reason = profile.unsupported(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(lambda body: _with_deltas(body, request, profile))


def _with_deltas(body: Body, request: ChatRequest, profile: CompatProfile) -> Body:
    if profile.rename_max_completion_tokens and "max_completion_tokens" in body:
        collapsed = request.params.max_tokens.default_value(None)
        body = {
            **{k: v for k, v in body.items() if k != "max_completion_tokens"},
            "max_tokens": collapsed,
        }
    if profile.drop_text_response_format and body.get("response_format") == {
        "type": "text"
    }:
        body = {k: v for k, v in body.items() if k != "response_format"}
    extras: dict[str, PlainJson] = {}
    user = request.user.default_value(None) if profile.emit_user else None
    if user is not None:
        extras = {**extras, "user": user}
    effort = (
        request.reasoning_effort.default_value(None)
        if profile.emit_reasoning_effort
        else None
    )
    if effort is not None:
        extras = {**extras, "reasoning_effort": effort}
    return {**body, **extras} if extras else body


def _base_list_gate(provider: str) -> _GateFn:
    def gate(request: ChatRequest, deps: TranslationDeps) -> str | None:
        return p.base_list_unsupported(request, deps, provider)

    return gate


TOGETHER_AI = CompatProfile(
    provider="together_ai",
    unsupported=p.together_ai_unsupported,
    drop_text_response_format=True,
)
CEREBRAS = CompatProfile(
    provider="cerebras",
    unsupported=p.cerebras_unsupported,
    rename_max_completion_tokens=True,
    emit_user=True,
    emit_reasoning_effort=True,
)
NVIDIA_NIM = CompatProfile(
    provider="nvidia_nim",
    unsupported=p.nvidia_nim_unsupported,
    rename_max_completion_tokens=True,
)
LM_STUDIO = CompatProfile(
    provider="lm_studio", unsupported=_base_list_gate("lm_studio")
)
LLAMAFILE = CompatProfile(
    provider="llamafile", unsupported=_base_list_gate("llamafile")
)
LAMBDA_AI = CompatProfile(
    provider="lambda_ai",
    unsupported=_base_list_gate("lambda_ai"),
    rename_max_completion_tokens=True,
)
NEBIUS = CompatProfile(
    provider="nebius",
    unsupported=_base_list_gate("nebius"),
    rename_max_completion_tokens=True,
)
NOVITA = CompatProfile(provider="novita", unsupported=_base_list_gate("novita"))
WANDB = CompatProfile(
    provider="wandb",
    unsupported=_base_list_gate("wandb"),
    rename_max_completion_tokens=True,
)
FEATHERLESS_AI = CompatProfile(
    provider="featherless_ai",
    unsupported=p.featherless_ai_unsupported,
    rename_max_completion_tokens=True,
)
NSCALE = CompatProfile(provider="nscale", unsupported=p.nscale_unsupported)
HYPERBOLIC = CompatProfile(
    provider="hyperbolic", unsupported=p.hyperbolic_unsupported, emit_user=True
)
VOLCENGINE = CompatProfile(
    provider="volcengine",
    unsupported=p.volcengine_unsupported,
    rename_max_completion_tokens=True,
)


def together_ai_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, TOGETHER_AI)


def cerebras_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, CEREBRAS)


def nvidia_nim_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, NVIDIA_NIM)


def lm_studio_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, LM_STUDIO)


def llamafile_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, LLAMAFILE)


def lambda_ai_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, LAMBDA_AI)


def nebius_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, NEBIUS)


def novita_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, NOVITA)


def wandb_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, WANDB)


def featherless_ai_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, FEATHERLESS_AI)


def nscale_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, NSCALE)


def hyperbolic_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, HYPERBOLIC)


def volcengine_serialize_request(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return serialize_with_profile(request, deps, VOLCENGINE)
