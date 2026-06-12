"""The flag-gated completion() forks for the google routes (the I/O half
of the google seam: vertex OAuth token + URL resolution and the HTTP send).
Split from translation_seam_google.py, which owns the pure deps/adapters."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import litellm

from litellm.translation import TranslationDeps
from litellm.translation.ir import Body
from litellm.translation_seam import HttpxJsonPort, _raw_openai_body, enabled_providers
from litellm.translation_seam_google import (
    GoogleProviderKey,
    build_google_deps,
    to_model_response_google,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
    from litellm.types.utils import ModelResponse

    from litellm.translation.engine.pipeline import PreparedRequest

_CompletionForkResult = Union[
    "ModelResponse", Coroutine[Any, Any, "ModelResponse"], None
]

# ---------------------------------------------------------------------------
# completion() forks (the bedrock fork's shape: return None to stay on v1)
# ---------------------------------------------------------------------------


def _ambient_blocks_v2() -> bool:
    """Globals the google serializers cannot honor purely: stay on v1."""
    return (
        litellm.modify_params is True
        or litellm.vertex_ai_safety_settings is not None
        or bool(litellm.custom_prompt_dict)
    )


# the shared raw-body builder carries the fixed semantics this module
# pioneered (every caller-set param rides in; unknowns fall back typed)
_raw_openai_body_google = _raw_openai_body

# the anthropic/bedrock forks' fallback timeout (httpx default ceiling v1
# also applies when no request timeout is set)
_DEFAULT_TIMEOUT_SECONDS = 600.0


def _prepare(
    provider_key: GoogleProviderKey, raw_body: Dict[str, Any], deps: TranslationDeps
) -> Optional["PreparedRequest"]:
    from litellm.translation.engine.pipeline import prepare_chat_request

    prepared_result = prepare_chat_request(raw_body, provider_key, deps)
    if prepared_result.is_error():
        litellm.verbose_logger.debug(
            "translation v2 fallback to v1: %s", prepared_result.error.summary
        )
        return None
    return prepared_result.ok


def _route_to_v2(provider_key: GoogleProviderKey) -> bool:
    """``dispatch.route`` is the single v1/v2 fork (CLAUDE.md): the REAL
    allowlist goes in and route() owns the membership decision too, so no
    half of the fork lives inline (critic-google M2, critic-integration N6)."""
    from typing import cast as _cast

    from litellm.translation import route
    from litellm.translation.dispatch import Provider

    decision = route(
        schema="openai_chat",
        provider=provider_key,
        enabled_providers=_cast("frozenset[Provider]", enabled_providers()),
        body_touching=False,
    )
    return decision.tag == "v2"


def try_completion_v2_vertex(
    *,
    model_route: "VertexAIModelRoute",
    model: str,
    messages: List["AllMessageValues"],
    optional_param_args: Dict[str, Any],
    non_default_params: Dict[str, Any],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional["VERTEX_CREDENTIALS_TYPES"],
    api_base: Optional[str],
    timeout: Optional[float],
    stream: Optional[bool],
    acompletion: Optional[bool],
    logging_obj: "LiteLLMLoggingObj",
    model_response: "ModelResponse",
    request_drop_params: Optional[bool],
) -> _CompletionForkResult:
    """The completion() fork for the vertex_ai branch (gemini + claude
    routes). The route comes from v1's own ``get_vertex_ai_model_route``
    (computed by the caller); everything else routes back to v1."""
    from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    if stream is True or _ambient_blocks_v2():
        return None
    provider_key: GoogleProviderKey
    if model_route == VertexAIModelRoute.GEMINI:
        provider_key = "vertex_ai"
    elif model_route == VertexAIModelRoute.PARTNER_MODELS and (
        VertexAIAnthropicConfig.is_supported_model(
            model=model, custom_llm_provider="vertex_ai"
        )
    ):
        provider_key = "vertex_anthropic"
    else:
        return None
    if not _route_to_v2(provider_key):
        return None
    raw_body = _raw_openai_body_google(
        model, messages, optional_param_args, non_default_params
    )
    deps = build_google_deps(provider_key, request_drop_params=request_drop_params)
    prepared = _prepare(provider_key, raw_body, deps)
    if prepared is None:
        return None
    coroutine = _send_v2_google(
        prepared=prepared,
        provider_key=provider_key,
        deps=deps,
        model=model,
        messages=messages,
        gemini_api_key=None,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_credentials=vertex_credentials,
        api_base=api_base,
        timeout=timeout,
        logging_obj=logging_obj,
        model_response=model_response,
    )
    if acompletion is True:
        return coroutine
    import asyncio

    return asyncio.run(coroutine)


def try_completion_v2_gemini(
    *,
    model: str,
    messages: List["AllMessageValues"],
    optional_param_args: Dict[str, Any],
    non_default_params: Dict[str, Any],
    gemini_api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[float],
    stream: Optional[bool],
    acompletion: Optional[bool],
    logging_obj: "LiteLLMLoggingObj",
    model_response: "ModelResponse",
    request_drop_params: Optional[bool],
) -> _CompletionForkResult:
    """The completion() fork for the Google AI Studio branch."""
    if stream is True or _ambient_blocks_v2():
        return None
    if not _route_to_v2("gemini"):
        return None
    raw_body = _raw_openai_body_google(
        model, messages, optional_param_args, non_default_params
    )
    deps = build_google_deps("gemini", request_drop_params=request_drop_params)
    prepared = _prepare("gemini", raw_body, deps)
    if prepared is None:
        return None
    coroutine = _send_v2_google(
        prepared=prepared,
        provider_key="gemini",
        deps=deps,
        model=model,
        messages=messages,
        gemini_api_key=gemini_api_key,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        api_base=api_base,
        timeout=timeout,
        logging_obj=logging_obj,
        model_response=model_response,
    )
    if acompletion is True:
        return coroutine
    import asyncio

    return asyncio.run(coroutine)


async def _google_endpoint(
    provider_key: GoogleProviderKey,
    model: str,
    gemini_api_key: Optional[str],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional["VERTEX_CREDENTIALS_TYPES"],
    api_base: Optional[str],
    messages: List["AllMessageValues"],
    wire: Body,
) -> Tuple[str, Dict[str, str]]:
    """Resolve token + URL + headers through v1's own envelope helpers (the
    vertex OAuth flow is I/O and stays here, injected as values). Async:
    the GCP credential refresh is network I/O and runs inside the send
    coroutine, so it must use v1's async variant or it would block the
    caller's event loop on the acompletion path (critic-google M1)."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
        VertexLLM,
    )

    if provider_key == "vertex_anthropic":
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.llms.vertex_ai.vertex_ai_partner_models.main import (
            VertexAIPartnerModels,
        )
        from litellm.types.llms.vertex_ai import VertexPartnerProvider

        vertex_llm = VertexLLM()
        access_token, project = await vertex_llm._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        url = VertexAIPartnerModels().get_complete_vertex_url(
            custom_api_base=api_base,
            vertex_location=vertex_location,
            vertex_project=None,
            project_id=project,
            partner=VertexPartnerProvider.claude,
            stream=False,
            model=model,
        )
        headers = AnthropicConfig().validate_environment(
            api_key=access_token,
            headers={"Authorization": f"Bearer {access_token}"},
            model=model,
            messages=messages,
            optional_params={**dict(wire), "is_vertex_request": True},
            litellm_params={},
        )
        return url, headers

    custom_llm_provider: Literal["vertex_ai", "gemini"] = (
        "vertex_ai" if provider_key == "vertex_ai" else "gemini"
    )
    vertex_llm = VertexLLM()
    _auth_header, project = await vertex_llm._ensure_access_token_async(
        credentials=vertex_credentials,
        project_id=vertex_project,
        custom_llm_provider=custom_llm_provider,
    )
    auth_header, url = vertex_llm._get_token_and_url(
        model=model,
        gemini_api_key=gemini_api_key,
        auth_header=_auth_header,
        vertex_project=project or None,
        vertex_location=vertex_location,
        vertex_credentials=vertex_credentials,
        stream=None,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        should_use_v1beta1_features=False,
    )
    headers = VertexGeminiConfig().validate_environment(
        api_key=auth_header,
        headers=None,
        model=model,
        messages=messages,
        optional_params=dict(wire),
        litellm_params={},
    )
    return url, headers


async def _send_v2_google(
    *,
    prepared: "PreparedRequest",
    provider_key: GoogleProviderKey,
    deps: TranslationDeps,
    model: str,
    messages: List["AllMessageValues"],
    gemini_api_key: Optional[str],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional["VERTEX_CREDENTIALS_TYPES"],
    api_base: Optional[str],
    timeout: Optional[float],
    logging_obj: "LiteLLMLoggingObj",
    model_response: "ModelResponse",
) -> "ModelResponse":
    import json

    from litellm.llms.vertex_ai.common_utils import VertexAIError

    from litellm.translation.engine.http import Endpoint
    from litellm.translation.engine.pipeline import (
        response_dialect,
        send_prepared,
        wire_body,
    )
    from litellm.translation_seam import to_model_response

    wire = wire_body(prepared, provider_key)
    url, headers = await _google_endpoint(
        provider_key,
        model,
        gemini_api_key,
        vertex_project,
        vertex_location,
        vertex_credentials,
        api_base,
        messages,
        wire,
    )
    endpoint = Endpoint(
        url=url,
        headers=headers,
        timeout_seconds=float(timeout) if timeout else _DEFAULT_TIMEOUT_SECONDS,
    )
    logging_obj.pre_call(
        input=messages,
        api_key="",
        additional_args={
            "complete_input_dict": wire,
            "api_base": url,
            "headers": dict(headers),
        },
    )
    # the engine's injected-HttpPort send (one skeleton shared with the
    # anthropic fork), not a third hand-rolled httpx client (critic-google M2)
    result = await send_prepared(
        prepared, provider_key, deps, HttpxJsonPort(), endpoint
    )
    if result.is_error():
        error = result.error
        if error.tag == "provider_http":
            logging_obj.post_call(
                input=messages,
                api_key="",
                original_response=error.provider_http.text,
                additional_args={"complete_input_dict": wire},
            )
            raise VertexAIError(
                status_code=error.provider_http.status_code,
                message=error.provider_http.text,
                headers=dict(error.provider_http.headers),
            )
        raise VertexAIError(status_code=500, message=error.summary)
    body = result.ok
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=json.dumps(body, default=str),
        additional_args={"complete_input_dict": wire},
    )
    dialect = response_dialect(provider_key)
    if dialect == "gemini":
        return to_model_response_google(body, model_response)
    return to_model_response(body, model_response, usage_style=dialect)
