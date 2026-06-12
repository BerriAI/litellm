"""The flag-gated completion() forks for the google routes (the I/O half
of the google seam: vertex OAuth token + URL resolution and the HTTP send).
Split from translation_seam_google.py, which owns the pure deps/adapters."""

from __future__ import annotations

from typing import Any, Optional

import litellm

from litellm.translation import TranslationDeps
from litellm.translation.ir import Body
from litellm.translation_seam import enabled_providers
from litellm.translation_seam_google import (
    build_google_deps,
    to_model_response_google,
)

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


# completion() locals that are routing/transport, not body payload.
_NON_BODY_ARGS = frozenset(
    {
        "model",
        "messages",
        "custom_llm_provider",
        "api_version",
        "max_retries",
        "stream_options",
    }
)


def _raw_openai_body_google(
    model: str, messages: list, optional_param_args: dict, non_default_params: dict
) -> dict:
    """EVERY caller-set OpenAI param rides into the parse (not the shared
    seam's _BODY_FIELDS whitelist): anything v2 does not account for (n,
    seed, penalties, modalities, ...) becomes a typed boundary error and
    falls back to v1 instead of being silently dropped."""
    named = {
        key: value
        for key, value in optional_param_args.items()
        if key not in _NON_BODY_ARGS and value is not None
    }
    return {"model": model, "messages": messages, **named, **non_default_params}


def _prepare(provider_key: str, raw_body: dict, deps: TranslationDeps):
    from litellm.translation.engine.pipeline import prepare_chat_request

    prepared_result = prepare_chat_request(raw_body, provider_key, deps)  # type: ignore[arg-type]
    if prepared_result.is_error():
        litellm.verbose_logger.debug(
            "translation v2 fallback to v1: %s", prepared_result.error.summary
        )
        return None
    return prepared_result.ok


def try_completion_v2_vertex(
    *,
    model_route: Any,
    model: str,
    messages: list,
    optional_param_args: dict,
    non_default_params: dict,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional[Any],
    api_base: Optional[str],
    timeout: Optional[float],
    stream: Optional[bool],
    acompletion: Optional[bool],
    logging_obj,
    model_response,
    request_drop_params: Optional[bool],
):
    """The completion() fork for the vertex_ai branch (gemini + claude
    routes). The route comes from v1's own ``get_vertex_ai_model_route``
    (computed by the caller); everything else routes back to v1."""
    from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    if stream is True or _ambient_blocks_v2():
        return None
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
    if provider_key not in enabled_providers():
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
    messages: list,
    optional_param_args: dict,
    non_default_params: dict,
    gemini_api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[float],
    stream: Optional[bool],
    acompletion: Optional[bool],
    logging_obj,
    model_response,
    request_drop_params: Optional[bool],
):
    """The completion() fork for the Google AI Studio branch."""
    if stream is True or _ambient_blocks_v2():
        return None
    if "gemini" not in enabled_providers():
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


def _google_endpoint(
    provider_key: str,
    model: str,
    gemini_api_key: Optional[str],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional[Any],
    api_base: Optional[str],
    messages: list,
    wire: Body,
):
    """Resolve token + URL + headers through v1's own envelope helpers (the
    vertex OAuth flow is I/O and stays here, injected as values)."""
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
        access_token, project = vertex_llm._ensure_access_token(
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

    custom_llm_provider = "vertex_ai" if provider_key == "vertex_ai" else "gemini"
    vertex_llm = VertexLLM()
    _auth_header, project = vertex_llm._ensure_access_token(
        credentials=vertex_credentials,
        project_id=vertex_project,
        custom_llm_provider=custom_llm_provider,  # type: ignore[arg-type]
    )
    auth_header, url = vertex_llm._get_token_and_url(
        model=model,
        gemini_api_key=gemini_api_key,
        auth_header=_auth_header,
        vertex_project=project or None,
        vertex_location=vertex_location,
        vertex_credentials=vertex_credentials,
        stream=None,
        custom_llm_provider=custom_llm_provider,  # type: ignore[arg-type]
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
    prepared,
    provider_key: str,
    deps: TranslationDeps,
    model: str,
    messages: list,
    gemini_api_key: Optional[str],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional[Any],
    api_base: Optional[str],
    timeout: Optional[float],
    logging_obj,
    model_response,
):
    import httpx

    from litellm.llms.vertex_ai.common_utils import VertexAIError

    from litellm.translation.engine.pipeline import (
        response_dialect,
        translate_chat_response,
        wire_body,
    )
    from litellm.translation_seam import to_model_response

    wire = wire_body(prepared, provider_key)  # type: ignore[arg-type]
    url, headers = _google_endpoint(
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
    logging_obj.pre_call(
        input=messages,
        api_key="",
        additional_args={
            "complete_input_dict": wire,
            "api_base": url,
            "headers": dict(headers),
        },
    )
    async with httpx.AsyncClient(
        timeout=float(timeout) if timeout else 600.0
    ) as client:
        raw = await client.post(url, headers=dict(headers), json=dict(wire))
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=raw.text,
        additional_args={"complete_input_dict": wire},
    )
    if raw.status_code < 200 or raw.status_code >= 300:
        raise VertexAIError(
            status_code=raw.status_code, message=raw.text, headers=raw.headers
        )
    try:
        payload = raw.json()
    except ValueError as parse_error:
        raise VertexAIError(
            status_code=422,
            message=f"non-JSON google response: {raw.text[:200]}",
            headers=raw.headers,
        ) from parse_error
    result = translate_chat_response(
        payload, prepared.request, provider_key, deps  # type: ignore[arg-type]
    )
    if result.is_error():
        raise VertexAIError(
            status_code=500, message=result.error.summary, headers=raw.headers
        )
    dialect = response_dialect(provider_key)  # type: ignore[arg-type]
    if dialect == "gemini":
        return to_model_response_google(result.ok, model_response)
    return to_model_response(result.ok, model_response, usage_style=dialect)
