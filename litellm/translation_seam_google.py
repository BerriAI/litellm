"""Google-route adapters for translation v2 (vertex gemini, AI Studio gemini,
vertex claude). Lives OUTSIDE litellm/translation like translation_seam.py:
ambient litellm state (model-map capability lookups keyed per provider,
vertex OAuth tokens, uuid/time) enters here as values; the translation
package stays pure. Route decisions call v1's own helpers
(``get_vertex_ai_model_route``) — never re-derived string matching.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, cast

import litellm
from litellm.llms.anthropic.common_utils import AnthropicModelInfo

from litellm.translation import TranslationDeps
from litellm.translation.ir import Body
from litellm.translation_seam import enabled_providers

GOOGLE_PROVIDER_KEYS = ("vertex_ai", "gemini", "vertex_anthropic")

_THOUGHT_SIGNATURE_SEPARATOR = "__thought__"

_VERTEX_RESPONSE_METADATA_FIELDS = (
    "vertex_ai_grounding_metadata",
    "vertex_ai_url_context_metadata",
    "vertex_ai_safety_results",
    "vertex_ai_citation_metadata",
)


def _supports_google(model: str, key: str, provider: str) -> bool:
    if key == "supports_response_schema":
        from litellm.utils import supports_response_schema

        return supports_response_schema(model, provider)
    from litellm.utils import _supports_factory

    try:
        return _supports_factory(model=model, custom_llm_provider=provider, key=key)
    except Exception:
        return False


def _flag_google(model: str, key: str, provider: str) -> Optional[bool]:
    candidates = (model, f"{provider}/{model}")
    for candidate in candidates:
        value = litellm.model_cost.get(candidate, {}).get(key)
        if isinstance(value, bool):
            return value
    return None


def _vertex_claude_candidates(model: str) -> tuple:
    return (model, f"vertex_ai/{model}")


def _supports_vertex_claude(model: str, key: str) -> bool:
    return any(
        AnthropicModelInfo._supports_model_capability(candidate, key)
        for candidate in _vertex_claude_candidates(model)
    )


def _flag_vertex_claude(model: str, key: str) -> Optional[bool]:
    for candidate in _vertex_claude_candidates(model):
        value = AnthropicModelInfo._get_model_capability(candidate, key)
        if value is not None:
            return value
    return None


def _max_tokens_vertex_claude(model: str) -> Optional[int]:
    for candidate in _vertex_claude_candidates(model):
        try:
            value = litellm.utils.get_max_tokens(candidate)
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def _count_response_tokens(text: str) -> int:
    from litellm.utils import token_counter

    return token_counter(text=text, count_response_tokens=True)


def build_google_deps(
    provider_key: str, request_drop_params: Optional[bool] = None
) -> TranslationDeps:
    """Capability lookups resolve against the PROVIDER's model-map rows (the
    dossier's drift item 5: supports_reasoning can disagree between the
    vertex and gemini rows of the same model)."""
    drop_params_global = litellm.drop_params is True
    if provider_key == "vertex_anthropic":
        supports = _supports_vertex_claude
        flag = _flag_vertex_claude
        max_tokens = _max_tokens_vertex_claude
    else:

        def supports(model: str, key: str) -> bool:
            return _supports_google(model, key, provider_key)

        def flag(model: str, key: str) -> Optional[bool]:
            return _flag_google(model, key, provider_key)

        def max_tokens(model: str) -> Optional[int]:
            try:
                return litellm.utils.get_max_tokens(model)
            except Exception:
                return None

    return TranslationDeps(
        max_tokens_for_model=max_tokens,
        supports_capability=supports,
        capability_flag=flag,
        count_response_tokens=_count_response_tokens,
        drop_params=drop_params_global or request_drop_params is True,
        drop_params_global=drop_params_global,
        modify_params=litellm.modify_params is True,
    )


def _mint_tool_call_id(raw_id: object) -> object:
    """v1 mints ``call_<uuid4.hex[:28]>`` per functionCall without a native
    id; the IR carries an empty prefix (optionally followed by the
    thought-signature suffix) as the sentinel."""
    if not isinstance(raw_id, str):
        return raw_id
    if raw_id == "" or raw_id.startswith(_THOUGHT_SIGNATURE_SEPARATOR):
        import uuid

        return f"call_{uuid.uuid4().hex[:28]}{raw_id}"
    return raw_id


def _minted_message(message: Dict[str, Any]) -> Dict[str, Any]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    minted = [
        (
            {**entry, "id": _mint_tool_call_id(entry.get("id"))}
            if isinstance(entry, dict)
            else entry
        )
        for entry in tool_calls
    ]
    return {**message, "tool_calls": minted}


def _build_usage_gemini(payload: Dict[str, Any]):
    """Construct ``Usage`` with v1 ``_calculate_usage``'s exact kwarg set:
    a five-field PromptTokensDetailsWrapper and a CompletionTokensDetails
    wrapper whose fields are only assigned when the wire reported them."""
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )

    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_payload = payload.get("completion_tokens_details")
    completion_details = None
    if isinstance(completion_payload, dict) and completion_payload:
        completion_details = CompletionTokensDetailsWrapper()
        for key, value in completion_payload.items():
            setattr(completion_details, key, value)
    return Usage(
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=prompt_details.get("cached_tokens"),
            audio_tokens=prompt_details.get("audio_tokens"),
            text_tokens=prompt_details.get("text_tokens"),
            image_tokens=prompt_details.get("image_tokens"),
            video_tokens=prompt_details.get("video_tokens"),
        ),
        cache_read_input_tokens=payload.get("cache_read_input_tokens"),
        reasoning_tokens=payload.get("reasoning_tokens"),
        completion_tokens_details=completion_details,
    )


def to_model_response_google(body: Body, model_response=None):
    """Adapt a v2 gemini-dialect response body onto ModelResponse the way
    v1's ``_transform_google_generate_content_to_openai_model_response``
    assembles it (fresh Choices list, vertex metadata attrs, responseId)."""
    import time

    from litellm.types.utils import Choices, Message, ModelResponse

    response = model_response if model_response is not None else ModelResponse()
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    message_payload = first.get("message") if isinstance(first, dict) else {}
    finish = first.get("finish_reason") if isinstance(first, dict) else None
    message = Message(
        **cast(Dict[str, Any], _minted_message(cast(Dict[str, Any], message_payload)))
    )
    response.choices = [
        Choices(
            finish_reason=finish if isinstance(finish, str) else "stop",
            index=0,
            message=message,
            logprobs=None,
            enhancements=None,
        )
    ]
    usage_payload = body.get("usage")
    if isinstance(usage_payload, dict):
        setattr(response, "usage", _build_usage_gemini(usage_payload))
    response.created = int(time.time())
    model = body.get("model")
    if isinstance(model, str):
        response.model = model
    response_id = body.get("id")
    if isinstance(response_id, str) and response_id:
        response.id = response_id
    for field in _VERTEX_RESPONSE_METADATA_FIELDS:
        setattr(response, field, [])
        response._hidden_params[field] = []
    return response


def to_model_response_stream_google(body: Body):
    """One v2 gemini chunk body -> ModelResponseStream, mirroring the two
    construction sites in v1 (the iterator's content chunks and the
    wrapper-synthesized finish chunk)."""
    from litellm.types.utils import (
        Delta,
        ModelResponseStream,
        StreamingChoices,
    )

    choices_payload = cast(List[Dict[str, Any]], body.get("choices") or [{}])
    first = choices_payload[0]
    delta_payload = cast(Dict[str, Any], first.get("delta") or {})
    finish = first.get("finish_reason")
    if finish is not None:
        chunk = ModelResponseStream(
            id=cast(Optional[str], body.get("id")),
            model=cast(Optional[str], body.get("model")),
            choices=[
                StreamingChoices(
                    finish_reason=finish,
                    index=0,
                    delta=Delta(),
                    logprobs=None,
                    enhancements=None,
                )
            ],
        )
        return chunk
    tool_calls = delta_payload.get("tool_calls")
    if isinstance(tool_calls, list):
        tool_calls = [
            (
                {**entry, "id": _mint_tool_call_id(entry.get("id"))}
                if isinstance(entry, dict)
                else entry
            )
            for entry in tool_calls
        ]
    delta = Delta(
        content=delta_payload.get("content"),
        reasoning_content=delta_payload.get("reasoning_content"),
        tool_calls=tool_calls,
        images=None,
        function_call=None,
        annotations=None,
        provider_specific_fields=delta_payload.get("provider_specific_fields"),
        role=delta_payload.get("role"),
    )
    chunk = ModelResponseStream(
        id=cast(Optional[str], body.get("id")),
        model=cast(Optional[str], body.get("model")),
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=delta,
                logprobs=None,
                enhancements=None,
            )
        ],
        system_fingerprint=None,
    )
    setattr(chunk, "citations", None)
    for field in (
        "vertex_ai_grounding_metadata",
        "vertex_ai_url_context_metadata",
        "vertex_ai_safety_ratings",
        "vertex_ai_safety_results",
        "vertex_ai_citation_metadata",
    ):
        setattr(chunk, field, body.get(field, []))
    return chunk


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
