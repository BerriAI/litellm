"""Google-route adapters for translation v2 (vertex gemini, AI Studio gemini,
vertex claude). Lives OUTSIDE litellm/translation like translation_seam.py:
ambient litellm state (model-map capability lookups keyed per provider,
vertex OAuth tokens, uuid/time) enters here as values; the translation
package stays pure. Route decisions call v1's own helpers
(``get_vertex_ai_model_route``) — never re-derived string matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR as _THOUGHT_SIGNATURE_SEPARATOR,
)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo

from litellm.translation import TranslationDeps
from litellm.translation.ir import Body

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

GoogleProviderKey = Literal["vertex_ai", "gemini", "vertex_anthropic"]

_VERTEX_RESPONSE_METADATA_FIELDS = (
    "vertex_ai_grounding_metadata",
    "vertex_ai_url_context_metadata",
    "vertex_ai_safety_results",
    "vertex_ai_citation_metadata",
)


def _supports_google(model: str, key: str, provider: str) -> bool:
    if key == "supports_response_schema":
        from litellm.utils import supports_response_schema

        # v1's own helper: documented "Does not raise error. Defaults to
        # 'False'", so the structured-output fork takes the SAME branch as
        # v1 on a lookup failure.
        return supports_response_schema(model, provider)
    from litellm.utils import _supports_factory

    # No except->False here: _supports_factory raises on unmapped models,
    # and swallowing that into capability-False would silently change the
    # wire body instead of failing loudly (critic-google M6). No google
    # serializer consults other keys today.
    return _supports_factory(model=model, custom_llm_provider=provider, key=key)


def _flag_google(model: str, key: str, provider: str) -> Optional[bool]:
    candidates = (model, f"{provider}/{model}")
    for candidate in candidates:
        value = litellm.model_cost.get(candidate, {}).get(key)
        if isinstance(value, bool):
            return value
    return None


def _vertex_claude_candidates(model: str) -> tuple[str, str]:
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
    provider_key: GoogleProviderKey, request_drop_params: Optional[bool] = None
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


def _build_usage_gemini(payload: Dict[str, Any]) -> "Usage":
    """Construct ``Usage`` with v1 ``_calculate_usage``'s exact kwarg set:
    a five-field PromptTokensDetailsWrapper and a CompletionTokensDetails
    wrapper whose fields are only assigned when the wire reported them
    (Usage serialization dumps explicitly-set fields only, so the wrapper
    is built from the reported keys, bounded to its declared fields)."""
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )

    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_payload = payload.get("completion_tokens_details")
    completion_details = None
    if isinstance(completion_payload, dict) and completion_payload:
        declared = CompletionTokensDetailsWrapper.model_fields
        completion_details = CompletionTokensDetailsWrapper(
            **{
                key: value
                for key, value in completion_payload.items()
                if key in declared
            }
        )
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


def to_model_response_google(
    body: Body, model_response: Optional["ModelResponse"] = None
) -> "ModelResponse":
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


def to_model_response_stream_google(body: Body) -> "ModelResponseStream":
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
