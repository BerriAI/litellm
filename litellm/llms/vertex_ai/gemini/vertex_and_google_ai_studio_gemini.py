# What is this?
## httpx client for vertex ai calls
## Initial implementation - covers gemini + image gen calls
import json
import re
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from functools import lru_cache, partial
from itertools import chain, repeat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Optional, Protocol, Union, cast, get_args

import httpx
from openai.types.responses.response_function_web_search import ResponseFunctionWebSearch
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO,
    PROVIDER_FINISH_REASONS_HIDDEN_PARAM,
)
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.allowed_tools_tool_choice import HostedMcpToolHandling
from litellm.litellm_core_utils.audio_utils.utils import as_raw_pcm16_base64
from litellm.litellm_core_utils.core_helpers import (
    as_object_list,
    as_str_mapping,
    drops_unsupported_params,
    map_finish_reason,
    normalize_drop_params,
)
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.hidden_params import set_hidden_param
from litellm.litellm_core_utils.json_fragment_accumulator import JSONFragmentAccumulator
from litellm.litellm_core_utils.model_param_helper import embeds_thought_signature_in_tool_call_id
from litellm.litellm_core_utils.prompt_templates.factory import encode_tool_call_id_with_signature
from litellm.litellm_core_utils.reasoning_effort_capability import declared_reasoning_efforts_for_model
from litellm.litellm_core_utils.reasoning_effort_utils import warn_dropped_reasoning
from litellm.litellm_core_utils.service_tier_utils import GEMINI_TRAFFIC_TYPE_SERVICE_TIERS
from litellm.litellm_core_utils.tool_fields import (
    declared_function_name,
    qualified_member_name,
    tool_params_without_hosted_mcp,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
    get_httpx_client,
)
from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines, iter_sse_lines
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.base import CachedTokensDetails
from litellm.types.llms.gemini import BidiGenerateContentServerMessage
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
    ChatCompletionAudioDelta,
    ChatCompletionResponseMessage,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParamFunctionChunk,
    ImageURLListItem,
    ImageURLObject,
    OpenAIChatCompletionFinishReason,
)
from litellm.types.llms.vertex_ai import (
    VERTEX_AI_PROVIDER_METADATA_FIELDS,
    VERTEX_CREDENTIALS_TYPES,
    Candidates,
    ContentType,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiFinishReason,
    GeminiImageProviderSpecificFields,
    GeminiImageURLListItem,
    GeminiThinkingConfig,
    GenerateContentResponseBody,
    HttpxPartType,
    LogprobsResult,
    PromptFeedback,
    RequestBody,
    ToolConfig,
    Tools,
    UsageMetadata,
    VertexToolName,
)
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    TopLogprob,
    Usage,
)
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    is_base64_encoded,
    is_explicitly_disabled_factory,
    supports_factory,
    supports_reasoning,
)

from ....utils import remove_additional_properties, remove_strict_from_schema
from ..common_utils import (
    VertexAIError,
    build_json_schema,
    build_vertex_schema,
    enums_fit,
    gemini_media_capability,
    has_bare_root_union,
    has_const,
    parameters_json_schema_for,
    supports_response_json_schema,
    with_enums_that_fit,
    with_pins_that_fit,
)
from ..context_caching.known_cached_contents import known_cached_contents
from ..context_caching.rejected_cache_retry import (
    cache_rebuilds,
    post_with_fresh_cache_async,
    post_with_fresh_cache_sync,
)
from ..request_labels import vertex_request_label_fields, without_caller_labels_header
from ..vertex_llm_base import VertexBase
from .grounding_requests import calculate_grounding_requests, search_tool_call_queries, web_search_queries
from .long_decode import (
    HOP_FAILURES,
    AsyncStreamResume,
    LongDecodeHops,
    StreamResume,
    SyncStreamResume,
    answer_bytes,
    client_continuation_token,
    collect_hops_async,
    collect_hops_sync,
    continuation_cap_of,
    hop_request,
    hops_original_response,
    initial_prompt_tokens_of,
    log_failed_hop,
    merge_hops,
    next_hop_token,
    output_tokens_of,
    parse_body,
    per_hop,
    sum_usage_metadata,
    url_for,
    with_initial_prompt_tokens,
    with_spans_shifted,
)
from .rejected_thought_signature_retry import (
    build_with_accepted_thought_signatures_async,
    build_with_accepted_thought_signatures_sync,
    post_with_accepted_thought_signatures_async,
    post_with_accepted_thought_signatures_sync,
    with_placeholder_thought_signatures,
)
from .transformation import (
    async_transform_request_body,
    gemini_convert_messages_with_history,
    sync_transform_request_body,
)
from .web_search_calls import gemini_web_search_calls

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponseStream, StreamingChoices

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any
    StreamingChoices = Any


SUPPORTED_REASONING_EFFORTS: Final = ("minimal", "low", "medium", "high", "none", "disable")


def _response_header(headers: Mapping[str, str], name: str) -> str | None:
    return headers.get(name)


class _HiddenParamsSink(Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


class _StreamMetadataTarget(Protocol):
    @property
    def hidden_params(self) -> _HiddenParamsSink: ...


# Gemini prebuilt voices per https://ai.google.dev/gemini-api/docs/speech-generation (2026-10-01), by nearest character;
# nova is left out since Gemini 3.x TTS models accept it as given (live check 2026-10-08)
OPENAI_TO_GEMINI_TTS_VOICES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "alloy": "Kore",
        "ash": "Iapetus",
        "ballad": "Algieba",
        "cedar": "Achird",
        "coral": "Sulafat",
        "echo": "Charon",
        "fable": "Umbriel",
        "marin": "Despina",
        "onyx": "Orus",
        "sage": "Vindemiatrix",
        "shimmer": "Achernar",
        "verse": "Puck",
    }
)


SENT_SERVICE_TIERS: Final = frozenset({"priority", "flex", "standard"})
PROVIDER_DEFAULT_SERVICE_TIERS: Final = frozenset({"auto", "default", "unspecified"})
VERTEX_SHARED_REQUEST_TIERS: Final = frozenset({"priority", "flex"})
SERVICE_TIER_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"standard_only": "standard"})
GEMINI_SERVICE_TIER_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {**SERVICE_TIER_ALIASES, "auto": "priority", "default": "standard"}
)


def _response_service_tier(header_tier: str | None, traffic_type: object) -> str | None:
    if header_tier:
        lowered: Final = header_tier.lower()
        return "default" if lowered == "standard" else lowered
    if isinstance(traffic_type, str) and traffic_type:
        return GEMINI_TRAFFIC_TYPE_SERVICE_TIERS.get(traffic_type.upper(), "default")
    return None


def _unsupported_service_tier(service_tier: str) -> UnsupportedParamsError:
    known_tiers: Final = sorted(SENT_SERVICE_TIERS | PROVIDER_DEFAULT_SERVICE_TIERS | frozenset(SERVICE_TIER_ALIASES))
    return UnsupportedParamsError(
        message=(
            f"Invalid `service_tier`: {service_tier!r}. "
            f"Must be one of: {', '.join(repr(tier) for tier in known_tiers)}. "
            "To drop this param, set `litellm.drop_params = True` or pass in `(.., drop_params=True)` "
            "in the request - https://docs.litellm.ai/docs/completion/drop_params"
        ),
        status_code=400,
    )


def _unsupported_reasoning_effort(reasoning_effort: str) -> UnsupportedParamsError:
    return UnsupportedParamsError(
        message=(
            f"Invalid `reasoning_effort`: {reasoning_effort!r}. "
            f"Must be one of: {', '.join(repr(effort) for effort in SUPPORTED_REASONING_EFFORTS)}. "
            "To drop this param, set `litellm.drop_params = True` or pass in `(.., drop_params=True)` "
            "in the request - https://docs.litellm.ai/docs/completion/drop_params"
        ),
        status_code=400,
    )


_SERVER_SIDE_TOOL_INVOCATIONS_KEYS: Final = ("include_server_side_tool_invocations", "includeServerSideToolInvocations")


def _turns_server_side_tools_off(params: Mapping[str, object]) -> bool:
    return any(params.get(key) is False for key in _SERVER_SIDE_TOOL_INVOCATIONS_KEYS)


def _pins_function_calling_mode(params: Mapping[str, object]) -> bool:
    match params.get("tool_choice"):
        case "required" | "none" | {"function": {"name": str()}}:
            return True
        case _:
            return False


_PINNED_FUNCTION_CALLING_MODES: Final = frozenset({"ANY", "NONE"})


def _tool_config_keeps_the_old_search_drop(tool_config: Mapping[str, object]) -> bool:
    calling: Final = as_str_mapping(
        tool_config.get("functionCallingConfig", tool_config.get("function_calling_config"))
    )
    mode: Final = calling.get("mode") if calling is not None else None
    return _turns_server_side_tools_off(tool_config) or (
        isinstance(mode, str) and mode in _PINNED_FUNCTION_CALLING_MODES
    )


_SAMPLING_DEPRECATION_UNMAPPED_MODELS: Final = "models outside litellm's model map"
_NON_INTEGER_BUDGET_REASON: Final = (
    "Gemini takes budget_tokens only as an integer, so it uses its default thinking budget."
)
_LEVEL_ONLY_BUDGET_REASON: Final = (
    "Gemini 3 and later set reasoning by thinking level, not budget, so litellm does not send the budget. "
    "Send reasoning_effort to pick a level."
)
_sampling_deprecation_lock: Final = threading.Lock()


@lru_cache(maxsize=None)
def _log_gemini_sampling_params_deprecated(model: str) -> None:
    verbose_logger.warning(
        "DeprecationWarning: `temperature`, `top_p`, and `top_k` continue to function for Gemini 3+ (%s) but are planned for removal in a future release. Move sampling guidance into the `system` instructions instead.",
        model,
    )


def _sampling_deprecation_notice_key(model: str) -> str:
    catalog: Final = GetModelCostMap.loaded_model_cost_map()
    is_cataloged: Final = any(name in catalog for name in (model, f"vertex_ai/{model}", f"gemini/{model}"))
    return model if is_cataloged else _SAMPLING_DEPRECATION_UNMAPPED_MODELS


def _warn_gemini_sampling_params_deprecated(model: str) -> None:
    with _sampling_deprecation_lock:
        _log_gemini_sampling_params_deprecated(_sampling_deprecation_notice_key(model))


def _cache_creation_tokens(logging_obj: LoggingClass) -> int:
    recorded: Final[object] = getattr(logging_obj, "vertex_ai_cache_creation_input_tokens", 0)
    return recorded if isinstance(recorded, int) and recorded > 0 else 0


def _prompt_block_fields(feedback: PromptFeedback) -> Mapping[str, str]:
    return MappingProxyType({"block_reason": feedback["blockReason"]})


def _names_its_finish(candidate: Candidates) -> bool:
    return (candidate.get("finishReason") or "FINISH_REASON_UNSPECIFIED") != "FINISH_REASON_UNSPECIFIED"


def _content_parts(candidate: Candidates) -> Sequence[HttpxPartType]:
    content: Final = candidate.get("content")
    return content.get("parts", ()) if content is not None else ()


def _generated_image_item(data_uri: str, thought_signature: str | None) -> ImageURLListItem:
    image_url: Final = ImageURLObject(url=data_uri, detail="auto")
    if thought_signature is None:
        return ImageURLListItem(image_url=image_url, index=0, type="image_url")
    return GeminiImageURLListItem(
        image_url=image_url,
        index=0,
        type="image_url",
        provider_specific_fields=GeminiImageProviderSpecificFields(thought_signature=thought_signature),
    )


def _inline_image(part: HttpxPartType) -> ImageURLListItem | None:
    inline_data: Final = part.get("inlineData", {})
    mime_type: Final = inline_data.get("mimeType", "")
    if not mime_type.startswith("image/"):
        return None
    return _generated_image_item(f"data:{mime_type};base64,{inline_data.get('data', '')}", part.get("thoughtSignature"))


def _images_in(parts: Sequence[HttpxPartType]) -> tuple[ImageURLListItem, ...]:
    return tuple(image for image in map(_inline_image, parts) if image is not None)


def _final_images(parts: Sequence[HttpxPartType]) -> tuple[ImageURLListItem, ...]:
    return _images_in(tuple(part for part in parts if part.get("thought") is not True))


def _last_thought_image(parts: Sequence[HttpxPartType]) -> tuple[ImageURLListItem, ...]:
    return _images_in(tuple(part for part in parts if part.get("thought") is True))[-1:]


def _blocks_its_output(candidate: Candidates) -> bool:
    reason: Final = candidate.get("finishReason")
    return reason is not None and map_finish_reason(reason) == "content_filter"


def _fallback_thought_image(candidate: Candidates) -> tuple[ImageURLListItem, ...]:
    return () if _blocks_its_output(candidate) else _last_thought_image(_content_parts(candidate))


def _with_delta_images(choice: "StreamingChoices", images: tuple[ImageURLListItem, ...]) -> "StreamingChoices":
    return choice.model_copy(update={"delta": choice.delta.model_copy(update={"images": list(images)})})


def _candidate_finish_fields(candidate: Candidates) -> Mapping[str, str] | None:
    reason: Final = candidate.get("finishReason")
    return MappingProxyType({"native_finish_reason": reason}) if reason is not None else None


def _candidate_parts(candidate: Candidates) -> Sequence[HttpxPartType]:
    content: Final = candidate.get("content")
    return (content.get("parts") or ()) if content else ()


def _parts_of_candidates(candidates: Sequence[Candidates]) -> Iterator[HttpxPartType]:
    for candidate in candidates:
        yield from _candidate_parts(candidate)


def _candidate_web_search_queries(candidate: Candidates) -> tuple[str, ...]:
    grounding: Final = candidate.get("groundingMetadata")
    grounding_items: Final = as_object_list(grounding) or (() if grounding is None else per_hop(grounding))
    return web_search_queries(
        tuple(filter(None, map(as_str_mapping, grounding_items))), search_tool_call_queries(_candidate_parts(candidate))
    )


_MALFORMED_FINISH_REASONS: Final[frozenset[GeminiFinishReason]] = frozenset(
    {"MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE"}
)
_OUTPUT_FIELDS: Final = ("content", "tool_calls", "function_call", "images", "audio")


_CHOICE_FIELDS: Final = TypeAdapter(dict[str, object])


def _native_finish_reason(choice: object) -> object:
    fields: Final[object] = getattr(choice, "provider_specific_fields", None)
    return _CHOICE_FIELDS.validate_python(fields).get("native_finish_reason") if isinstance(fields, dict) else None


def _is_malformed_without_output(choice: object) -> bool:
    return (
        isinstance(choice, litellm.Choices)
        and _native_finish_reason(choice) in _MALFORMED_FINISH_REASONS
        and not any(getattr(choice.message, field, None) for field in _OUTPUT_FIELDS)
    )


def malformed_without_output_message(response: ModelResponse) -> str | None:
    if not response.choices or not all(_is_malformed_without_output(choice) for choice in response.choices):
        return None
    return f"Gemini finished with {_native_finish_reason(response.choices[0])} and returned no output"


def _malformed_without_output_error(response: ModelResponse, headers: httpx.Headers) -> VertexAIError | None:
    message: Final = malformed_without_output_message(response)
    return None if message is None else VertexAIError(status_code=500, message=message, headers=headers)


def _served_model_name(model_version: object) -> str | None:
    if not isinstance(model_version, str) or not model_version:
        return None
    return model_version.split("@", 1)[0]


def requests_audio_mime_type(request_body: Mapping[str, object]) -> bool:
    match request_body.get("generationConfig") or request_body.get("generation_config"):
        case {
            "responseFormat": {
                "audio": {"mimeType": str() | int() | float() as mime_type}
                | {"mime_type": str() | int() | float() as mime_type}
            }
        }:
            return bool(mime_type)
        case {
            "response_format": {
                "audio": {"mimeType": str() | int() | float() as mime_type}
                | {"mime_type": str() | int() | float() as mime_type}
            }
        }:
            return bool(mime_type)
        case _:
            return False


_GEMINI_SPEECH_CONFIG_KEY_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "multi_speaker_voice_config": "multiSpeakerVoiceConfig",
        "speaker_voice_configs": "speakerVoiceConfigs",
        "voice_config": "voiceConfig",
        "prebuilt_voice_config": "prebuiltVoiceConfig",
        "voice_name": "voiceName",
        "replicated_voice_config": "replicatedVoiceConfig",
        "mime_type": "mimeType",
        "voice_sample_audio": "voiceSampleAudio",
        "language_code": "languageCode",
        "speaker_alias": "speakerAlias",
        "speaker_id": "speakerId",
    }
)


_SPEECH_CONFIG_ITEMS: Final = TypeAdapter(list[object])
_CALLER_HEADERS: Final = TypeAdapter(dict[str, str])
_REQUEST_PARAMS: Final = TypeAdapter(dict[str, object])


def _normalize_gemini_speech_config_item(value: Mapping[str, object]) -> Mapping[str, object]:
    normalized: Final[dict[str, object]] = {}  # mutable-ok: iterative traversal fills nested output without recursion

    pending: Final[  # mutable-ok: the explicit worklist must grow while avoiding recursive traversal
        list[tuple[dict[str, object] | list[object], Mapping[str, object] | list[object]]]
    ] = [(normalized, value)]
    while pending:
        target, source = pending.pop()
        items = source.items() if isinstance(source, Mapping) else enumerate(source)
        for key, item in items:
            if isinstance(source, Mapping):
                if not isinstance(key, str):
                    continue
                normalized_key: str | int = _GEMINI_SPEECH_CONFIG_KEY_MAP.get(key, key)
            else:
                normalized_key = len(target)

            child: object
            if (mapping_item := as_str_mapping(item)) is not None:
                pending.append((child_mapping := dict[str, object](), mapping_item))
                child = child_mapping
            elif isinstance(item, list):
                pending.append((child_items := list[object](), _SPEECH_CONFIG_ITEMS.validate_python(item)))
                child = child_items
            else:
                child = (
                    OPENAI_TO_GEMINI_TTS_VOICES.get(item.lower(), item)
                    if normalized_key == "voiceName" and isinstance(item, str)
                    else item
                )

            if isinstance(target, dict):
                assert isinstance(normalized_key, str)
                target[normalized_key] = child
            else:
                target.append(child)

    return normalized


def _gemini_speaker_voice_config(item: object) -> object:
    speaker_config: Final = as_str_mapping(item)
    if speaker_config is None:
        return item
    speaker: Final = speaker_config.get("speakerAlias")
    voice_name: Final = speaker_config.get("speakerId")
    if not isinstance(speaker, str) or not isinstance(voice_name, str):
        return item
    return {
        **{key: value for key, value in speaker_config.items() if key not in ("speakerAlias", "speakerId")},
        "speaker": speaker,
        "voiceConfig": {
            "prebuiltVoiceConfig": {"voiceName": OPENAI_TO_GEMINI_TTS_VOICES.get(voice_name.lower(), voice_name)}
        },
    }


def _with_gemini_speaker_voice_configs(config: Mapping[str, object]) -> Mapping[str, object]:
    multi_speaker_config: Final = as_str_mapping(config.get("multiSpeakerVoiceConfig"))
    if multi_speaker_config is None:
        return config
    speaker_voice_configs: Final = multi_speaker_config.get("speakerVoiceConfigs")
    if not isinstance(speaker_voice_configs, list):
        return config
    return {
        **config,
        "multiSpeakerVoiceConfig": {
            **multi_speaker_config,
            "speakerVoiceConfigs": [
                _gemini_speaker_voice_config(item)
                for item in _SPEECH_CONFIG_ITEMS.validate_python(speaker_voice_configs)
            ],
        },
    }


_GEMINI_SPEECH_CONFIG_FIELDS: Final = frozenset({"voiceConfig", "multiSpeakerVoiceConfig", "languageCode"})

_GEMINI_VOICE_CONFIG_FIELDS: Final = MappingProxyType(
    {
        "prebuiltVoiceConfig": frozenset({"voiceName"}),
        "replicatedVoiceConfig": frozenset({"mimeType", "voiceSampleAudio"}),
        "voice": None,
    }
)

_GEMINI_SPEAKER_VOICE_CONFIG_FIELDS: Final = frozenset({"speaker", "voiceConfig"})


def _known_leaf_fields(value: object, fields: frozenset[str] | None) -> object:
    mapping: Final = as_str_mapping(value)
    if fields is None or mapping is None:
        return value
    return {key: item for key, item in mapping.items() if key in fields}


def _known_voice_config_fields(value: object) -> object:
    mapping: Final = as_str_mapping(value)
    if mapping is None:
        return value
    return {
        key: _known_leaf_fields(item, _GEMINI_VOICE_CONFIG_FIELDS[key])
        for key, item in mapping.items()
        if key in _GEMINI_VOICE_CONFIG_FIELDS
    }


def _known_speaker_voice_config_fields(value: object) -> object:
    mapping: Final = as_str_mapping(value)
    if mapping is None:
        return value
    return {
        key: _known_voice_config_fields(item) if key == "voiceConfig" else item
        for key, item in mapping.items()
        if key in _GEMINI_SPEAKER_VOICE_CONFIG_FIELDS
    }


def _known_speaker_voice_configs(value: object) -> object:
    if not isinstance(value, list):
        return value
    return [_known_speaker_voice_config_fields(item) for item in _SPEECH_CONFIG_ITEMS.validate_python(value)]


def _known_multi_speaker_voice_config_fields(value: object) -> object:
    mapping: Final = as_str_mapping(value)
    if mapping is None:
        return value
    return {key: _known_speaker_voice_configs(item) for key, item in mapping.items() if key == "speakerVoiceConfigs"}


def _known_speech_config_field(key: str, value: object) -> object:
    match key:
        case "voiceConfig":
            return _known_voice_config_fields(value)
        case "multiSpeakerVoiceConfig":
            return _known_multi_speaker_voice_config_fields(value)
        case _:
            return value


def _known_gemini_speech_fields(
    config: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: provider request serialization requires a concrete dict
    return {
        key: _known_speech_config_field(key, item)
        for key, item in config.items()
        if key in _GEMINI_SPEECH_CONFIG_FIELDS
    }


_GEMINI_TTS_VOICE_KEYS: Final = frozenset(
    {
        "format",
        "name",
        "voice",
        "voiceName",
        "voice_name",
        "languageCode",
        "language_code",
        "modelName",
        "model_name",
        "speechConfig",
        "speech_config",
        "voiceConfig",
        "voice_config",
        "multiSpeakerVoiceConfig",
        "multi_speaker_voice_config",
    }
)


_NAMED_UNKNOWN_VOICE_KEYS: Final = 10

_NAMED_VOICE_KEY_CHARS: Final = 100

_NAMED_AUDIO_FORMAT_CHARS: Final = 100


def _gemini_speech_config_wrapper(value: Mapping[str, object]) -> Mapping[str, object] | None:
    camel_config: Final = as_str_mapping(value.get("speechConfig"))
    return camel_config if camel_config is not None else as_str_mapping(value.get("speech_config"))


def named_voice_keys(keys: Sequence[str]) -> str:
    named: Final = ", ".join(key[:_NAMED_VOICE_KEY_CHARS] for key in keys[:_NAMED_UNKNOWN_VOICE_KEYS])
    unnamed_count: Final = len(keys) - _NAMED_UNKNOWN_VOICE_KEYS
    return f"{named}, and {unnamed_count} more" if unnamed_count > 0 else named


def reject_unknown_gemini_tts_voice_keys(
    model: str, voice: Mapping[str, object], drop_params: bool, llm_provider: str | None
) -> None:
    nested_config: Final = _gemini_speech_config_wrapper(voice) or {}
    ignored_wrapper: Final = (
        ("speech_config",)
        if as_str_mapping(voice.get("speechConfig")) is not None and voice.get("speech_config") is not None
        else ()
    )
    unknown_keys: Final = sorted(
        {*(key for key in (*voice, *nested_config) if key not in _GEMINI_TTS_VOICE_KEYS), *ignored_wrapper}
    )
    if not unknown_keys or drops_unsupported_params(drop_params):
        return
    raise UnsupportedParamsError(
        message=(
            f"{model} does not support the voice keys {named_voice_keys(unknown_keys)}. Remove them, or set "
            "drop_params to ignore them."
        ),
        model=model,
        llm_provider=llm_provider,
    )


def _named_audio_format(audio_format: object) -> str:
    if isinstance(audio_format, str):
        return repr(audio_format[:_NAMED_AUDIO_FORMAT_CHARS])
    return repr(audio_format)[:_NAMED_AUDIO_FORMAT_CHARS]


def unsupported_gemini_tts_audio_format(
    model: str, audio: Mapping[str, object], drop_params: bool, llm_provider: str | None
) -> UnsupportedParamsError | None:
    nested_config: Final = _gemini_speech_config_wrapper(audio) or {}
    audio_format: Final = next(
        (item for item in (audio.get("format"), nested_config.get("format")) if item not in (None, "pcm16")), None
    )
    if audio_format is None or drops_unsupported_params(drop_params):
        return None
    return UnsupportedParamsError(
        message=(
            f"{model} does not take the audio format {_named_audio_format(audio_format)}. Set format "
            "to 'pcm16' or leave it out, or set drop_params to drop it."
        ),
        model=model,
        llm_provider=llm_provider,
    )


def gemini_image_candidate_count(
    model: str,
    request_params: Mapping[str, object],
    drop_params: bool,
    allowed_openai_params: Sequence[str],
    llm_provider: str,
) -> Literal["forward", "drop"] | UnsupportedParamsError:
    n: Final = request_params.get("n")
    modalities: Final = request_params.get("modalities")
    asks_for_several_candidates: Final = isinstance(n, int) and n > 1
    asks_for_image_output: Final = isinstance(modalities, list) and "image" in modalities
    if not (asks_for_several_candidates and asks_for_image_output):
        return "forward"
    if "n" in allowed_openai_params:
        return "forward"
    if drops_unsupported_params(drop_params):
        return "drop"
    return UnsupportedParamsError(
        message=(
            f"n={n} with image output in modalities is not supported for {model}. Gemini returns one candidate "
            "per request when modalities asks for image output, so send n=1 or separate requests, or set "
            "drop_params to drop n."
        ),
        model=model,
        llm_provider=llm_provider,
    )


def normalize_gemini_speech_config(
    value: Mapping[str, object], drop_unknown_fields: bool = False
) -> dict[str, object]:  # mutable-ok: provider request serialization requires a concrete dict
    speech_config: Final = {**_with_gemini_speaker_voice_configs(mapped_gemini_speech_config(value))}
    return _known_gemini_speech_fields(speech_config) if drop_unknown_fields else speech_config


def _merge_gemini_speech_config_wrapper(value: Mapping[str, object]) -> Mapping[str, object]:
    nested_config: Final = _gemini_speech_config_wrapper(value)
    if nested_config is None:
        return value
    return {
        **{key: item for key, item in value.items() if key not in ("speechConfig", "speech_config")},
        **nested_config,
    }


def mapped_gemini_speech_config(
    value: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: provider request serialization requires a concrete dict
    normalized: Final = _normalize_gemini_speech_config_item(_merge_gemini_speech_config_wrapper(value))
    speech_config: Final = {key: item for key, item in normalized.items() if key in _GEMINI_SPEECH_CONFIG_FIELDS}
    voice_name: Final = next(
        (normalized[key] for key in ("name", "voiceName", "voice") if isinstance(normalized.get(key), str)),
        None,
    )
    if "voiceConfig" in speech_config or "multiSpeakerVoiceConfig" in speech_config or not isinstance(voice_name, str):
        return speech_config
    return {
        **speech_config,
        "voiceConfig": {
            "prebuiltVoiceConfig": {"voiceName": OPENAI_TO_GEMINI_TTS_VOICES.get(voice_name.lower(), voice_name)}
        },
    }


class _CallerFunction(TypedDict, total=False):
    parameters: ReadOnly[Mapping[str, object]]


class _CallerTool(_CallerFunction, total=False):
    function: ReadOnly[_CallerFunction]


class _ToolChoiceParam(TypedDict, total=False):
    tool_choice: ReadOnly[object]


class _CandidateCountParams(TypedDict, total=False):
    n: ReadOnly[object]
    modalities: ReadOnly[object]


_CALLER_TOOL: Final = TypeAdapter(_CallerTool)
_CALLER_TOOLS: Final = TypeAdapter(tuple[object, ...])
_TOOL_CHOICE_PARAM: Final = TypeAdapter(_ToolChoiceParam)
_CANDIDATE_COUNT_PARAMS: Final = TypeAdapter(_CandidateCountParams)


def _caller_parameters(tool: object) -> Mapping[str, object] | None:
    try:
        caller_tool: Final = _CALLER_TOOL.validate_python(tool)
    except ValidationError:
        return None
    return caller_tool.get("function", caller_tool).get("parameters")


def _parameters_json_schema(
    tool: object, forced_function_names: frozenset[str], pin_consts: bool
) -> Mapping[str, JsonValue] | None:
    parameters: Final = _caller_parameters(tool)
    json_schema: Final = None if parameters is None else parameters_json_schema_for(parameters, pin_consts)
    if json_schema is None or declared_function_name(tool) not in forced_function_names:
        return json_schema
    return None if has_bare_root_union(json_schema) else json_schema


def _parameters_json_schemas(
    tools: object,
    allow_parameters_json_schema: bool,
    forced_function_names: frozenset[str],
    pin_consts: bool,
) -> tuple[Mapping[str, JsonValue] | None, ...]:
    return tuple(
        _parameters_json_schema(tool, forced_function_names, pin_consts) if allow_parameters_json_schema else None
        for tool in _CALLER_TOOLS.validate_python(tools)
    )


def _as_chat_tool(member: JsonValue) -> JsonValue:
    if not isinstance(member, dict) or "function" in member or "name" not in member:
        return member
    return {
        "type": member.get("type", "function"),
        "function": {key: value for key, value in member.items() if key != "type"},
    }


def _namespace_members_or_tool(tool: JsonValue) -> tuple[JsonValue, ...]:
    if not isinstance(tool, dict) or tool.get("type") != "namespace":
        return (tool,)
    members: Final = tool.get("tools")
    return _members_named_once(tuple(map(_as_chat_tool, members))) if isinstance(members, list) else ()


def _members_named_once(
    members: tuple[JsonValue, ...], repeatable: frozenset[str] = frozenset()
) -> tuple[JsonValue, ...]:
    names: Final = tuple(map(declared_function_name, members))
    first_index: Final = {name: index for index, name in reversed(tuple(enumerate(names)))}
    return tuple(
        member
        for index, (member, name) in enumerate(zip(members, names, strict=True))
        if name is None or name in repeatable or first_index[name] == index
    )


def _is_namespace(tool: JsonValue) -> bool:
    return isinstance(tool, dict) and tool.get("type") == "namespace"


def _top_level_name(tool: JsonValue) -> str | None:
    name: Final = tool.get("name") if isinstance(tool, dict) else None
    return name if isinstance(name, str) and name else None


def _listed(tools: JsonValue) -> tuple[JsonValue, ...]:
    return tuple(tools) if isinstance(tools, list) else ()


def _declarable_names(tool: JsonValue) -> tuple[str | None, ...]:
    members: Final = _namespace_members_or_tool(tool) if _is_namespace(tool) else ()
    return (declared_function_name(tool), *map(declared_function_name, members))


def _declarable_names_in(tools: tuple[JsonValue, ...]) -> tuple[str | None, ...]:
    return tuple(chain.from_iterable(map(_declarable_names, tools)))


def _inlines_members(tool: JsonValue, declarable: Counter[str | None]) -> bool:
    members: Final = _namespace_members_or_tool(tool)
    return _is_namespace(tool) and all(
        name is None or declarable[name] == 1 for name in map(declared_function_name, members)
    )


def _qualified_member(namespace: str, member: JsonValue) -> JsonValue:
    function: Final = member.get("function") if isinstance(member, dict) else None
    name: Final = declared_function_name(member)
    if not isinstance(member, dict) or not isinstance(function, dict) or name is None:
        return member
    return {**member, "function": {**function, "name": qualified_member_name(namespace, name)}}


def _as_declared(
    tool: JsonValue, declarable: Counter[str | None], top_level_names: frozenset[str]
) -> tuple[JsonValue, ...]:
    if _inlines_members(tool, declarable):
        return _namespace_members_or_tool(tool)
    namespace: Final = _top_level_name(tool)
    if not _is_namespace(tool) or namespace is None:
        return (tool,)
    qualified: Final = (_qualified_member(namespace, member) for member in _namespace_members_or_tool(tool))
    return tuple(member for member in qualified if declared_function_name(member) not in top_level_names)


def _top_level_names_in(tools: tuple[JsonValue, ...]) -> frozenset[str]:
    top_level_tools: Final = (tool for tool in tools if not _is_namespace(tool))
    return frozenset(name for name in map(declared_function_name, top_level_tools) if name is not None)


def _with_namespace_members_inlined(tools: JsonValue, outside_declarations: JsonValue = None) -> JsonValue:
    if not isinstance(tools, list) or not any(map(_is_namespace, tools)):
        return tools
    candidates: Final = (*tools, *_listed(outside_declarations))
    declarable: Final = Counter(_declarable_names_in(candidates))
    top_level_names: Final = _top_level_names_in(candidates)
    declared: Final = chain.from_iterable(_as_declared(tool, declarable, top_level_names) for tool in tools)
    return list(_members_named_once(tuple(declared), top_level_names))


def _names_each_tool_declares(candidates: tuple[JsonValue, ...]) -> tuple[tuple[str, ...], ...]:
    declarable: Final = Counter(_declarable_names_in(candidates))
    top_level_names: Final = _top_level_names_in(candidates)
    declared: Final = tuple(
        tuple(filter(None, map(declared_function_name, _as_declared(tool, declarable, top_level_names))))
        for tool in candidates
    )
    owned_names: Final = chain.from_iterable(zip(repeat(index), names) for index, names in enumerate(declared))
    first_owner: Final = {name: index for index, name in reversed(tuple(owned_names))}
    return tuple(tuple(name for name in names if first_owner[name] == index) for index, names in enumerate(declared))


def _qualified_names_of(
    member_name: str, tool: JsonValue, declarable: Counter[str | None], declared: tuple[str, ...]
) -> tuple[str, ...]:
    namespace: Final = _top_level_name(tool)
    if not _is_namespace(tool) or namespace is None or _inlines_members(tool, declarable):
        return ()
    qualified: Final = qualified_member_name(namespace, member_name)
    return (qualified,) if qualified in declared else ()


def _forced_function_names(name: str, tools: JsonValue) -> tuple[str, ...]:
    candidates: Final = _listed(tools)
    if any(not _is_namespace(tool) and declared_function_name(tool) == name for tool in candidates):
        return (name,)
    declarable: Final = Counter(_declarable_names_in(candidates))
    declared_by_tool: Final = tuple(zip(candidates, _names_each_tool_declares(candidates), strict=True))
    named: Final = next(
        (declared for tool, declared in declared_by_tool if _is_namespace(tool) and _top_level_name(tool) == name),
        None,
    )
    if named is not None:
        return named
    qualified: Final = chain.from_iterable(
        _qualified_names_of(name, tool, declarable, declared) for tool, declared in declared_by_tool
    )
    return tuple(qualified) or (name,)


def _function_names_forced_by(tool_choice: object, tools: JsonValue) -> frozenset[str]:
    match tool_choice:
        case "required":
            return frozenset(chain.from_iterable(_names_each_tool_declares(_listed(tools))))
        case {"function": {"name": str() as name}}:
            return frozenset(_forced_function_names(name, tools))
        case _:
            return frozenset()


class VertexAIBaseConfig:
    def get_mapped_special_auth_params(self) -> dict[str, str]:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: Mapping[str, object], optional_params: dict[str, object]):
        mapped_params: Final = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> list[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    def get_us_regions(self) -> list[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-south1",
            "us-west1",
            "us-west4",
            "us-west5",
        ]


class VertexGeminiConfig(VertexAIBaseConfig, BaseConfig):
    """
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

    The class `VertexAIConfig` provides configuration for the VertexAI's API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    - `response_mime_type` (str): The MIME type of the response. The default value is 'text/plain'.

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    - `frequency_penalty` (float): This parameter is used to penalize the model from repeating the same output. The default value is 0.0.

    - `presence_penalty` (float): This parameter is used to penalize the model from generating the same output as the input. The default value is 0.0.

    - `seed` (int): The seed value is used to help generate the same output for the same input. The default value is None.

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    _route: ClassVar[str] = "vertex_ai"
    _removes_undeclared_thinking_levels: ClassVar[bool] = False
    _service_tier_aliases: ClassVar[Mapping[str, str]] = SERVICE_TIER_ALIASES

    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    response_mime_type: str | None = None
    candidate_count: int | None = None
    stop_sequences: list | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    def __init__(
        self,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        response_mime_type: str | None = None,
        candidate_count: int | None = None,
        stop_sequences: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        locals_: Final[Mapping[str, object]] = MappingProxyType(locals().copy())
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_json_schema_from_pydantic_object(self, response_format: type["BaseModel"] | dict | None) -> dict | None:
        """
        Override to use Pydantic's model_json_schema() instead of OpenAI's
        to_strict_json_schema().

        OpenAI's to_strict_json_schema() inlines all $ref references, which
        dramatically increases schema nesting depth and causes Gemini to reject
        schemas with 'exceeds maximum allowed nesting depth' errors.

        Pydantic's model_json_schema() preserves $ref/$defs, keeping the schema
        compact. Gemini 2.0+ (responseJsonSchema) natively supports $ref, and
        Gemini 1.5 (responseSchema) handles unpacking via _build_vertex_schema.

        See: https://github.com/BerriAI/litellm/issues/21014
        """
        from pydantic import BaseModel as _BaseModel

        if response_format is None:
            return None

        if isinstance(response_format, dict):
            return response_format

        if isinstance(response_format, type) and issubclass(response_format, _BaseModel):
            schema: Final = response_format.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {
                    "schema": schema,
                    "name": response_format.__name__,
                    "strict": True,
                },
            }

        # Fallback: delegate to parent for unknown types
        return super().get_json_schema_from_pydantic_object(response_format)

    @staticmethod
    def _is_gemini_3_or_newer(model: str) -> bool:
        """
        Check if the model is Gemini 3 or newer.
        """
        model_name: Final = model.split("/")[-1].lower()
        is_vertex_fine_tuned_model: Final = model_name.isdigit() or (
            model.startswith("gemini/") and not model_name.startswith("gemini-")
        )
        if not model_name or is_vertex_fine_tuned_model or model_name.startswith("gemma-"):
            return False
        # Pre-Gemini 3 models: gemini-1.x, gemini-2.x, gemini-pro, gemini-flash, gemini-exp
        if re.match(r"^gemini-(?:[12](?:\.\d+)?|exp|(?:pro|flash)(?!-(?:lite-)?latest$))(?:-|$)", model_name):
            return False
        return True

    @classmethod
    def is_gemini_3_or_newer(
        cls,
        model: str,
    ) -> bool:
        return cls._is_gemini_3_or_newer(model)

    @staticmethod
    def _forward_gemini_function_call_id(model: str) -> bool:
        """
        Whether to include `id` on function_call / function_response parts.

        Gemini 3+ accepts (and returns) `id` for strict tool-call matching, on Vertex AI and
        Google AI Studio alike. Older Gemini models reject the field with HTTP 400.
        """
        return VertexGeminiConfig._is_gemini_3_or_newer(model)

    @classmethod
    def forward_gemini_function_call_id(
        cls,
        model: str,
    ) -> bool:
        return cls._forward_gemini_function_call_id(model)

    @classmethod
    def supports_multimodal_function_responses(cls, model: str, custom_llm_provider: str | None = None) -> bool:
        return gemini_media_capability(
            model, "supports_media_in_tool_response", custom_llm_provider=custom_llm_provider
        )

    def _supports_penalty_parameters(self, model: str) -> bool:
        # Gemini 3 models do not support penalty parameters
        if VertexGeminiConfig._is_gemini_3_or_newer(model):
            return False
        unsupported_models: Final = ["gemini-2.5-pro-preview-06-05"]
        if model in unsupported_models:
            return False
        return True

    @staticmethod
    def supports_sampling_params(model: str) -> bool:
        """Whether Google still honors ``temperature`` / ``top_p`` / ``top_k`` for ``model``.

        Driven by ``supports_sampling_params`` in the model map: Gemini 3.6 Flash, 3.5 Flash-Lite
        and later accept the fields but ignore them, and a later generation will 400 on them.
        A model the map says nothing about keeps forwarding them. ``temperature`` and ``top_p``
        leave ``get_supported_openai_params`` when the flag is off, so the shared unsupported-param
        check raises or drops them and ``allowed_openai_params`` still forwards them verbatim;
        ``top_k`` is a provider-specific kwarg that bypasses that check and is gated while the
        request body is built."""
        rows: Final = cast(Mapping[str, Mapping[str, object]], litellm.model_cost)  # cast-ok: JSON rows by model id
        bare: Final = model.split("/")[-1]
        for candidate in (model, bare, f"vertex_ai/{bare}", f"gemini/{bare}"):
            if isinstance(flag := rows.get(candidate, {}).get("supports_sampling_params"), bool):
                return flag
        return True

    def _supports_search_tools_with_functions(self, model: str) -> bool:
        rows: Final = cast(Mapping[str, Mapping[str, object]], litellm.model_cost)  # cast-ok: JSON rows by model id
        bare: Final = model.split("/")[-1]
        flags: Final = (
            rows.get(candidate, {}).get("supports_search_tools_with_functions")
            for candidate in (f"{self._route}/{bare}", bare)
        )
        return next((flag for flag in flags if isinstance(flag, bool)), False)

    @staticmethod
    def caller_tool_config_keeps_the_old_search_drop(tool_config: JsonValue | None) -> bool:
        fields: Final = as_str_mapping(tool_config)
        return fields is not None and _tool_config_keeps_the_old_search_drop(fields)

    def _keeps_search_tools_with_functions(self, model: str, *param_sources: Mapping[str, object]) -> bool:
        keeps_the_old_drop: Final = any(
            _turns_server_side_tools_off(params) or _pins_function_calling_mode(params) for params in param_sources
        )
        return not keeps_the_old_drop and self._supports_search_tools_with_functions(model)

    @staticmethod
    def reject_unsupported_sampling_param(model: str, param: str, drop_params: bool) -> None:
        """Return under ``drop_params`` so the caller drops ``param``, else raise the client-side 400
        that names the escape hatch."""
        if litellm.drop_params or drop_params:
            return
        raise UnsupportedParamsError(
            message=(
                f"{model} does not support {param}: the model map flags it `supports_sampling_params: false` "
                "because Google ignores sampling params on this model. "
                "To drop unsupported params, set `litellm.drop_params = True`."
            ),
            status_code=400,
        )

    def get_supported_openai_params(self, model: str) -> list[str]:
        sampling_params: Final = ("temperature", "top_p", "top_k") if self.supports_sampling_params(model) else ()
        supported_params: Final = [
            *sampling_params,
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "functions",
            "tool_choice",
            "response_format",
            "n",
            "stop",
            "extra_headers",
            "seed",
            "logprobs",
            "top_logprobs",
            "modalities",
            "audio",
            "parallel_tool_calls",
            "web_search_options",
            "include_server_side_tool_invocations",
            "service_tier",
        ]

        # Add penalty parameters only for non-preview models
        if self._supports_penalty_parameters(model):
            supported_params.extend(["frequency_penalty", "presence_penalty"])

        if supports_reasoning(model) or self._is_gemini_3_or_newer(model):
            supported_params.append("reasoning_effort")
            supported_params.append("thinking")

        return supported_params

    def map_tool_choice_values(
        self, model: str, tool_choice: str | dict, tools: JsonValue = None, drop_params: bool = False
    ) -> ToolConfig | None:
        if tool_choice == "none":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="NONE"))
        elif tool_choice == "required":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY"))
        elif tool_choice == "auto":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            name: Final = tool_choice.get("function", {}).get("name", "")
            forced: Final = _forced_function_names(name, tools)
            if forced:
                return ToolConfig(
                    functionCallingConfig=FunctionCallingConfig(mode="ANY", allowed_function_names=list(forced))
                )
            if drops_unsupported_params(drop_params):
                return None
            raise litellm.utils.UnsupportedParamsError(
                message=f"VertexAI can't force tool_choice '{name}', since none of that namespace's members can be sent to Gemini as a function. Name one of its functions instead, or set `litellm.drop_params = True` to drop tool_choice.",
                status_code=400,
            )
        else:
            raise litellm.utils.UnsupportedParamsError(
                message=f"VertexAI doesn't support tool_choice={tool_choice}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.",
                status_code=400,
            )

    def _map_web_search_options(self, value: dict[str, object]) -> Tools:
        """
        Base Case: empty dict

        Google doesn't support user_location or search_context_size params
        """
        return Tools(googleSearch={})

    def map_web_search_options(
        self,
        value: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> Tools:
        return self._map_web_search_options(value)

    @staticmethod
    def _search_tool_keys() -> set[str]:
        return {
            VertexToolName.GOOGLE_SEARCH.value,
            VertexToolName.GOOGLE_SEARCH_RETRIEVAL.value,
            VertexToolName.ENTERPRISE_WEB_SEARCH.value,
            VertexToolName.URL_CONTEXT.value,
            "google_search",
            "google_search_retrieval",
            "enterprise_web_search",
            "urlContext",
        }

    @classmethod
    def search_tool_keys(
        cls,
    ) -> set[str]:  # mutable-ok: mirrors override contract
        return cls._search_tool_keys()

    @classmethod
    def _is_search_tool(cls, tool: object) -> bool:
        return not cls.search_tool_keys().isdisjoint(as_str_mapping(tool) or ())

    @classmethod
    def _mixes_search_tools_with_functions(cls, tools: Sequence[object]) -> bool:
        return any((as_str_mapping(tool) or {}).get("function_declarations") for tool in tools) and any(
            map(cls._is_search_tool, tools)
        )

    @staticmethod
    def _is_search_retrieval_tool(tool: object) -> bool:
        return VertexToolName.GOOGLE_SEARCH_RETRIEVAL.value in (as_str_mapping(tool) or ())

    @classmethod
    def _drop_search_tools_mixed_with_functions(
        cls,
        optional_params: dict[str, object],  # mutable-ok: map_openai_params builds the request params in place
        keeps_search_tools: bool = False,
        drops_search_retrieval: bool = False,
    ) -> None:
        """
        Drop search tools from optional_params when mixed with function declarations
        and include_server_side_tool_invocations is not enabled. A model that keeps
        search tools next to function declarations (Gemini 3+) keeps both and gets
        include_server_side_tool_invocations set instead, minus googleSearchRetrieval,
        which Gemini 3+ rejects. A caller who set include_server_side_tool_invocations
        keeps googleSearchRetrieval unless drop_params is set, as before.

        Runs after map_openai_params merges tools and web_search_options so both
        code paths (single _map_function call vs split tools + web_search_options)
        get the same conflict resolution.
        """
        tools: Final = as_object_list(optional_params.get("tools")) or ()
        if not cls._mixes_search_tools_with_functions(tools):
            return
        if keeps_search_tools:
            caller_flag_on: Final = optional_params.get("include_server_side_tool_invocations") is True
            cls._keep_search_tools_next_to_functions(
                optional_params, tools, drops_search_retrieval or not caller_flag_on
            )
            return
        if optional_params.get("include_server_side_tool_invocations"):
            return
        optional_params["tools"] = list(cls.without_search_tools_next_to_functions(tools))

    @classmethod
    def without_search_tools_next_to_functions(cls, tools: Sequence[object]) -> tuple[object, ...]:
        if not cls._mixes_search_tools_with_functions(tools):
            return tuple(tools)
        verbose_logger.warning(
            "Vertex AI does not support mixing function declarations with "
            "search tools (googleSearch, enterpriseWebSearch, urlContext, "
            "googleSearchRetrieval) in the same request. Dropping search "
            "tools and keeping function declarations. To use search tools, "
            "send a request without function calling tools."
        )
        return tuple(tool for tool in tools if not cls._is_search_tool(tool))

    @classmethod
    def _keep_search_tools_next_to_functions(
        cls,
        optional_params: dict[str, object],  # mutable-ok: map_openai_params builds the request params in place
        tools: Sequence[object],
        drops_search_retrieval: bool,
    ) -> None:
        kept: Final = (
            [tool for tool in tools if not cls._is_search_retrieval_tool(tool)] if drops_search_retrieval else tools
        )
        if len(kept) < len(tools):
            verbose_logger.warning(
                "Gemini 3+ does not support googleSearchRetrieval next to function declarations. Dropping it, "
                "use googleSearch instead."
            )
            optional_params["tools"] = kept  # rebind-ok: map_openai_params builds the request params in place
        if cls._mixes_search_tools_with_functions(kept):
            optional_params["include_server_side_tool_invocations"] = True  # rebind-ok: built in place as above

    @classmethod
    def drop_search_tools_mixed_with_functions(
        cls,
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> None:
        return cls._drop_search_tools_mixed_with_functions(optional_params)

    def _map_service_tier_param(self, value: str, optional_params: dict[str, object], drop_params: bool) -> None:
        unprefixed: Final = value.lower().removeprefix("service_tier_")
        normalized: Final = self._service_tier_aliases.get(unprefixed, unprefixed)
        if normalized in SENT_SERVICE_TIERS:
            optional_params["service_tier"] = normalized
            return
        if normalized in PROVIDER_DEFAULT_SERVICE_TIERS or drops_unsupported_params(drop_params):
            return
        raise _unsupported_service_tier(value)

    def _transform_computer_use_config(self, computer_use_config: Mapping[str, object]) -> dict[str, object]:
        """
        Transform Computer Use configuration to Gemini API format.

        Args:
            computer_use_config: The computer use configuration from LiteLLM

        Returns:
            Transformed computer use configuration for Gemini API
        """
        transformed_config: Final[dict[str, object]] = {}

        # Transform environment values if needed
        if "environment" in computer_use_config:
            env_value: Final = computer_use_config["environment"]
            if env_value == "browser":
                transformed_config["environment"] = "ENVIRONMENT_BROWSER"
            elif env_value == "unspecified":
                transformed_config["environment"] = "ENVIRONMENT_UNSPECIFIED"
            elif env_value in ["ENVIRONMENT_BROWSER", "ENVIRONMENT_UNSPECIFIED"]:
                # Already in correct format
                transformed_config["environment"] = env_value
            else:
                verbose_logger.info(
                    "Invalid environment value for computer_use: %s. Supported: 'browser', 'unspecified', 'ENVIRONMENT_BROWSER', 'ENVIRONMENT_UNSPECIFIED'",
                    env_value,
                )

        # Transform excluded_predefined_functions to camelCase
        if "excluded_predefined_functions" in computer_use_config:
            transformed_config["excludedPredefinedFunctions"] = computer_use_config["excluded_predefined_functions"]
        elif "excludedPredefinedFunctions" in computer_use_config:
            transformed_config["excludedPredefinedFunctions"] = computer_use_config["excludedPredefinedFunctions"]

        return transformed_config

    def _extract_google_maps_retrieval_config(
        self, google_maps_config: Mapping[str, object]
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        """
        Extract location configuration from googleMaps tool for Vertex AI toolConfig.

        Supports two interface styles:
        1. Nested (recommended): {"enableWidget": "...", "retrievalConfig": {"latitude": ..., "longitude": ...}}
        2. Flat (backward compat): {"enableWidget": "...", "latitude": ..., "longitude": ...}

        Args:
            google_maps_config: The googleMaps tool configuration from LiteLLM

        Returns:
            Tuple of (cleaned_google_maps_config, retrieval_config):
                - cleaned_google_maps_config: googleMaps config without location fields
                - retrieval_config: Location config for toolConfig.retrievalConfig or None
        """
        retrieval_config: dict[str, object] | None = None
        latitude: Final = google_maps_config.get("latitude")
        longitude: Final = google_maps_config.get("longitude")
        language_code: Final = google_maps_config.get("languageCode")

        if latitude is not None and longitude is not None:
            retrieval_config = {
                "latLng": {
                    "latitude": latitude,
                    "longitude": longitude,
                }
            }
            if language_code is not None:
                retrieval_config["languageCode"] = language_code

        # Remove location fields from tool definition
        cleaned_config: Final = {
            k: v for k, v in google_maps_config.items() if k not in ["latitude", "longitude", "languageCode"]
        }

        return cleaned_config, retrieval_config

    def get_tool_value(self, tool: dict, tool_name: str) -> dict | None:
        """
        Helper function to get tool value handling both camelCase and underscore_case variants

        Args:
            tool (dict): The tool dictionary
            tool_name (str): The base tool name (e.g. "codeExecution")

        Returns:
            Optional[dict]: The tool value if found, None otherwise
        """
        # Convert camelCase to underscore_case
        underscore_name: Final = "".join(["_" + c.lower() if c.isupper() else c for c in tool_name]).lstrip("_")
        # Try both camelCase and underscore_case variants

        if tool.get(tool_name) is not None:
            return tool.get(tool_name)
        elif tool.get(underscore_name) is not None:
            return tool.get(underscore_name)
        else:
            return None

    @staticmethod
    def _resolve_search_tool_conflict(
        gtool_func_declarations: Sequence[object],
        googleSearch: dict[str, object] | None,
        googleSearchRetrieval: dict[str, object] | None,
        enterpriseWebSearch: dict[str, object] | None,
        urlContext: dict[str, object] | None,
        optional_params: dict[str, object],
        keep_search_tools_with_functions: bool = False,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None, dict[str, object] | None, dict[str, object] | None]:
        """
        Resolve the Gemini 2.x constraint: multiple Tool objects in a request must
        ALL be search tools. When function declarations are mixed with search
        tools, drop search tools to avoid 400 error.

        Skip when include_server_side_tool_invocations is enabled or the model
        keeps search tools next to function declarations (Gemini 3+).

        Note: code_execution, computerUse, and googleMaps are NOT search tools
        and CAN coexist with function declarations, so they are preserved.

        Ref: https://github.com/BerriAI/litellm/issues/23337

        Returns:
            tuple of (googleSearch, googleSearchRetrieval, enterpriseWebSearch, urlContext)
        """
        has_search_tools: Final = any(
            v is not None
            for v in [
                googleSearch,
                googleSearchRetrieval,
                enterpriseWebSearch,
                urlContext,
            ]
        )
        keeps_search_tools: Final = keep_search_tools_with_functions or optional_params.get(
            "include_server_side_tool_invocations", False
        )
        if gtool_func_declarations and has_search_tools and not keeps_search_tools:
            verbose_logger.warning(
                "Vertex AI does not support mixing function declarations with "
                "search tools (googleSearch, enterpriseWebSearch, urlContext, "
                "googleSearchRetrieval) in the same request. Dropping search "
                "tools and keeping function declarations. To use search tools, "
                "send a request without function calling tools."
            )
            return None, None, None, None

        return googleSearch, googleSearchRetrieval, enterpriseWebSearch, urlContext

    def _map_function(
        self,
        value: list[dict[str, object]],  # mutable-ok: mirrors override contract
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        send_integer_enums: bool = False,
        allow_parameters_json_schema: bool = True,
        outside_declarations: JsonValue = None,
        tool_choice: object = None,
    ) -> list[Tools]:  # mutable-ok: mirrors override contract
        return with_pins_that_fit(
            partial(
                self._build_function_tools,
                value,
                optional_params,
                send_integer_enums,
                allow_parameters_json_schema=allow_parameters_json_schema,
                outside_declarations=outside_declarations,
                tool_choice=tool_choice,
            ),
            has_const(cast(JsonValue, value)),  # cast-ok: tools are JSON request data
        )

    def _build_function_tools(
        self,
        value: list[dict[str, object]],
        optional_params: dict,
        send_integer_enums: bool = False,
        pin_consts: bool = True,
        allow_parameters_json_schema: bool = True,
        outside_declarations: JsonValue = None,
        tool_choice: object = None,
        keep_search_tools_with_functions: bool = False,
    ) -> list[Tools]:
        """
        Map OpenAI-style tools/functions to Vertex AI format.

        Args:
            value: List of tool definitions
            optional_params: Request-scoped parameters to store retrieval config
            send_integer_enums: Whether to send integer enums with string values
            pin_consts: Whether to send each scalar `const` as a one-value enum
            allow_parameters_json_schema: Whether a tool whose parameters root Vertex cannot take as `parameters`
                goes as `parametersJsonSchema`

        Returns:
            List of mapped tools in Vertex AI format

        Side effects:
            May add 'toolConfig' with 'retrievalConfig' to optional_params if
            googleMaps tools contain location data
        """
        gtool_func_declarations: Final[list[FunctionDeclaration]] = []
        googleSearch: dict[str, object] | None = None
        googleSearchRetrieval: dict[str, object] | None = None
        enterpriseWebSearch: dict[str, object] | None = None
        urlContext: dict[str, object] | None = None
        code_execution: dict[str, object] | None = None
        googleMaps: dict[str, object] | None = None
        google_maps_retrieval_config: dict[str, object] | None = None
        computerUse: dict[str, object] | None = None
        # remove 'additionalProperties' from tools
        schema_value: Final[JsonValue] = deepcopy(cast(JsonValue, value))  # cast-ok: tools are JSON request data
        parameters_json_schemas: Final = _parameters_json_schemas(
            _with_namespace_members_inlined(schema_value, outside_declarations),
            allow_parameters_json_schema,
            _function_names_forced_by(tool_choice, [*_listed(schema_value), *_listed(outside_declarations)]),
            pin_consts,
        )
        remove_additional_properties(schema_value)
        # remove 'strict' from tools
        remove_strict_from_schema(schema_value)

        inlined_tools: Final = _with_namespace_members_inlined(schema_value, outside_declarations)
        for index, parameters_json_schema in enumerate(parameters_json_schemas):
            tool = cast(list[dict], inlined_tools)[index]  # cast-ok: a deep copy keeps the caller's tool shape
            openai_function_object: ChatCompletionToolParamFunctionChunk | None = None
            if "function" in tool:  # tools list
                _openai_function_object = ChatCompletionToolParamFunctionChunk(**tool["function"])

                if (
                    parameters_json_schema is None
                    and "parameters" in _openai_function_object
                    and _openai_function_object["parameters"] is not None
                    and isinstance(_openai_function_object["parameters"], dict)
                ):  # OPENAI accepts JSON Schema, Google accepts OpenAPI schema.
                    _openai_function_object["parameters"] = build_vertex_schema(
                        _openai_function_object["parameters"],
                        send_integer_enums=send_integer_enums,
                        pin_consts=pin_consts,
                    )

                openai_function_object = _openai_function_object

            elif "name" in tool:  # functions list
                openai_function_object = ChatCompletionToolParamFunctionChunk(**tool)

            if "type" in tool and tool["type"] == "computer_use":
                computer_use_config = {k: v for k, v in tool.items() if k != "type"}
                tool = {VertexToolName.COMPUTER_USE.value: computer_use_config}
            # Handle OpenAI-style web_search and web_search_preview tools
            # Transform them to Gemini's googleSearch tool
            elif "type" in tool and tool["type"] in (
                "web_search",
                "web_search_preview",
            ):
                verbose_logger.info("Gemini: Transforming OpenAI-style '%s' tool to googleSearch", tool["type"])
                tool = {VertexToolName.GOOGLE_SEARCH.value: {}}
            # Handle tools with 'type' field (OpenAI spec compliance) Ignore this field -> https://github.com/BerriAI/litellm/issues/14644#issuecomment-3342061838
            elif "type" in tool:
                tool = {k: tool[k] for k in tool if k != "type"}
            tool_name = list(tool.keys())[0] if len(tool.keys()) == 1 else None
            if tool_name and (
                tool_name == "codeExecution" or tool_name == VertexToolName.CODE_EXECUTION.value
            ):  # code_execution maintained for backwards compatibility
                code_execution = self.get_tool_value(tool, "codeExecution")
            elif tool_name and (tool_name == VertexToolName.GOOGLE_SEARCH.value or tool_name == "google_search"):
                googleSearch = self.get_tool_value(tool, tool_name)
            elif tool_name and (
                tool_name == VertexToolName.GOOGLE_SEARCH_RETRIEVAL.value or tool_name == "google_search_retrieval"
            ):
                googleSearchRetrieval = self.get_tool_value(tool, tool_name)
            elif tool_name and (
                tool_name == VertexToolName.ENTERPRISE_WEB_SEARCH.value or tool_name == "enterprise_web_search"
            ):
                enterpriseWebSearch = self.get_tool_value(tool, tool_name)
            elif tool_name and (tool_name == VertexToolName.URL_CONTEXT.value or tool_name == "urlContext"):
                urlContext = self.get_tool_value(tool, tool_name)
            elif tool_name and (tool_name == VertexToolName.GOOGLE_MAPS.value or tool_name == "google_maps"):
                google_maps_value = self.get_tool_value(tool, VertexToolName.GOOGLE_MAPS.value)

                # Extract and transform location configuration for toolConfig
                if google_maps_value is not None:
                    (
                        googleMaps,
                        google_maps_retrieval_config,
                    ) = self._extract_google_maps_retrieval_config(google_maps_config=google_maps_value)
            elif tool_name and (tool_name == VertexToolName.COMPUTER_USE.value or tool_name == "computer_use"):
                computer_use_value = self.get_tool_value(tool, VertexToolName.COMPUTER_USE.value)

                # Transform Computer Use configuration to Gemini API format
                if computer_use_value is not None:
                    computerUse = self._transform_computer_use_config(computer_use_config=computer_use_value)
                else:
                    # Empty config - Gemini will use defaults
                    computerUse = {}
            elif openai_function_object is not None:
                gtool_func_declaration = (
                    FunctionDeclaration(name=openai_function_object["name"])
                    if parameters_json_schema is None
                    else FunctionDeclaration(
                        name=openai_function_object["name"], parametersJsonSchema=parameters_json_schema
                    )
                )
                _description = openai_function_object.get("description", None)
                _parameters = openai_function_object.get("parameters", None)
                if isinstance(_parameters, str) and len(_parameters) == 0:
                    _parameters = {
                        "type": "object",
                    }
                if _description is not None:
                    gtool_func_declaration["description"] = _description
                if _parameters is not None and parameters_json_schema is None:
                    gtool_func_declaration["parameters"] = _parameters
                gtool_func_declarations.append(gtool_func_declaration)
            else:
                # assume it's a provider-specific param
                verbose_logger.warning(
                    "Invalid tool={}. Use `litellm.set_verbose` or `litellm --detailed_debug` to see raw request."
                )

        # Build list of Tool objects - each Tool should contain exactly one type
        # per Vertex AI API spec: "A Tool object should contain exactly one type of Tool"
        _tools_list: Final[list[Tools]] = []

        (
            googleSearch,
            googleSearchRetrieval,
            enterpriseWebSearch,
            urlContext,
        ) = self._resolve_search_tool_conflict(
            gtool_func_declarations=gtool_func_declarations,
            googleSearch=googleSearch,
            googleSearchRetrieval=googleSearchRetrieval,
            enterpriseWebSearch=enterpriseWebSearch,
            urlContext=urlContext,
            optional_params=optional_params,
            keep_search_tools_with_functions=keep_search_tools_with_functions,
        )

        # Function declarations can be grouped together in one Tool
        if gtool_func_declarations:
            func_tool: Final = Tools()
            func_tool["function_declarations"] = gtool_func_declarations
            _tools_list.append(func_tool)

        # Each special tool type must be in its own Tool object
        if googleSearch is not None:
            search_tool: Final = Tools()
            search_tool[VertexToolName.GOOGLE_SEARCH.value] = googleSearch
            _tools_list.append(search_tool)
        if googleSearchRetrieval is not None:
            retrieval_tool: Final = Tools()
            retrieval_tool[VertexToolName.GOOGLE_SEARCH_RETRIEVAL.value] = googleSearchRetrieval
            _tools_list.append(retrieval_tool)
        if enterpriseWebSearch is not None:
            enterprise_tool: Final = Tools()
            enterprise_tool[VertexToolName.ENTERPRISE_WEB_SEARCH.value] = enterpriseWebSearch
            _tools_list.append(enterprise_tool)
        if code_execution is not None:
            code_tool: Final = Tools()
            code_tool[VertexToolName.CODE_EXECUTION.value] = code_execution
            _tools_list.append(code_tool)
        if urlContext is not None:
            url_tool: Final = Tools()
            url_tool[VertexToolName.URL_CONTEXT.value] = urlContext
            _tools_list.append(url_tool)
        if googleMaps is not None:
            maps_tool: Final = Tools()
            maps_tool[VertexToolName.GOOGLE_MAPS.value] = googleMaps
            _tools_list.append(maps_tool)
        if computerUse is not None:
            computer_tool: Final = Tools()
            computer_tool[VertexToolName.COMPUTER_USE.value] = computerUse
            _tools_list.append(computer_tool)

        # Add retrieval config to toolConfig if googleMaps has location data
        if google_maps_retrieval_config is not None:
            if "toolConfig" not in optional_params:
                optional_params["toolConfig"] = {}
            optional_params["toolConfig"]["retrievalConfig"] = google_maps_retrieval_config

        return _tools_list

    def map_function(
        self,
        value: list[dict[str, object]],  # mutable-ok: mirrors override contract
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        send_integer_enums: bool = False,
        allow_parameters_json_schema: bool = True,
    ) -> list[Tools]:  # mutable-ok: mirrors override contract
        return self._map_function(value, optional_params, send_integer_enums, allow_parameters_json_schema)

    def _map_response_schema(
        self,
        value: dict[str, object],
        send_integer_enums: bool = False,
        pin_consts: bool = True,
    ) -> Mapping[str, object]:
        old_schema = deepcopy(value)
        if isinstance(old_schema, list):
            for item in old_schema:
                if isinstance(item, dict):
                    item = build_vertex_schema(
                        parameters=item,
                        add_property_ordering=True,
                        send_integer_enums=send_integer_enums,
                        pin_consts=pin_consts,
                    )

        elif isinstance(old_schema, dict):
            old_schema = build_vertex_schema(
                parameters=old_schema,
                add_property_ordering=True,
                send_integer_enums=send_integer_enums,
                pin_consts=pin_consts,
            )
        return old_schema

    def apply_response_schema_transformation(
        self,
        value: dict,
        optional_params: dict[str, object],
        model: str,
        allow_integer_enums: bool = True,
        pin_consts: bool = True,
    ):
        new_value = deepcopy(value)
        # remove 'strict' from json schema (not supported by Gemini)
        new_value = remove_strict_from_schema(new_value)

        # Automatically use responseJsonSchema for Gemini 2.0+ models
        # responseJsonSchema uses standard JSON Schema format and supports additionalProperties
        # For older models (Gemini 1.5), fall back to responseSchema (OpenAPI format)
        use_json_schema: Final = supports_response_json_schema(model)

        if not use_json_schema:
            # For responseSchema, remove 'additionalProperties' (not supported)
            new_value = remove_additional_properties(new_value)

        # Handle response type
        if new_value.get("type") == "json_object":
            optional_params["response_mime_type"] = "application/json"
        elif new_value.get("type") == "text":
            optional_params["response_mime_type"] = "text/plain"

        # Extract schema from response_format
        schema = None
        if "response_schema" in new_value:
            optional_params["response_mime_type"] = "application/json"
            schema = new_value["response_schema"]
        elif new_value.get("type") == "json_schema":
            if "json_schema" in new_value and "schema" in new_value["json_schema"]:
                optional_params["response_mime_type"] = "application/json"
                schema = new_value["json_schema"]["schema"]

        if schema and isinstance(schema, dict):
            if use_json_schema:
                # Use responseJsonSchema (Gemini 2.0+ only, opt-in)
                # - Standard JSON Schema format (lowercase types)
                # - Supports additionalProperties
                # - No propertyOrdering needed
                json_schema: Final[Mapping[str, object]] = schema
                optional_params["response_json_schema"] = with_pins_that_fit(
                    partial(build_json_schema, json_schema),
                    pin_consts and has_const(cast(JsonValue, schema)),  # cast-ok: response schemas are JSON
                )
            else:
                # Use responseSchema (default, backwards compatible)
                # - OpenAPI-style format (uppercase types)
                # - No additionalProperties support
                # - Requires propertyOrdering
                optional_params["response_schema"] = with_enums_that_fit(
                    partial(self._map_response_schema, schema),
                    allow_integer_enums and self._is_gemini_3_or_newer(model),
                    pin_consts and has_const(cast(JsonValue, schema)),  # cast-ok: response schemas are JSON
                )

    @property
    def accepts_reasoning_summary_in_reasoning_effort(self) -> bool:
        return True

    @staticmethod
    def _include_thoughts(requested: bool | None) -> bool:
        if requested is not None:
            return requested
        return normalize_drop_params(litellm.gemini_include_thoughts_default) is not False

    @staticmethod
    def _include_thoughts_requested_by_reasoning_effort(reasoning_effort: object) -> bool | None:
        match reasoning_effort:
            case {"summary": str()}:
                return True
            case _:
                return None

    @staticmethod
    def _map_reasoning_effort(effort: str, model: str, reasoning_effort: object) -> GeminiThinkingConfig:
        include_thoughts: Final = VertexGeminiConfig._include_thoughts_requested_by_reasoning_effort(reasoning_effort)
        if VertexGeminiConfig._is_gemini_3_or_newer(model):
            return VertexGeminiConfig._map_reasoning_effort_to_thinking_level(effort, model, include_thoughts)
        return VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(effort, model, include_thoughts)

    @staticmethod
    def _map_reasoning_effort_to_thinking_budget(
        reasoning_effort: str,
        model: str | None = None,
        include_thoughts: bool | None = None,
    ) -> GeminiThinkingConfig:
        sends_thoughts: Final = VertexGeminiConfig._include_thoughts(include_thoughts)
        if reasoning_effort == "minimal":
            # Use model-specific minimum thinking budget or fallback
            # Check for exact matches first, then partial matches
            if model and "gemini-2.5-flash-lite" in model.lower():
                budget = DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE
            elif model and "gemini-2.5-pro" in model.lower():
                budget = DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO
            elif model and "gemini-2.5-flash" in model.lower():
                budget = DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH
            else:
                budget = DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET

            return {
                "thinkingBudget": budget,
                "includeThoughts": sends_thoughts,
            }
        elif reasoning_effort == "low":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
                "includeThoughts": sends_thoughts,
            }
        elif reasoning_effort == "medium":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                "includeThoughts": sends_thoughts,
            }
        elif reasoning_effort == "high":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
                "includeThoughts": sends_thoughts,
            }
        elif reasoning_effort == "disable":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET,
                "includeThoughts": False,
            }
        elif reasoning_effort == "none":
            return {
                "thinkingBudget": 0,
                "includeThoughts": False,
            }
        else:
            raise _unsupported_reasoning_effort(reasoning_effort)

    @staticmethod
    def _supports_minimal_thinking_level(model: str) -> bool:
        providers: Final = ("gemini", "vertex_ai")
        key: Final = "supports_minimal_reasoning_effort"
        declared_by_model_map: Final = any(
            supports_factory(model=model, custom_llm_provider=provider, key=key) for provider in providers
        )
        is_gemini3_or_newer_flash: Final = VertexGeminiConfig._is_gemini_3_or_newer(model) and "flash" in model.lower()
        return (declared_by_model_map or is_gemini3_or_newer_flash) and not any(
            is_explicitly_disabled_factory(model=model, custom_llm_provider=provider, key=key) for provider in providers
        )

    @staticmethod
    def _supports_xhigh_thinking_level(model: str) -> bool:
        return any(
            supports_factory(model=model, custom_llm_provider=provider, key="supports_xhigh_reasoning_effort")
            for provider in ("gemini", "vertex_ai")
        )

    @staticmethod
    def _map_reasoning_effort_to_thinking_level(
        reasoning_effort: str,
        model: str | None = None,
        include_thoughts: bool | None = None,
    ) -> GeminiThinkingConfig:
        """
        Map reasoning_effort to thinking_level for Gemini 3+ models.
        Args:
            reasoning_effort: The reasoning effort value
            model: The model name

        Returns:
            GeminiThinkingConfig with thinkingLevel and includeThoughts
        """
        supports_minimal: Final = bool(model) and VertexGeminiConfig._supports_minimal_thinking_level(model)
        sends_thoughts: Final = VertexGeminiConfig._include_thoughts(include_thoughts)
        if reasoning_effort == "minimal":
            if supports_minimal:
                return {"thinkingLevel": "minimal", "includeThoughts": sends_thoughts}
            else:
                return {"thinkingLevel": "low", "includeThoughts": sends_thoughts}
        elif reasoning_effort == "low":
            return {"thinkingLevel": "low", "includeThoughts": sends_thoughts}
        elif reasoning_effort == "medium":
            return {"thinkingLevel": "medium", "includeThoughts": sends_thoughts}
        elif reasoning_effort == "high":
            return {"thinkingLevel": "high", "includeThoughts": sends_thoughts}
        elif reasoning_effort == "xhigh" and model and VertexGeminiConfig._supports_xhigh_thinking_level(model):
            return {"thinkingLevel": "xhigh", "includeThoughts": sends_thoughts}
        elif reasoning_effort in ("disable", "none"):
            return {
                "thinkingLevel": "minimal" if supports_minimal else "low",
                "includeThoughts": False,
            }
        else:
            raise _unsupported_reasoning_effort(reasoning_effort)

    @staticmethod
    def _is_thinking_budget_zero(thinking_budget: int | None) -> bool:
        return thinking_budget is not None and thinking_budget == 0

    @staticmethod
    def _validate_thinking_config_conflicts(
        optional_params: dict,
        param_name: str,
        param_description: str = "thinking_budget",
    ) -> None:
        """
        Validate that thinking_level and thinking_budget are not both specified.
        """
        if "thinkingConfig" in optional_params:
            existing_config: Final = optional_params["thinkingConfig"]
            if "thinkingLevel" in existing_config:
                raise litellm.utils.UnsupportedParamsError(
                    message=(
                        f"Cannot specify both `{param_name}` (which maps to `{param_description}`) "
                        "and `thinking_level` in the same request. "
                        "For Gemini 3 models, use `thinking_level` instead."
                    ),
                    status_code=400,
                )

    @staticmethod
    def _validate_thinking_level_conflicts(
        optional_params: dict,
    ) -> None:
        """
        Validate that thinking_level and thinking_budget are not both specified.
        Called when setting thinking_level.
        """
        if "thinkingConfig" in optional_params:
            existing_config: Final = optional_params["thinkingConfig"]
            if "thinkingBudget" in existing_config:
                raise litellm.utils.UnsupportedParamsError(
                    message=(
                        "Cannot specify both `thinking_level` and `thinking_budget` in the same request. "
                        "For Gemini 3 models, use `thinking_level` instead of `thinking_budget`."
                    ),
                    status_code=400,
                )

    @staticmethod
    def _thinking_asks_for_thoughts(thinking_param: Mapping[str, object]) -> bool:
        display: Final = thinking_param.get("display")
        return isinstance(thinking_param.get("summary"), str) or (display is not None and display != "omitted")

    @staticmethod
    def _map_thinking_param(
        thinking_param: AnthropicThinkingParam,
        model: str | None = None,
    ) -> GeminiThinkingConfig:
        thinking_enabled: Final = thinking_param.get("type") == "enabled"
        thinking_budget: Final = thinking_param.get("budget_tokens")
        asks_for_thoughts: Final = VertexGeminiConfig._thinking_asks_for_thoughts(thinking_param)
        sends_thoughts: Final = VertexGeminiConfig._include_thoughts(True if asks_for_thoughts else None)

        params: Final[GeminiThinkingConfig] = {}

        # For Gemini 3+ models, use thinkingLevel instead of thinkingBudget
        if model and VertexGeminiConfig._is_gemini_3_or_newer(model):
            if thinking_enabled:
                if thinking_budget == 0:
                    params["includeThoughts"] = False
                else:
                    params["includeThoughts"] = sends_thoughts
                    if thinking_budget is not None:
                        warn_dropped_reasoning(
                            "thinking.budget_tokens",
                            thinking_budget,
                            model,
                            _LEVEL_ONLY_BUDGET_REASON,
                            drop_params_gated=False,
                        )
                    # Follow provider defaults unless explicitly opted into legacy behavior.
                    if litellm.enable_gemini_default_thinking_level_low is True:
                        params["thinkingLevel"] = (  # pyright: ignore[reportTypedDictNotRequiredAccess]  # built in place
                            "minimal" if VertexGeminiConfig._supports_minimal_thinking_level(model) else "low"
                        )
            else:
                # Thinking disabled
                params["includeThoughts"] = False
        else:
            # For older Gemini models, use thinkingBudget
            if thinking_enabled and not VertexGeminiConfig._is_thinking_budget_zero(thinking_budget):
                params["includeThoughts"] = sends_thoughts
            if isinstance(thinking_budget, int):
                params["thinkingBudget"] = thinking_budget
            elif thinking_budget is not None:
                warn_dropped_reasoning(
                    "thinking.budget_tokens",
                    thinking_budget,
                    model,
                    _NON_INTEGER_BUDGET_REASON,
                    drop_params_gated=False,
                )

        return params

    def map_response_modalities(self, value: Sequence[object]) -> list[str]:
        response_modalities: Final[list[str]] = []
        for modality in value:
            if modality == "text":
                response_modalities.append("TEXT")
            elif modality == "image":
                response_modalities.append("IMAGE")
            elif modality == "audio":
                response_modalities.append("AUDIO")
            elif modality == "video":
                response_modalities.append("VIDEO")
            else:
                response_modalities.append("MODALITY_UNSPECIFIED")
        return response_modalities

    def validate_parallel_tool_calls(self, value: bool, non_default_params: Mapping[str, object]):
        tools: Final = non_default_params.get("tools", non_default_params.get("functions"))
        num_function_declarations: Final = len(tools) if isinstance(tools, list) else 0
        if num_function_declarations > 1:
            raise litellm.utils.UnsupportedParamsError(
                message=(
                    "`parallel_tool_calls=False` is not supported by Gemini when multiple tools are "
                    "provided. Specify a single tool, or set "
                    "`parallel_tool_calls=True`. If you want to drop this param, set `litellm.drop_params = True` or pass in `(.., drop_params=True)` in the requst - https://docs.litellm.ai/docs/completion/drop_params"
                ),
                status_code=400,
            )

    def _map_audio_params(
        self, value: Mapping[str, object], drop_unknown_fields: bool = False
    ) -> dict[str, object]:  # mutable-ok: the public forwarding interface returns a concrete request dictionary
        """
        Expected input:
        {
            "voice": "alloy",
            "format": "mp3",
            "language_code": "en-US",
        }

        Expected output:
        speechConfig = {
            voiceConfig: {
                prebuiltVoiceConfig: {
                    voiceName: "Kore",
                }
            },
            languageCode: "en-US",
        }
        """
        audio_format_error: Final = unsupported_gemini_tts_audio_format(
            "Gemini TTS", value, drop_unknown_fields, self.custom_llm_provider
        )
        if audio_format_error is not None:
            raise audio_format_error
        camel_speech_config: Final = value.get("speechConfig")
        snake_speech_config: Final = value.get("speech_config")
        speech_config_value: Final = (
            camel_speech_config if isinstance(camel_speech_config, Mapping) else snake_speech_config
        )
        language_code: Final = value.get("language_code", value.get("languageCode"))
        if isinstance(speech_config_value, Mapping):
            return normalize_gemini_speech_config(value, drop_unknown_fields)

        voice: Final = value.get("voice")
        speech_config_without_language: Final[  # mutable-ok: provider request serialization requires a concrete dict
            dict[str, object]
        ] = (
            {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": OPENAI_TO_GEMINI_TTS_VOICES.get(voice.lower(), voice)
                        if isinstance(voice, str)
                        else voice,
                    }
                }
            }
            if "voice" in value
            else {}
        )

        if language_code is not None:
            return {
                **speech_config_without_language,
                "languageCode": language_code,
            }

        return speech_config_without_language

    def _map_chat_audio_params(
        self, model: str, value: Mapping[str, object], drop_params: bool
    ) -> dict[str, object]:  # mutable-ok: the public forwarding interface returns a concrete request dictionary
        audio_format_error: Final = unsupported_gemini_tts_audio_format(
            model, value, drop_params, self.custom_llm_provider
        )
        if audio_format_error is not None:
            raise audio_format_error
        if _gemini_speech_config_wrapper(value) is not None:
            reject_unknown_gemini_tts_voice_keys(model, value, drop_params, self.custom_llm_provider)
        return self._map_audio_params(value, drop_unknown_fields=drops_unsupported_params(drop_params))

    def map_audio_params(
        self,
        value: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        return self._map_audio_params(value)

    @staticmethod
    def _apply_include_server_side_tool_invocations(
        non_default_params: Mapping[str, object],
        optional_params: dict[str, object],
    ) -> None:
        """
        Set include_server_side_tool_invocations before tools are mapped.

        map_openai_params iterates non_default_params in request order; if tools
        appear before this flag, _resolve_search_tool_conflict would drop search
        tools before the flag is applied.
        """
        for key in _SERVER_SIDE_TOOL_INVOCATIONS_KEYS:
            if non_default_params.get(key) is True or optional_params.get(key) is True:
                optional_params["include_server_side_tool_invocations"] = True
                return

    def _drops_logprobs(self, model: str, drop_params: bool, allowed_openai_params: Sequence[str]) -> bool:
        if not drops_unsupported_params(drop_params) or {"logprobs", "top_logprobs"} & set(allowed_openai_params):
            return False
        rows: Final = cast(Mapping[str, Mapping[str, object]], litellm.model_cost)  # cast-ok: JSON rows by model id
        return rows.get(f"{self._route}/{model.split('/')[-1]}", {}).get("supports_logprobs") is False

    def _without_undeclared_thinking_level(
        self,
        thinking_config: GeminiThinkingConfig,
        *,
        model: str,
        drop_params: bool,
        allowed_openai_params: Sequence[str],
    ) -> GeminiThinkingConfig:
        level: Final = thinking_config.get("thinkingLevel")
        if not self._removes_undeclared_thinking_levels or level is None:
            return thinking_config
        if not drops_unsupported_params(drop_params) or "reasoning_effort" in allowed_openai_params:
            return thinking_config
        declared: Final = declared_reasoning_efforts_for_model(model.split("/")[-1], self._route)
        if declared is None or level in declared:
            return thinking_config
        warn_dropped_reasoning(
            "reasoning_effort",
            level,
            model,
            "Google AI Studio does not accept this thinking level for it, so Google's default applies.",
            drop_params_gated=True,
        )
        return {"includeThoughts": VertexGeminiConfig._include_thoughts(thinking_config.get("includeThoughts"))}

    def hosted_mcp_tool_handling(self, model: str) -> HostedMcpToolHandling:
        return "drops"

    def _tools_builder(
        self,
        param: str,
        request_params: Mapping[str, object],
        optional_params: dict[str, object],  # mutable-ok: _map_function adds toolConfig to the request's params
        allow_parameters_json_schema: bool,
        *,
        keep_search_tools_with_functions: bool,
    ) -> Callable[[bool, bool], Sequence[Tools]]:
        return partial(
            self._build_function_tools,
            cast(list[dict], request_params[param]),  # cast-ok: the caller checked it is a non-empty tools list
            optional_params,
            allow_parameters_json_schema=allow_parameters_json_schema,
            outside_declarations=cast(  # cast-ok: tools are JSON request data
                JsonValue, request_params.get("functions" if param == "tools" else "tools")
            ),
            tool_choice=_TOOL_CHOICE_PARAM.validate_python(request_params).get("tool_choice"),
            keep_search_tools_with_functions=keep_search_tools_with_functions,
        )

    def _schema_pins_fit(
        self,
        request_params: Mapping[str, object],
        model: str,
        allow_integer_enums: bool,
        allow_parameters_json_schema: bool,
        keep_search_tools_with_functions: bool,
    ) -> bool:
        tool_params: Final = tuple(
            param
            for param in ("tools", "functions")
            if isinstance(listed := request_params.get(param), list) and listed
        )
        response_format: Final = request_params.get("response_format")
        schema_parts: Final = [request_params.get(param) for param in (*tool_params, "response_format")]
        if not tool_params or not isinstance(response_format, dict):
            return True
        if not has_const(cast(JsonValue, schema_parts)):  # cast-ok: tools and response formats are JSON request data
            return True
        response_params: Final[dict[str, object]] = {}  # mutable-ok: the transformation writes the schema into it
        self.apply_response_schema_transformation(response_format, response_params, model, allow_integer_enums)
        tools: Final = [
            with_enums_that_fit(
                self._tools_builder(
                    param,
                    request_params,
                    {},
                    allow_parameters_json_schema,
                    keep_search_tools_with_functions=keep_search_tools_with_functions,
                ),
                allow_integer_enums and self._is_gemini_3_or_newer(model),
                has_const(cast(JsonValue, request_params[param])),  # cast-ok: tools are JSON request data
            )
            for param in tool_params
        ]
        return enums_fit(cast(JsonValue, [response_params, tools]))  # cast-ok: built tools and schemas are JSON

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        allow_integer_enums: bool = True,
        allow_parameters_json_schema: bool = True,
        *,
        allowed_openai_params: Sequence[str] = (),
        allow_unsupported_n: bool = False,
    ) -> dict:
        request_params: Final = tool_params_without_hosted_mcp(
            non_default_params,
            model=model,
            route=self._route,
            drop_params=drop_params,
            handling=self.hosted_mcp_tool_handling(model),
        )
        self._apply_include_server_side_tool_invocations(request_params, optional_params)
        keeps_search_tools_with_functions: Final = self._keeps_search_tools_with_functions(
            model,
            cast(Mapping[str, object], request_params),  # cast-ok: map_openai_params keys request params by name
            cast(Mapping[str, object], optional_params),  # cast-ok: map_openai_params keys request params by name
        )
        supports_sampling_params: Final = self.supports_sampling_params(model)
        drops_logprobs: Final = self._drops_logprobs(model, drop_params, allowed_openai_params)
        typed_request_params: Final = _REQUEST_PARAMS.validate_python(request_params)
        candidate_params: Final = _CANDIDATE_COUNT_PARAMS.validate_python(typed_request_params)
        candidate_count: Final = gemini_image_candidate_count(
            model, candidate_params, drop_params, allowed_openai_params, self._route
        )
        if isinstance(candidate_count, UnsupportedParamsError) and not allow_unsupported_n:
            raise candidate_count
        pins_fit: Final = self._schema_pins_fit(
            typed_request_params,
            model,
            allow_integer_enums,
            allow_parameters_json_schema,
            keeps_search_tools_with_functions,
        )
        for param, value in request_params.items():
            if param in ("temperature", "top_p", "top_k") and not supports_sampling_params:
                continue
            if param in ("logprobs", "top_logprobs") and drops_logprobs:
                continue
            if param == "temperature":
                if VertexGeminiConfig._is_gemini_3_or_newer(model):
                    if value is not None and value < 1.0:
                        verbose_logger.info(
                            "Warning: Setting temperature < 1.0 for Gemini 3 models (%s) can cause infinite loops, degraded reasoning performance, and failure on complex tasks. Strongly recommended to use temperature = 1.0 (default).",
                            model,
                        )
                    _warn_gemini_sampling_params_deprecated(model)
                optional_params["temperature"] = value
            elif param == "top_p":
                if VertexGeminiConfig._is_gemini_3_or_newer(model):
                    _warn_gemini_sampling_params_deprecated(model)
                optional_params["top_p"] = value
            elif param == "top_k":
                if VertexGeminiConfig._is_gemini_3_or_newer(model):
                    _warn_gemini_sampling_params_deprecated(model)
                optional_params["top_k"] = value
            elif (
                param == "stream" and value is True
            ):  # sending stream = False, can cause it to get passed unchecked and raise issues
                optional_params["stream"] = value
            elif param == "n" and candidate_count == "drop":
                verbose_logger.warning(
                    "Dropping n=%s for %s because Gemini returns one candidate per request when modalities asks "
                    "for image output",
                    candidate_params.get("n"),
                    model,
                )
            elif param == "n":
                optional_params["candidate_count"] = value
            elif param == "audio" and isinstance(value, dict):
                optional_params["speechConfig"] = self._map_chat_audio_params(model, value, drop_params)
            elif param == "stop":
                if isinstance(value, str):
                    optional_params["stop_sequences"] = [value]
                elif isinstance(value, list):
                    optional_params["stop_sequences"] = value
            elif param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_output_tokens"] = value
            elif param == "response_format" and isinstance(value, dict):
                self.apply_response_schema_transformation(
                    value=value,
                    optional_params=optional_params,
                    model=model,
                    allow_integer_enums=allow_integer_enums,
                    pin_consts=pins_fit,
                )
            elif param == "frequency_penalty":
                if self._supports_penalty_parameters(model):
                    optional_params["frequency_penalty"] = value
            elif param == "presence_penalty":
                if self._supports_penalty_parameters(model):
                    optional_params["presence_penalty"] = value
            elif param == "logprobs":
                optional_params["responseLogprobs"] = value
            elif param == "top_logprobs":
                optional_params["logprobs"] = value
            elif (param == "tools" or param == "functions") and isinstance(value, list) and value:
                # Pass optional_params so _map_function can add toolConfig if needed
                mapped_tools = with_enums_that_fit(
                    self._tools_builder(
                        param,
                        typed_request_params,
                        optional_params,
                        allow_parameters_json_schema,
                        keep_search_tools_with_functions=keeps_search_tools_with_functions,
                    ),
                    allow_integer_enums and self._is_gemini_3_or_newer(model),
                    pins_fit and has_const(cast(JsonValue, value)),  # cast-ok: tools are JSON request data
                )
                optional_params = self._add_tools_to_optional_params(optional_params, list(mapped_tools))
            elif param == "tool_choice" and (isinstance(value, str) or isinstance(value, dict)):
                _tool_choice_value = self.map_tool_choice_values(
                    model=model,
                    tool_choice=value,
                    tools=[*_listed(non_default_params.get("tools")), *_listed(non_default_params.get("functions"))],
                    drop_params=drop_params,
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value
            elif param == "parallel_tool_calls":
                tools_list = request_params.get("tools", request_params.get("functions"))
                num_tools = len(tools_list) if isinstance(tools_list, list) else 0
                # Gemini does not support parallel_tool_calls=False with multiple
                # tools. Drop the param instead of failing — Responses API clients
                # often send parallel_tool_calls=false by default.
                if not (value is False and num_tools > 1):
                    optional_params["parallel_tool_calls"] = value
            elif param == "seed":
                optional_params["seed"] = value
            elif param == "reasoning_effort":
                # Extract effort value - handle both string and dict formats
                # Dict format comes from OpenAI Agents SDK: {"effort": "high", "summary": "auto"}
                effort_value: str | None = None
                if isinstance(value, str):
                    effort_value = value
                elif isinstance(value, dict):
                    effort_value = value.get("effort")

                if effort_value is not None:
                    # Validate no conflict with thinking_level
                    VertexGeminiConfig._validate_thinking_config_conflicts(
                        optional_params=optional_params,
                        param_name="reasoning_effort",
                        param_description="thinking_budget",
                    )
                    optional_params["thinkingConfig"] = self._without_undeclared_thinking_level(
                        VertexGeminiConfig._map_reasoning_effort(
                            effort=effort_value, model=model, reasoning_effort=value
                        ),
                        model=model,
                        drop_params=drop_params,
                        allowed_openai_params=allowed_openai_params,
                    )
            elif param == "thinking":
                # Validate no conflict with thinking_level
                VertexGeminiConfig._validate_thinking_config_conflicts(
                    optional_params=optional_params,
                    param_name="thinking",
                    param_description="thinking_budget",
                )
                optional_params["thinkingConfig"] = VertexGeminiConfig._map_thinking_param(
                    cast(AnthropicThinkingParam, value),
                    model=model,
                )
            elif param == "modalities" and isinstance(value, list):
                response_modalities = self.map_response_modalities(value)
                optional_params["responseModalities"] = response_modalities
            elif param == "web_search_options" and isinstance(value, dict):
                _tools = self._map_web_search_options(value)
                optional_params = self._add_tools_to_optional_params(optional_params, [_tools])
            elif param == "service_tier" and isinstance(value, str):
                self._map_service_tier_param(value, optional_params, drop_params)
            elif param == "include_server_side_tool_invocations" and value is True:
                optional_params["include_server_side_tool_invocations"] = True
        if litellm.vertex_ai_safety_settings is not None:
            optional_params["safety_settings"] = litellm.vertex_ai_safety_settings

        # if audio param is set, ensure responseModalities is set to AUDIO
        audio_param: Final = optional_params.get("speechConfig")
        if audio_param is not None:
            if "responseModalities" not in optional_params:
                optional_params["responseModalities"] = ["AUDIO"]
            elif "AUDIO" not in optional_params["responseModalities"]:
                optional_params["responseModalities"].append("AUDIO")

        self._drop_search_tools_mixed_with_functions(
            cast(dict[str, object], optional_params),  # cast-ok: map_openai_params keys request params by name
            keeps_search_tools_with_functions,
            drops_search_retrieval=drops_unsupported_params(drop_params),
        )

        return optional_params

    def get_mapped_special_auth_params(self) -> dict[str, str]:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: Mapping[str, object], optional_params: dict[str, object]):
        mapped_params: Final = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> list[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    @staticmethod
    def get_model_for_vertex_ai_url(model: str) -> str:
        """
        Returns the model name to use in the request to Vertex AI

        Handles 2 cases:
        1. User passed `model="vertex_ai/gemini/ft-uuid"`, we need to return `ft-uuid` for the request to Vertex AI
        2. User passed `model="vertex_ai/gemini-2.0-flash-001"`, we need to return `gemini-2.0-flash-001` for the request to Vertex AI

        Args:
            model (str): The model name to use in the request to Vertex AI

        Returns:
            str: The model name to use in the request to Vertex AI
        """
        if VertexGeminiConfig._is_model_gemini_spec_model(model):
            return VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
        return model

    @staticmethod
    def _is_model_gemini_spec_model(model: str | None) -> bool:
        """
        Returns true if user is trying to call custom model in `/gemini` request/response format
        """
        if model is None:
            return False
        if "gemini/" in model:
            return True
        return False

    @classmethod
    def is_model_gemini_spec_model(
        cls,
        model: str | None,
    ) -> bool:
        return cls._is_model_gemini_spec_model(model)

    @staticmethod
    def _get_model_name_from_gemini_spec_model(model: str) -> str:
        """
        Returns the model name if model="vertex_ai/gemini/<unique_id>"

        Example:
        - model = "gemini/1234567890"
        - returns "1234567890"
        """
        if "gemini/" in model:
            return model.split("/")[-1]
        return model

    def get_flagged_finish_reasons(self) -> dict[str, str]:
        """
        Return Dictionary of finish reasons which indicate response was flagged

        and what it means
        """
        return {
            "SAFETY": "The token generation was stopped as the response was flagged for safety reasons. NOTE: When streaming the Candidate.content will be empty if content filters blocked the output.",
            "RECITATION": "The token generation was stopped as the response was flagged for unauthorized citations.",
            "BLOCKLIST": "The token generation was stopped as the response was flagged for the terms which are included from the terminology blocklist.",
            "PROHIBITED_CONTENT": "The token generation was stopped as the response was flagged for the prohibited contents.",
            "SPII": "The token generation was stopped as the response was flagged for Sensitive Personally Identifiable Information (SPII) contents.",
            "IMAGE_SAFETY": "The token generation was stopped as the response was flagged for image safety reasons.",
            "IMAGE_PROHIBITED_CONTENT": "The token generation was stopped as the response was flagged for prohibited image content.",
        }

    _GEMINI_FINISH_REASON_KEYS: Final[frozenset[str]] = frozenset(get_args(GeminiFinishReason))

    @staticmethod
    def get_finish_reason_mapping() -> dict[str, OpenAIChatCompletionFinishReason]:
        """
        Return Dictionary of Gemini/Vertex AI finish reasons and their
        OpenAI-compatible mappings.
        """
        from litellm.litellm_core_utils.core_helpers import _FINISH_REASON_MAP

        return {k: v for k, v in _FINISH_REASON_MAP.items() if k in VertexGeminiConfig._GEMINI_FINISH_REASON_KEYS}

    def translate_exception_str(self, exception_string: str):
        if (
            "GenerateContentRequest.tools[0].function_declarations[0].parameters.properties: should be non-empty for OBJECT type"
            in exception_string
        ):
            return f"'properties' field in tools[0]['function']['parameters'] cannot be empty if 'type' == 'object'. Received error from provider - {exception_string}"
        return exception_string

    def get_assistant_content_message(self, parts: list[HttpxPartType]) -> tuple[str | None, str | None]:
        content_str: str | None = None
        reasoning_content_str: str | None = None

        for part in parts:
            _content_str = ""
            if "text" in part:
                text_content = part["text"]
                # Check if text content is audio data URI - if so, exclude from text content
                if text_content.startswith("data:audio") and ";base64," in text_content:
                    try:
                        if is_base64_encoded(text_content):
                            media_type, _ = text_content.split("data:")[1].split(";base64,")
                            if media_type.startswith("audio/"):
                                continue
                    except (ValueError, IndexError):
                        # If parsing fails, treat as regular text
                        pass
                _content_str += text_content
            elif "inlineData" in part:
                inline_data = part.get("inlineData", {})
                mime_type = inline_data.get("mimeType", "")
                data = inline_data.get("data", "")
                # Check if inline data is audio or image - if so, exclude from text content
                # Images and audio are now handled separately in their respective response fields
                if mime_type.startswith("audio/") or mime_type.startswith("image/"):
                    continue
                _content_str += f"data:{mime_type};base64,{data}"

            if len(_content_str) > 0:
                if part.get("thought") is True:
                    if reasoning_content_str is None:
                        reasoning_content_str = ""
                    reasoning_content_str += _content_str
                else:
                    if content_str is None:
                        content_str = ""
                    content_str += _content_str

        return content_str, reasoning_content_str

    def _extract_thinking_blocks_from_parts(self, parts: list[HttpxPartType]) -> list[ChatCompletionThinkingBlock]:
        """Extract thinking blocks from parts if present.

        Per Google's docs (https://ai.google.dev/gemini-api/docs/thinking):
        - Parts with `thought: true` contain thinking/reasoning content
        - `thoughtSignature` is a separate token for multi-turn context preservation,
          it does NOT indicate that the content is thinking (a part can have
          thoughtSignature without thought: true, e.g., function calls)
        """
        thinking_blocks: Final[list[ChatCompletionThinkingBlock]] = []
        for part in parts:
            if part.get("thought") is True and "inlineData" not in part:
                thinking_text = part.get("text", "")
                block: ChatCompletionThinkingBlock = {
                    "type": "thinking",
                    "thinking": thinking_text,
                }
                signature = part.get("thoughtSignature")
                if signature is not None:
                    block["signature"] = signature
                thinking_blocks.append(block)
        return thinking_blocks

    def _extract_thought_signatures_from_parts(self, parts: list[HttpxPartType]) -> list[str] | None:
        """Extract thoughtSignature values from parts.

        Per Google's docs, thoughtSignature is returned for multi-turn context preservation
        and can appear on parts even without thought: true (e.g., regular text responses,
        function calls). This method extracts all thoughtSignature values from parts.

        Returns:
            List of thoughtSignature strings if any are found, None otherwise
        """
        signatures: Final[list[str]] = []
        for part in parts:
            signature = part.get("thoughtSignature")
            if signature is not None:
                signatures.append(signature)
        return signatures if signatures else None

    @staticmethod
    def _extract_server_side_tool_invocations(
        parts: list[HttpxPartType],
    ) -> list[dict[str, object]] | None:
        """Extract server-side tool invocations (toolCall/toolResponse) from parts.

        These are returned by Gemini when context circulation is enabled
        (includeServerSideToolInvocations=true). They represent tools executed
        server-side (e.g. Google Search) and must be circulated back in
        subsequent turns for multi-turn coherence.

        Returns:
            List of server-side invocation dicts if any found, None otherwise.
        """
        invocations: Final[list[dict[str, object]]] = []
        # Index toolCalls by id so we can pair them with responses
        tool_calls_by_id: Final[dict[str, dict[str, object]]] = {}
        tool_responses_by_id: Final[dict[str, dict[str, object]]] = {}

        for part in parts:
            if "toolCall" in part:
                tc = part["toolCall"]
                entry: dict[str, object] = {
                    "tool_type": tc.get("toolType"),
                    "id": tc.get("id"),
                    "args": tc.get("args"),
                }
                signature = part.get("thoughtSignature")
                if signature is not None:
                    entry["thought_signature"] = signature
                tool_calls_by_id[tc.get("id", "")] = entry

            elif "toolResponse" in part:
                tr = part["toolResponse"]
                entry = {
                    "id": tr.get("id"),
                    "tool_type": tr.get("toolType"),
                    "response": tr.get("response"),
                }
                signature = part.get("thoughtSignature")
                if signature is not None:
                    entry["thought_signature"] = signature
                tool_responses_by_id[tr.get("id", "")] = entry

        # Merge calls with their responses
        for call_id, call_entry in tool_calls_by_id.items():
            merged = dict(call_entry)
            resp = tool_responses_by_id.pop(call_id, None)
            if resp is not None:
                merged["response"] = resp.get("response")
                if "thought_signature" in resp:
                    merged["response_thought_signature"] = resp["thought_signature"]
            invocations.append(merged)

        # Any orphan responses (shouldn't happen, but be safe)
        for resp_id, resp_entry in tool_responses_by_id.items():
            if "thought_signature" in resp_entry:
                resp_entry["response_thought_signature"] = resp_entry["thought_signature"]
            invocations.append(resp_entry)

        return invocations if invocations else None

    def _extract_audio_response_from_parts(
        self, parts: list[HttpxPartType], keeps_requested_audio_format: bool
    ) -> ChatCompletionAudioResponse | None:
        """Extract audio response from parts if present"""
        for part in parts:
            if "text" in part:
                text_content = part["text"]
                # Check if text content contains audio data URI
                if text_content.startswith("data:audio") and ";base64," in text_content:
                    try:
                        if is_base64_encoded(text_content):
                            media_type, audio_data = text_content.split("data:")[1].split(";base64,")

                            if media_type.startswith("audio/"):
                                expires_at = int(time.time()) + (24 * 60 * 60)
                                transcript = ""  # Gemini doesn't provide transcript

                                return ChatCompletionAudioResponse(
                                    data=audio_data
                                    if keeps_requested_audio_format
                                    else as_raw_pcm16_base64(media_type, audio_data),
                                    expires_at=expires_at,
                                    transcript=transcript,
                                )
                    except (ValueError, IndexError):
                        pass

            elif "inlineData" in part:
                inline_data = part.get("inlineData", {})
                mime_type = inline_data.get("mimeType", "")
                data = inline_data.get("data", "")

                if mime_type.startswith("audio/"):
                    expires_at = int(time.time()) + (24 * 60 * 60)
                    transcript = ""  # Gemini doesn't provide transcript

                    return ChatCompletionAudioResponse(
                        data=data if keeps_requested_audio_format else as_raw_pcm16_base64(mime_type, data),
                        expires_at=expires_at,
                        transcript=transcript,
                    )

        return None

    @staticmethod
    def _transform_parts(
        parts: list[HttpxPartType],
        cumulative_tool_call_idx: int,
        is_function_call: bool | None,
        thought_signature_in_tool_call_id: bool = True,
    ) -> tuple[
        ChatCompletionToolCallFunctionChunk | None,
        list[ChatCompletionToolCallChunk] | None,
        int,
    ]:
        function: ChatCompletionToolCallFunctionChunk | None = None
        _tools: Final[list[ChatCompletionToolCallChunk]] = []
        for part in parts:
            if "functionCall" in part:
                _function_chunk: ChatCompletionToolCallFunctionChunk = {
                    "name": part["functionCall"]["name"],
                    "arguments": json.dumps(part["functionCall"]["args"], ensure_ascii=False),
                }
                # Extract thought signature if present
                thought_signature = part.get("thoughtSignature")
                # Gemini 3.5+ returns a stable `id` per function call to enable
                # strict response matching. Preserve it as the OpenAI
                # tool_call_id so it can be echoed back unchanged.
                gemini_call_id = part["functionCall"].get("id")

                if is_function_call is True:
                    function = (
                        {**_function_chunk, "provider_specific_fields": {"thought_signature": thought_signature}}
                        if thought_signature
                        else {**_function_chunk}
                    )
                else:
                    _tool_response_chunk: ChatCompletionToolCallChunk = {
                        "id": f"call_{uuid.uuid4().hex[:28]}",
                        "type": "function",
                        "function": _function_chunk,
                        "index": cumulative_tool_call_idx,
                    }
                    # Gemini 3.5+ returns a stable native `id`; prefer it over
                    # the synthetic call_<uuid> so the same value can be echoed
                    # back on the matching `functionResponse`.
                    if gemini_call_id:
                        _tool_response_chunk["id"] = gemini_call_id
                    if thought_signature:
                        _tool_response_chunk["provider_specific_fields"] = {"thought_signature": thought_signature}
                    if thought_signature and thought_signature_in_tool_call_id:
                        _tool_response_chunk["id"] = encode_tool_call_id_with_signature(
                            _tool_response_chunk["id"] or "", thought_signature
                        )
                    _tools.append(_tool_response_chunk)
                cumulative_tool_call_idx += 1
        if len(_tools) == 0:
            tools: list[ChatCompletionToolCallChunk] | None = None
        else:
            tools = _tools
        return function, tools, cumulative_tool_call_idx

    @staticmethod
    def _transform_logprobs(
        logprobs_result: LogprobsResult | None,
    ) -> ChoiceLogprobs | None:
        if logprobs_result is None:
            return None
        if "chosenCandidates" not in logprobs_result:
            return None
        logprobs_list: Final[list[ChatCompletionTokenLogprob]] = []
        for index, candidate in enumerate(logprobs_result["chosenCandidates"]):
            top_logprobs: list[TopLogprob] = []
            if "topCandidates" in logprobs_result and index < len(logprobs_result["topCandidates"]):
                top_candidates_for_index = logprobs_result["topCandidates"][index]["candidates"]

                for options in top_candidates_for_index:
                    top_logprobs.append(TopLogprob(token=options["token"], logprob=options["logProbability"]))
            logprobs_list.append(
                ChatCompletionTokenLogprob(
                    token=candidate["token"],
                    logprob=candidate["logProbability"],
                    top_logprobs=top_logprobs,
                )
            )
        return ChoiceLogprobs(content=logprobs_list)

    def _content_filter_response(
        self,
        model_response: ModelResponse,
        completion_response: GenerateContentResponseBody,
        provider_specific_fields: Mapping[str, str],
    ) -> ModelResponse:
        chat_completion_message: Final[ChatCompletionResponseMessage] = {
            "role": "assistant",
            "content": None,
        }

        choice: Final = litellm.Choices(
            finish_reason="content_filter",
            index=0,
            message=chat_completion_message,
            logprobs=None,
            enhancements=None,
            provider_specific_fields=dict(provider_specific_fields),
        )

        model_response.choices = [choice]

        usage: Final = VertexGeminiConfig._calculate_usage(completion_response=completion_response)

        setattr(model_response, "usage", usage)

        return model_response

    @staticmethod
    def is_candidate_token_count_inclusive(usage_metadata: Mapping[str, object]) -> bool:
        """
        Check if the candidate token count is inclusive of the thinking token count

        if promptTokenCount + candidatesTokenCount + toolUsePromptTokenCount == totalTokenCount, then the candidate token count is inclusive of the thinking token count

        else the candidate token count is exclusive of the thinking token count

        Addresses - https://github.com/BerriAI/litellm/pull/10141#discussion_r2052272035
        """
        prompt_tokens: Final = usage_metadata.get("promptTokenCount", 0)
        candidate_tokens: Final = usage_metadata.get("candidatesTokenCount", 0)
        tool_use_prompt_tokens: Final = usage_metadata.get("toolUsePromptTokenCount", 0)
        total_tokens: Final = usage_metadata.get("totalTokenCount", 0)
        return (
            isinstance(prompt_tokens, int)
            and isinstance(candidate_tokens, int)
            and isinstance(tool_use_prompt_tokens, int)
            and isinstance(total_tokens, int)
            and prompt_tokens + candidate_tokens + tool_use_prompt_tokens == total_tokens
        )

    @staticmethod
    def _response_has_search_grounding(
        completion_response: GenerateContentResponseBody | BidiGenerateContentServerMessage,
    ) -> bool:
        """
        Whether the response used Grounding with Google Search or Grounding with Google Maps,
        detected via groundingMetadata.webSearchQueries (an actual web search was performed) or
        groundingMetadata.groundingChunks[].maps (a Maps lookup was performed).

        Google bills both groundings separately (a per-request / per-query fee) and excludes their
        retrieved tokens from input token billing, unlike URL context / File Search / code execution
        whose tool-use tokens are charged at the input token rate. URL context also emits
        groundingMetadata (with web groundingChunks but no webSearchQueries), so presence of
        groundingMetadata alone is not a sufficient signal.
        See https://ai.google.dev/gemini-api/docs/pricing and
        https://github.com/BerriAI/litellm/discussions/33198
        """
        if "candidates" not in completion_response:
            return False
        for candidate in completion_response["candidates"] or []:
            grounding_metadata, _, _, _ = VertexGeminiConfig._extract_candidate_metadata(candidate)
            if calculate_grounding_requests(grounding_metadata).has_billable_grounding():
                return True
        return False

    @staticmethod
    def _calculate_usage(
        completion_response: GenerateContentResponseBody | BidiGenerateContentServerMessage,
        cache_creation_tokens: int = 0,
    ) -> Usage:
        if completion_response is not None and "usageMetadata" not in completion_response:
            raise ValueError(f"usageMetadata not found in completion_response. Got={completion_response}")
        cached_tokens: int | None = None
        # Separate variables for prompt tokens by modality
        prompt_audio_tokens: int | None = None
        prompt_image_tokens: int | None = None
        prompt_text_tokens: int | None = None
        prompt_video_tokens: int | None = None
        prompt_tokens_details: PromptTokensDetailsWrapper | None = None
        reasoning_tokens: int | None = None
        response_tokens: int | None = None
        response_tokens_details: CompletionTokensDetailsWrapper | None = None
        usage_metadata: Final = completion_response["usageMetadata"]

        def _get_token_count(detail: Mapping[str, object]) -> int:
            raw_token_count: Final = detail.get("tokenCount", detail.get("token_count", 0))
            return raw_token_count if isinstance(raw_token_count, int) else 0

        if "cachedContentTokenCount" in usage_metadata:
            cached_tokens = usage_metadata["cachedContentTokenCount"]

        ## GEMINI LIVE API ONLY PARAMS ##
        if "responseTokenCount" in usage_metadata:
            response_tokens = usage_metadata["responseTokenCount"]
        if "responseTokensDetails" in usage_metadata:
            response_tokens_details = CompletionTokensDetailsWrapper()
            for detail in usage_metadata["responseTokensDetails"]:
                modality = str(detail.get("modality", "")).upper()
                token_count = _get_token_count(detail)
                if modality == "TEXT":
                    response_tokens_details.text_tokens = (response_tokens_details.text_tokens or 0) + token_count
                elif modality == "AUDIO":
                    response_tokens_details.audio_tokens = (response_tokens_details.audio_tokens or 0) + token_count
                elif modality == "DOCUMENT":
                    response_tokens_details.text_tokens = (response_tokens_details.text_tokens or 0) + token_count

        #########################################################

        ## CANDIDATES TOKEN DETAILS (e.g., for image generation models) ##
        if "candidatesTokensDetails" in usage_metadata:
            if response_tokens_details is None:
                response_tokens_details = CompletionTokensDetailsWrapper()
            for detail in usage_metadata["candidatesTokensDetails"]:
                modality = str(detail.get("modality", "")).upper()
                token_count = _get_token_count(detail)
                if modality == "TEXT":
                    response_tokens_details.text_tokens = (response_tokens_details.text_tokens or 0) + token_count
                elif modality == "AUDIO":
                    response_tokens_details.audio_tokens = (response_tokens_details.audio_tokens or 0) + token_count
                elif modality == "IMAGE":
                    response_tokens_details.image_tokens = (response_tokens_details.image_tokens or 0) + token_count
                elif modality == "VIDEO":
                    response_tokens_details.video_tokens = (response_tokens_details.video_tokens or 0) + token_count
                elif modality == "DOCUMENT":
                    response_tokens_details.text_tokens = (response_tokens_details.text_tokens or 0) + token_count

        # Calculate text_tokens if not explicitly provided in candidatesTokensDetails
        # candidatesTokenCount includes all modalities, so: text = total - (image + audio + video)
        candidates_token_count: Final = usage_metadata.get("candidatesTokenCount", 0)
        if candidates_token_count > 0:
            if response_tokens_details is None:
                response_tokens_details = CompletionTokensDetailsWrapper()
            if response_tokens_details.text_tokens is None:
                completion_image_tokens: Final = response_tokens_details.image_tokens or 0
                completion_audio_tokens: Final = response_tokens_details.audio_tokens or 0
                completion_video_tokens: Final = response_tokens_details.video_tokens or 0
                calculated_text_tokens: Final = (
                    candidates_token_count - completion_image_tokens - completion_audio_tokens - completion_video_tokens
                )
                response_tokens_details.text_tokens = calculated_text_tokens
        #########################################################

        ## Parse promptTokensDetails (total tokens by modality, includes cached + non-cached)
        if "promptTokensDetails" in usage_metadata:
            for detail in usage_metadata["promptTokensDetails"]:
                modality = str(detail.get("modality", "")).upper()
                token_count = _get_token_count(detail)
                if modality == "AUDIO":
                    prompt_audio_tokens = (prompt_audio_tokens or 0) + token_count
                elif modality == "TEXT":
                    prompt_text_tokens = (prompt_text_tokens or 0) + token_count
                elif modality == "IMAGE":
                    prompt_image_tokens = (prompt_image_tokens or 0) + token_count
                elif modality == "VIDEO":
                    prompt_video_tokens = (prompt_video_tokens or 0) + token_count
                elif modality == "DOCUMENT":
                    prompt_text_tokens = (prompt_text_tokens or 0) + token_count

        ## Parse cacheTokensDetails (breakdown of cached tokens by modality)
        ## When explicit caching is used, Gemini provides this field to show which modalities were cached
        cached_text_tokens: int | None = None
        cached_audio_tokens: int | None = None
        cached_image_tokens: int | None = None
        cached_video_tokens: int | None = None

        if "cacheTokensDetails" in usage_metadata:
            for detail in usage_metadata["cacheTokensDetails"]:
                modality = str(detail.get("modality", "")).upper()
                token_count = _get_token_count(detail)
                if modality == "AUDIO":
                    cached_audio_tokens = (cached_audio_tokens or 0) + token_count
                elif modality == "TEXT":
                    cached_text_tokens = (cached_text_tokens or 0) + token_count
                elif modality == "IMAGE":
                    cached_image_tokens = (cached_image_tokens or 0) + token_count
                elif modality == "VIDEO":
                    cached_video_tokens = (cached_video_tokens or 0) + token_count
                elif modality == "DOCUMENT":
                    cached_text_tokens = (cached_text_tokens or 0) + token_count

        if (
            cached_tokens is not None
            and prompt_text_tokens is not None
            and cached_text_tokens is None
            and "cacheTokensDetails" not in usage_metadata
        ):
            prompt_text_tokens = prompt_text_tokens - cached_tokens
        if cached_video_tokens is not None and prompt_video_tokens is not None:
            prompt_video_tokens = prompt_video_tokens - cached_video_tokens

        if "thoughtsTokenCount" in usage_metadata:
            reasoning_tokens = usage_metadata["thoughtsTokenCount"]
            # Also add reasoning tokens to response_tokens_details
            if response_tokens_details is None:
                response_tokens_details = CompletionTokensDetailsWrapper()
            response_tokens_details.reasoning_tokens = reasoning_tokens

        tool_use_prompt_tokens: Final = usage_metadata.get("toolUsePromptTokenCount") or None

        billable_tool_use_prompt_tokens: Final = (
            0
            if VertexGeminiConfig._response_has_search_grounding(completion_response)
            else (tool_use_prompt_tokens or 0)
        )
        billable_text_tokens: Final = (
            prompt_text_tokens if prompt_text_tokens is None else prompt_text_tokens + billable_tool_use_prompt_tokens
        )
        cached_tokens_details_fields: Final = MappingProxyType(
            {
                key: token_count
                for key, token_count in (
                    ("text_tokens", cached_text_tokens),
                    ("audio_tokens", cached_audio_tokens),
                    ("image_tokens", cached_image_tokens),
                )
                if token_count is not None
            }
        )
        cached_tokens_details: Final = (
            CachedTokensDetails.model_validate(cached_tokens_details_fields) if cached_tokens_details_fields else None
        )

        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens,
            cached_tokens_details=cached_tokens_details,
            audio_tokens=prompt_audio_tokens,
            text_tokens=billable_text_tokens,
            image_tokens=prompt_image_tokens,
            video_tokens=prompt_video_tokens,
            tool_use_tokens=tool_use_prompt_tokens,
        )

        completion_tokens = response_tokens or completion_response["usageMetadata"].get("candidatesTokenCount", 0)
        if not VertexGeminiConfig.is_candidate_token_count_inclusive(usage_metadata) and reasoning_tokens:
            completion_tokens = reasoning_tokens + completion_tokens
        ## GET USAGE ##
        usage: Final = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0)
            + billable_tool_use_prompt_tokens
            + cache_creation_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage_metadata.get("totalTokenCount", 0) + cache_creation_tokens,
            prompt_tokens_details=prompt_tokens_details,
            cache_read_input_tokens=cached_tokens,
            cache_creation_input_tokens=cache_creation_tokens or None,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_details=response_tokens_details,
        )

        return usage

    @classmethod
    def calculate_usage(
        cls,
        completion_response: GenerateContentResponseBody | BidiGenerateContentServerMessage,
        cache_creation_tokens: int = 0,
    ) -> Usage:
        return cls._calculate_usage(completion_response, cache_creation_tokens)

    @staticmethod
    def _check_finish_reason(
        chat_completion_message: ChatCompletionResponseMessage | None,
        finish_reason: str | None,
    ) -> OpenAIChatCompletionFinishReason:
        from litellm.litellm_core_utils.core_helpers import map_finish_reason

        if chat_completion_message and chat_completion_message.get("function_call"):
            return "function_call"
        elif chat_completion_message and chat_completion_message.get("tool_calls"):
            return "tool_calls"
        elif finish_reason:
            return map_finish_reason(finish_reason)
        else:
            return "stop"

    @classmethod
    def check_finish_reason(
        cls,
        chat_completion_message: ChatCompletionResponseMessage | None,
        finish_reason: str | None,
    ) -> OpenAIChatCompletionFinishReason:
        return cls._check_finish_reason(chat_completion_message, finish_reason)

    @staticmethod
    def _check_prompt_level_content_filter(
        processed_chunk: GenerateContentResponseBody,
        response_id: str | None,
        model: str | None = None,
    ) -> Optional["ModelResponseStream"]:
        """
        Check if prompt is blocked due to content filtering at the prompt level.

        This handles the case where Vertex AI blocks the prompt before generation begins,
        indicated by promptFeedback.blockReason being present.

        Args:
            processed_chunk: The parsed response chunk from Vertex AI
            response_id: The response ID from the chunk

        Returns:
            ModelResponseStream with content_filter finish_reason if blocked, None otherwise.

        Note:
            This is consistent with non-streaming _content_filter_response() behavior.
            Candidate-level content filtering (SAFETY, RECITATION, etc.) is handled
            separately via _process_candidates() → _check_finish_reason().
        """
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        # Check if prompt is blocked due to content filtering
        prompt_feedback: Final = processed_chunk.get("promptFeedback")
        if prompt_feedback and "blockReason" in prompt_feedback:
            verbose_logger.debug(
                "Prompt blocked due to: %s - %s",
                prompt_feedback.get("blockReason"),
                prompt_feedback.get("blockReasonMessage"),
            )

            # Create a content_filter response (consistent with non-streaming _content_filter_response)
            choice: Final = StreamingChoices(
                finish_reason="content_filter",
                index=0,
                delta=Delta(content=None, role="assistant"),
                logprobs=None,
                enhancements=None,
                provider_specific_fields=dict(_prompt_block_fields(prompt_feedback)),
            )

            model_response: Final = ModelResponseStream(choices=[choice], id=response_id, model=model)
            return model_response

        return None

    @classmethod
    def check_prompt_level_content_filter(
        cls,
        processed_chunk: GenerateContentResponseBody,
        response_id: str | None,
        model: str | None = None,
    ) -> Optional["ModelResponseStream"]:
        return cls._check_prompt_level_content_filter(processed_chunk, response_id, model)

    @staticmethod
    def _calculate_web_search_requests(grounding_metadata: list[dict[str, object]]) -> int | None:
        return calculate_grounding_requests(grounding_metadata).web_search_requests

    @classmethod
    def calculate_web_search_requests(
        cls,
        grounding_metadata: list[dict[str, object]],  # mutable-ok: mirrors override contract
    ) -> int | None:
        return cls._calculate_web_search_requests(grounding_metadata)

    @staticmethod
    def _set_grounding_usage_counters(
        usage: Usage,
        grounding_metadata: Sequence[Mapping[str, object]],
        tool_call_queries: Sequence[str] = (),
    ) -> None:
        grounding_requests: Final = calculate_grounding_requests(grounding_metadata, tool_call_queries)
        details: Final = cast(PromptTokensDetailsWrapper, usage.prompt_tokens_details)
        if grounding_requests.web_search_requests is not None:
            details.web_search_requests = grounding_requests.web_search_requests
        if grounding_requests.google_maps_grounding_requests is not None:
            details.google_maps_grounding_requests = grounding_requests.google_maps_grounding_requests

    @classmethod
    def set_grounding_usage_counters(
        cls,
        usage: Usage,
        grounding_metadata: Sequence[Mapping[str, object]],
        tool_call_queries: Sequence[str] = (),
    ) -> None:
        return cls._set_grounding_usage_counters(usage, grounding_metadata, tool_call_queries)

    @staticmethod
    def _create_streaming_choice(
        chat_completion_message: ChatCompletionResponseMessage,
        candidate: Candidates,
        idx: int,
        tools: list[ChatCompletionToolCallChunk] | None,
        functions: ChatCompletionToolCallFunctionChunk | None,
        chat_completion_logprobs: ChoiceLogprobs | None,
        image_response: list[ImageURLListItem] | None,
        audio_response: ChatCompletionAudioResponse | None = None,
    ) -> StreamingChoices:
        """
        Helper method to create a streaming choice object for Vertex AI
        """
        from litellm.types.utils import Delta, StreamingChoices

        annotations: Final = chat_completion_message.get("annotations")
        provider_specific_fields: Final = chat_completion_message.get("provider_specific_fields")
        finish_fields: Final = _candidate_finish_fields(candidate) if _names_its_finish(candidate) else None
        native_finish_reason: Final = candidate.get("finishReason")
        choice: Final = StreamingChoices(
            finish_reason=VertexGeminiConfig._check_finish_reason(chat_completion_message, native_finish_reason)
            if native_finish_reason
            else None,
            index=candidate.get("index", idx),
            delta=Delta(
                content=chat_completion_message.get("content"),
                reasoning_content=chat_completion_message.get("reasoning_content"),
                tool_calls=tools,
                audio=ChatCompletionAudioDelta(data=audio_response.data) if audio_response is not None else None,
                images=image_response,
                function_call=functions,
                annotations=annotations,
                provider_specific_fields=provider_specific_fields,
            ),
            logprobs=chat_completion_logprobs,
            enhancements=None,
            **({"provider_specific_fields": dict(finish_fields)} if finish_fields is not None else {}),
        )
        return choice

    @staticmethod
    def _extract_candidate_metadata(
        candidate: Candidates,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[object], list[object]]:
        """
        Extract metadata from a single candidate response.

        Returns:
            grounding_metadata: List[dict]
            url_context_metadata: List[dict]
            safety_ratings: List
            citation_metadata: List
        """
        grounding_metadata: Final[list[dict[str, object]]] = []
        url_context_metadata: Final[list[dict[str, object]]] = []
        safety_ratings: Final[list[object]] = []
        citation_metadata: Final[list[object]] = []

        if "groundingMetadata" in candidate:
            if isinstance(candidate["groundingMetadata"], (list, tuple)):
                grounding_metadata.extend(candidate["groundingMetadata"])
            else:
                grounding_metadata.append(candidate["groundingMetadata"])

        if "safetyRatings" in candidate:
            safety_ratings.append(candidate["safetyRatings"])

        if "citationMetadata" in candidate:
            citation_metadata.extend(per_hop(candidate["citationMetadata"]))

        if "urlContextMetadata" in candidate:
            url_context_metadata.extend(cast(tuple[dict[str, object], ...], per_hop(candidate["urlContextMetadata"])))

        return (
            grounding_metadata,
            url_context_metadata,
            safety_ratings,
            citation_metadata,
        )

    @staticmethod
    def _get_stream_chunk_attr(chunk: object, field_name: str) -> object:
        if isinstance(chunk, dict):
            value = chunk.get(field_name)
            if value is not None:
                return value
            model_extra: Final = chunk.get("model_extra")
            if isinstance(model_extra, dict):
                value = model_extra.get(field_name)
                if value is not None:
                    return value
            hidden_params: Final = chunk.get("_hidden_params")
            if isinstance(hidden_params, dict):
                return hidden_params.get(field_name)
            return None
        return getattr(chunk, field_name, None)

    @staticmethod
    def _set_stream_metadata_on_response(
        model_response: _StreamMetadataTarget,
        grounding_metadata: Sequence[Mapping[str, object]],
        url_context_metadata: Sequence[Mapping[str, object]],
        safety_ratings: Sequence[Mapping[str, object]],
        citation_metadata: Sequence[Mapping[str, object]],
    ) -> None:
        setattr(model_response, "vertex_ai_grounding_metadata", grounding_metadata)
        if grounding_metadata:
            model_response.hidden_params["vertex_ai_grounding_metadata"] = grounding_metadata
        setattr(model_response, "vertex_ai_url_context_metadata", url_context_metadata)
        if url_context_metadata:
            model_response.hidden_params["vertex_ai_url_context_metadata"] = url_context_metadata
        setattr(model_response, "vertex_ai_safety_ratings", safety_ratings)
        setattr(model_response, "vertex_ai_safety_results", safety_ratings)
        if safety_ratings:
            model_response.hidden_params["vertex_ai_safety_ratings"] = safety_ratings
            model_response.hidden_params["vertex_ai_safety_results"] = safety_ratings
        setattr(model_response, "vertex_ai_citation_metadata", citation_metadata)
        if citation_metadata:
            model_response.hidden_params["vertex_ai_citation_metadata"] = citation_metadata

    @classmethod
    def set_stream_metadata_on_response(
        cls,
        model_response: Union[ModelResponse, "ModelResponseStream"],
        grounding_metadata: list[dict[str, object]],  # mutable-ok: mirrors override contract
        url_context_metadata: list[dict[str, object]],  # mutable-ok: mirrors override contract
        safety_ratings: list[dict[str, object]],  # mutable-ok: mirrors override contract
        citation_metadata: list[dict[str, object]],  # mutable-ok: mirrors override contract
    ) -> None:
        return cls._set_stream_metadata_on_response(
            model_response, grounding_metadata, url_context_metadata, safety_ratings, citation_metadata
        )

    def apply_assembled_streaming_response_metadata(
        self,
        response: ModelResponse,
        chunks: list[object],
    ) -> None:
        for field_name in VERTEX_AI_PROVIDER_METADATA_FIELDS:
            merged: list[object] = []
            for chunk in chunks:
                value = VertexGeminiConfig._get_stream_chunk_attr(chunk, field_name)
                if not value:
                    continue
                if isinstance(value, list):
                    merged.extend(value)
                else:
                    merged.append(value)
            if merged:
                setattr(response, field_name, merged)
                response.hidden_params[field_name] = merged

    @staticmethod
    def _convert_grounding_metadata_to_annotations(
        grounding_metadata: list[dict],
        content_text: str | None,
    ) -> list[ChatCompletionAnnotation]:
        """
        Convert Vertex AI grounding metadata to OpenAI-style annotations.
        """

        annotations: Final[list[ChatCompletionAnnotation]] = []

        for metadata in grounding_metadata:
            # Extract groundingSupports - these map text segments to sources
            grounding_supports = metadata.get("groundingSupports", [])
            grounding_chunks = metadata.get("groundingChunks", [])

            # Build a map of chunk indices to web URIs
            chunk_to_uri_map: dict[int, dict[str, str]] = {}
            for idx, chunk in enumerate(grounding_chunks):
                if "web" in chunk:
                    web_data = chunk["web"]
                    chunk_to_uri_map[idx] = {
                        "url": web_data.get("uri", ""),
                        "title": web_data.get("title", ""),
                    }

            # Process each grounding support to create annotations
            for support in grounding_supports:
                segment = support.get("segment", {})
                start_index = segment.get("startIndex")
                end_index = segment.get("endIndex")

                # Get the chunk indices for this support
                chunk_indices = support.get("groundingChunkIndices", [])

                if start_index is not None and end_index is not None and chunk_indices:
                    for chunk_index in chunk_indices:
                        if chunk_index not in chunk_to_uri_map:
                            continue
                        uri_info = chunk_to_uri_map[chunk_index]

                        url_citation: ChatCompletionAnnotationURLCitation = {
                            "start_index": start_index,
                            "end_index": end_index,
                            "url": uri_info["url"],
                            "title": uri_info["title"],
                        }

                        annotation: ChatCompletionAnnotation = {
                            "type": "url_citation",
                            "url_citation": url_citation,
                        }
                        annotations.append(annotation)
        return annotations

    @staticmethod
    def _process_candidates(
        _candidates: list[Candidates],
        model_response: Union[ModelResponse, "ModelResponseStream"],
        standard_optional_params: dict[str, object],
        cumulative_tool_call_index: int = 0,
        thought_signature_in_tool_call_id: bool = True,
        keeps_requested_audio_format: bool = False,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list, list, int]:
        """
        Helper method to process candidates and extract metadata

        Returns:
            grounding_metadata: List[dict]
            url_context_metadata: List[dict]
            safety_ratings: List
            citation_metadata: List
            cumulative_tool_call_index: int
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            is_function_call,
        )
        from litellm.types.utils import ModelResponseStream

        grounding_metadata: Final[list[dict[str, object]]] = []
        url_context_metadata: Final[list[dict[str, object]]] = []
        safety_ratings: Final[list] = []
        citation_metadata: Final[list] = []

        for idx, candidate in enumerate(_candidates):
            if "content" not in candidate and "finishReason" not in candidate:
                continue

            image_response: list[ImageURLListItem] | None = None
            audio_response: ChatCompletionAudioResponse | None = None
            chat_completion_message: ChatCompletionResponseMessage = {"role": "assistant"}
            chat_completion_logprobs: ChoiceLogprobs | None = None
            tools: list[ChatCompletionToolCallChunk] | None = None
            functions: ChatCompletionToolCallFunctionChunk | None = None
            thinking_blocks: list[ChatCompletionThinkingBlock] | None = None
            reasoning_content: str | None = None
            thought_signatures: Sequence[str] | None = None
            server_side_tool_invocations: list[dict[str, object]] | None = None

            # Extract metadata using helper function
            (
                candidate_grounding_metadata,
                candidate_url_context_metadata,
                candidate_safety_ratings,
                candidate_citation_metadata,
            ) = VertexGeminiConfig._extract_candidate_metadata(candidate)

            grounding_metadata.extend(candidate_grounding_metadata)
            url_context_metadata.extend(candidate_url_context_metadata)
            safety_ratings.extend(candidate_safety_ratings)
            citation_metadata.extend(candidate_citation_metadata)

            if "content" in candidate and candidate["content"] and "parts" in candidate["content"]:
                (
                    content,
                    reasoning_content,
                ) = VertexGeminiConfig().get_assistant_content_message(parts=candidate["content"]["parts"])

                audio_response = VertexGeminiConfig()._extract_audio_response_from_parts(
                    parts=candidate["content"]["parts"],
                    keeps_requested_audio_format=keeps_requested_audio_format,
                )
                image_response = list(
                    _final_images(candidate["content"]["parts"])
                    if isinstance(model_response, ModelResponseStream)
                    else _final_images(candidate["content"]["parts"]) or _fallback_thought_image(candidate)
                )

                thinking_blocks = VertexGeminiConfig()._extract_thinking_blocks_from_parts(
                    parts=candidate["content"]["parts"]
                )

                # Extract thoughtSignatures from parts (can exist without thought: true)
                thought_signatures = VertexGeminiConfig()._extract_thought_signatures_from_parts(
                    parts=candidate["content"]["parts"]
                )

                # Extract server-side tool invocations (context circulation)
                server_side_tool_invocations = VertexGeminiConfig._extract_server_side_tool_invocations(
                    parts=candidate["content"]["parts"]
                )

                if audio_response is not None:
                    cast(dict[str, object], chat_completion_message)["audio"] = audio_response
                    chat_completion_message["content"] = None  # OpenAI spec
                cast(dict[str, object], chat_completion_message)["images"] = image_response
                if content is not None:
                    chat_completion_message["content"] = content

                if reasoning_content is not None:
                    chat_completion_message["reasoning_content"] = reasoning_content

                if candidate_grounding_metadata:
                    annotations = VertexGeminiConfig._convert_grounding_metadata_to_annotations(
                        grounding_metadata=candidate_grounding_metadata,
                        content_text=content,
                    )
                    if annotations:
                        chat_completion_message["annotations"] = annotations
                (
                    functions,
                    tools,
                    cumulative_tool_call_index,
                ) = VertexGeminiConfig._transform_parts(
                    parts=candidate["content"]["parts"],
                    cumulative_tool_call_idx=cumulative_tool_call_index,
                    is_function_call=is_function_call(standard_optional_params),
                    thought_signature_in_tool_call_id=thought_signature_in_tool_call_id,
                )

            if "logprobsResult" in candidate:
                chat_completion_logprobs = VertexGeminiConfig._transform_logprobs(
                    logprobs_result=candidate["logprobsResult"]
                )

            if tools:
                chat_completion_message["tool_calls"] = tools

            if functions is not None:
                chat_completion_message["function_call"] = functions

            if thinking_blocks is not None:
                chat_completion_message["thinking_blocks"] = thinking_blocks

                # Convert thinking_blocks to reasoning_content for streaming
                # This ensures reasoning_content is available in streaming responses
                if isinstance(model_response, ModelResponseStream) and reasoning_content is None:
                    reasoning_content_parts = []
                    for block in thinking_blocks:
                        thinking_text = block.get("thinking")
                        if thinking_text:
                            reasoning_content_parts.append(thinking_text)

                    if reasoning_content_parts:
                        reasoning_content = "\n".join(reasoning_content_parts)
                        chat_completion_message["reasoning_content"] = reasoning_content

            message_provider_fields = {
                key: value
                for key, value in (
                    ("thought_signatures", thought_signatures),
                    ("server_side_tool_invocations", server_side_tool_invocations),
                    (
                        "web_search_calls",
                        list(gemini_web_search_calls(model_response.id, _candidate_web_search_queries(candidate)))
                        or None,
                    ),
                    ("continuation_token", client_continuation_token(candidate)),
                )
                if value is not None
            }
            if message_provider_fields:
                chat_completion_message["provider_specific_fields"] = message_provider_fields

            if isinstance(model_response, ModelResponseStream):
                choice = VertexGeminiConfig._create_streaming_choice(
                    chat_completion_message=chat_completion_message,
                    candidate=candidate,
                    idx=idx,
                    tools=tools,
                    functions=functions,
                    chat_completion_logprobs=chat_completion_logprobs,
                    image_response=image_response,
                    audio_response=audio_response,
                )
                model_response.choices.append(choice)
            elif isinstance(model_response, ModelResponse):
                choice = litellm.Choices(
                    finish_reason=VertexGeminiConfig._check_finish_reason(
                        chat_completion_message, candidate.get("finishReason")
                    ),
                    index=candidate.get("index", idx),
                    message=chat_completion_message,
                    logprobs=chat_completion_logprobs,
                    enhancements=None,
                    provider_specific_fields=(
                        dict(finish_fields)
                        if (finish_fields := _candidate_finish_fields(candidate)) is not None
                        else None
                    ),
                )
                model_response.choices.append(choice)

        return (
            grounding_metadata,
            url_context_metadata,
            safety_ratings,
            citation_metadata,
            cumulative_tool_call_index,
        )

    @classmethod
    def process_candidates(
        cls,
        _candidates: list[Candidates],  # mutable-ok: mirrors override contract
        model_response: Union[ModelResponse, "ModelResponseStream"],
        standard_optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        cumulative_tool_call_index: int = 0,
        thought_signature_in_tool_call_id: bool = True,
        keeps_requested_audio_format: bool = False,
    ) -> tuple[  # mutable-ok: mirrors override contract
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        int,
    ]:
        return cls._process_candidates(
            _candidates,
            model_response,
            standard_optional_params,
            cumulative_tool_call_index,
            thought_signature_in_tool_call_id,
            keeps_requested_audio_format,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Mapping[str, object],
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        return self._log_and_transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            logged_input=messages,
            request_data=request_data,
            keeps_requested_audio_format=requests_audio_mime_type(request_data),
        )

    def transform_pass_through_response(
        self, model: str, raw_response: httpx.Response, logging_obj: LoggingClass
    ) -> ModelResponse:
        return self._log_and_transform_response(
            model=model,
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            logged_input=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
            request_data={},
            keeps_requested_audio_format=True,
        )

    def _log_and_transform_response(
        self,
        *,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        logged_input: Sequence[AllMessageValues],
        request_data: Mapping[str, object],
        keeps_requested_audio_format: bool,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=logged_input,
            api_key="",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        return self._transform_google_generate_content_to_openai_model_response(
            completion_response=parse_body(raw_response),
            model_response=model_response,
            model=model,
            logging_obj=logging_obj,
            raw_response=raw_response,
            keeps_requested_audio_format=keeps_requested_audio_format,
        )

    def _transform_google_generate_content_to_openai_model_response(
        self,
        completion_response: GenerateContentResponseBody | dict[str, object],
        model_response: ModelResponse,
        model: str,
        logging_obj: LoggingClass,
        raw_response: httpx.Response,
        keeps_requested_audio_format: bool = False,
    ) -> ModelResponse:
        """
        Transforms a Google GenAI generate content response to an OpenAI model response.
        """
        if isinstance(completion_response, dict):
            completion_response = GenerateContentResponseBody(**completion_response)

        ## GET MODEL ##
        served: Final = _served_model_name(completion_response.get("modelVersion"))
        model_response.model = served if served is not None else model

        ## CHECK IF RESPONSE FLAGGED
        if "promptFeedback" in completion_response and "blockReason" in completion_response["promptFeedback"]:
            return self._content_filter_response(
                model_response=model_response,
                completion_response=completion_response,
                provider_specific_fields=_prompt_block_fields(completion_response["promptFeedback"]),
            )

        _candidates: Final = completion_response.get("candidates")
        if _candidates and len(_candidates) > 0:
            content_policy_violations: Final = VertexGeminiConfig().get_flagged_finish_reasons()
            flagged_fields: Final = _candidate_finish_fields(_candidates[0])
            if flagged_fields is not None and flagged_fields["native_finish_reason"] in content_policy_violations:
                return self._content_filter_response(
                    model_response=model_response,
                    completion_response=completion_response,
                    provider_specific_fields=flagged_fields,
                )

        model_response.choices = []
        response_id: Final = completion_response.get("responseId")
        if response_id:
            model_response.id = response_id
        url_context_metadata: list[dict[str, object]] = []
        try:
            grounding_metadata: list[dict[str, object]] = []
            safety_ratings: list[dict] = []
            citation_metadata: list[dict] = []
            if _candidates:
                (
                    grounding_metadata,
                    url_context_metadata,
                    safety_ratings,
                    citation_metadata,
                    _,  # cumulative_tool_call_index not needed in non-streaming
                ) = VertexGeminiConfig._process_candidates(
                    _candidates,
                    model_response,
                    logging_obj.optional_params,
                    thought_signature_in_tool_call_id=embeds_thought_signature_in_tool_call_id(
                        logging_obj.litellm_params  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Logging.litellm_params is untyped
                    ),
                    keeps_requested_audio_format=keeps_requested_audio_format,
                )

            usage: Final = VertexGeminiConfig._calculate_usage(
                completion_response=completion_response,
                cache_creation_tokens=_cache_creation_tokens(logging_obj),
            )

            VertexGeminiConfig._set_grounding_usage_counters(
                usage, grounding_metadata, search_tool_call_queries(_parts_of_candidates(_candidates or ()))
            )

            setattr(model_response, "usage", usage)

            ## ADD METADATA TO RESPONSE ##

            setattr(model_response, "vertex_ai_grounding_metadata", grounding_metadata)
            model_response.hidden_params["vertex_ai_grounding_metadata"] = grounding_metadata

            setattr(model_response, "vertex_ai_url_context_metadata", url_context_metadata)

            model_response.hidden_params["vertex_ai_url_context_metadata"] = url_context_metadata

            setattr(model_response, "vertex_ai_safety_results", safety_ratings)
            model_response.hidden_params["vertex_ai_safety_results"] = (
                safety_ratings  # older approach - maintaining to prevent regressions
            )

            ## ADD CITATION METADATA ##
            setattr(model_response, "vertex_ai_citation_metadata", citation_metadata)
            model_response.hidden_params["vertex_ai_citation_metadata"] = (
                citation_metadata  # older approach - maintaining to prevent regressions
            )

            ## ADD TRAFFIC TYPE ##
            traffic_type: Final = completion_response.get("usageMetadata", {}).get("trafficType")
            if traffic_type:
                model_response.hidden_params.setdefault("provider_specific_fields", {})["traffic_type"] = traffic_type

            ## ADD SERVICE TIER ##
            response_headers: Final = getattr(raw_response, "headers", None)
            service_tier: Final = _response_service_tier(
                _response_header(response_headers, "x-gemini-service-tier") if response_headers else None,
                traffic_type,
            )
            if service_tier:
                setattr(model_response, "service_tier", service_tier)

        except Exception as e:
            raise VertexAIError(
                message=f"Error converting to valid response block={e}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues",
                status_code=422,
                headers=raw_response.headers,
            )

        return model_response

    def transform_google_generate_content_to_openai_model_response(
        self,
        completion_response: GenerateContentResponseBody | dict[str, object],  # mutable-ok: mirrors override contract
        model_response: ModelResponse,
        model: str,
        logging_obj: LoggingClass,
        raw_response: httpx.Response,
        keeps_requested_audio_format: bool = False,
    ) -> ModelResponse:
        return self._transform_google_generate_content_to_openai_model_response(
            completion_response, model_response, model, logging_obj, raw_response, keeps_requested_audio_format
        )

    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str | None = None,
        litellm_params: dict[str, object] | None = None,
    ) -> list[ContentType]:
        return gemini_convert_messages_with_history(
            messages=messages,
            model=model,
            litellm_params=litellm_params,
            custom_llm_provider="vertex_ai",
        )

    def transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: mirrors override contract
        model: str | None = None,
        litellm_params: dict[str, object] | None = None,  # mutable-ok: mirrors override contract
    ) -> list[ContentType]:  # mutable-ok: mirrors override contract
        return self._transform_messages(messages, model, litellm_params)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VertexAIError(message=error_message, status_code=status_code, headers=headers)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:
        raise NotImplementedError("Vertex AI has a custom implementation of transform_request. Needs sync + async.")

    def validate_environment(
        self,
        headers: Mapping[str, str] | None,
        model: str,
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | dict[str, str] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        default_headers: Final = {
            "Content-Type": "application/json",
        }
        if isinstance(api_key, dict):
            default_headers.update(api_key)
        elif api_key is not None:
            default_headers["Authorization"] = f"Bearer {api_key}"

        caller_headers: Final[Mapping[str, str]] = cast(Mapping[str, str], headers or {})  # cast-ok: bare dict
        service_tier: Final = optional_params.get("service_tier")
        is_vertex: Final = litellm_params.get("custom_llm_provider") != litellm.LlmProviders.GEMINI
        if is_vertex and isinstance(service_tier, str) and service_tier in VERTEX_SHARED_REQUEST_TIERS:
            caller_header_names: Final = frozenset(name.lower() for name in caller_headers)
            if "x-vertex-ai-llm-shared-request-type" not in caller_header_names:
                default_headers["X-Vertex-AI-LLM-Shared-Request-Type"] = service_tier

        default_headers.update(caller_headers)
        return default_headers


def _log_generate_request(
    logging_obj: LoggingClass,
    messages: Sequence[AllMessageValues],
    api_base: str,
    headers: Mapping[str, str],
    body: RequestBody,
) -> None:
    logging_obj.pre_call(
        input=messages,
        api_key="",
        additional_args={"complete_input_dict": body, "api_base": api_base, "headers": headers},
    )


async def make_call(
    client: AsyncHTTPHandler | None,  # module-level client
    gemini_client: AsyncHTTPHandler | None,  # if passed by user
    api_base: str,
    headers: dict[str, str],
    data: str,
    model: str,
    messages: Sequence[object],
    logging_obj: LoggingClass,
    request_body: RequestBody,
    hop_headers: Callable[[], Awaitable[Mapping[str, str]]],
    timeout: float | httpx.Timeout | None = None,
):
    if gemini_client is not None:
        client = gemini_client
    if client is None:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )

    try:
        response: Final = await client.post(
            url_for(api_base, request_body),
            headers=headers,
            data=data,
            stream=True,
            logging_obj=logging_obj,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        exception_string: Final = str(await e.response.aread())
        known_cached_contents.forget_rejected(request_body.get("cachedContent"), e.response.status_code)
        raise VertexAIError(
            status_code=e.response.status_code,
            message=VertexGeminiConfig().translate_exception_str(exception_string),
            headers=e.response.headers,
        )
    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=response.text,
            headers=response.headers,
        )

    async def open_hop(body: RequestBody) -> httpx.Response:
        try:
            return await client.post(
                url_for(api_base, body),
                headers=dict(await hop_headers()),
                data=json.dumps(body),
                stream=True,
                logging_obj=logging_obj,
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            known_cached_contents.forget_rejected(body.get("cachedContent"), e.response.status_code)
            raise

    completion_stream: Final = ModelResponseIterator(
        streaming_response=aiter_sse_lines(response.aiter_text()),
        sync_stream=False,
        logging_obj=logging_obj,
        response_headers=response.headers,
        response=response,
        resume=AsyncStreamResume(request_body=request_body, open_hop=open_hop),
    )
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: HTTPHandler | None,  # module-level client
    gemini_client: HTTPHandler | None,  # if passed by user
    api_base: str,
    headers: dict[str, str],
    data: str,
    model: str,
    messages: Sequence[object],
    logging_obj: LoggingClass,
    request_body: RequestBody,
    hop_headers: Callable[[], Mapping[str, str]],
    timeout: float | httpx.Timeout | None = None,
):
    if gemini_client is not None:
        client = gemini_client
    if client is None:
        client = HTTPHandler()  # Create a new client if none provided

    try:
        response: Final = client.post(
            url_for(api_base, request_body),
            headers=headers,
            data=data,
            stream=True,
            logging_obj=logging_obj,
            timeout=timeout,
        )
    except httpx.HTTPStatusError as e:
        known_cached_contents.forget_rejected(request_body.get("cachedContent"), e.response.status_code)
        raise

    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=str(response.read()),
            headers=response.headers,
        )

    def open_hop(body: RequestBody) -> httpx.Response:
        try:
            return client.post(
                url_for(api_base, body),
                headers=dict(hop_headers()),
                data=json.dumps(body),
                stream=True,
                logging_obj=logging_obj,
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            known_cached_contents.forget_rejected(body.get("cachedContent"), e.response.status_code)
            raise

    completion_stream: Final = ModelResponseIterator(
        streaming_response=iter_sse_lines(response.iter_text()),
        sync_stream=True,
        logging_obj=logging_obj,
        response_headers=response.headers,
        response=response,
        resume=SyncStreamResume(request_body=request_body, open_hop=open_hop),
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class VertexLLM(VertexBase):
    def __init__(self) -> None:
        super().__init__()

    async def _fresh_headers_async(
        self,
        headers: Mapping[str, str],
        extra_headers: Mapping[str, str] | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    ) -> Mapping[str, str]:
        if custom_llm_provider == "gemini":
            return headers
        try:
            token, _ = await self._ensure_access_token_async(
                credentials=vertex_credentials, project_id=vertex_project, custom_llm_provider=custom_llm_provider
            )
        except Exception as e:
            raise VertexAIError(status_code=401, message=str(e), headers=None) from e
        return {**headers, "Authorization": f"Bearer {token}", **(extra_headers or {})}

    def _fresh_headers(
        self,
        headers: Mapping[str, str],
        extra_headers: Mapping[str, str] | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    ) -> Mapping[str, str]:
        if custom_llm_provider == "gemini":
            return headers
        try:
            token, _ = self._ensure_access_token(
                credentials=vertex_credentials, project_id=vertex_project, custom_llm_provider=custom_llm_provider
            )
        except Exception as e:
            raise VertexAIError(status_code=401, message=str(e), headers=None) from e
        return {**headers, "Authorization": f"Bearer {token}", **(extra_headers or {})}

    async def async_streaming(
        self,
        model: str,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        messages: list[AllMessageValues],
        model_response: ModelResponse,
        print_verbose: Callable[..., object],
        data: dict,
        timeout: float | httpx.Timeout | None,
        encoding: object,
        logging_obj: LoggingClass,
        stream: bool | None,
        optional_params: dict,
        litellm_params: Mapping[str, object],
        logger_fn: object = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None = None,
        gemini_api_key: str | None = None,
        extra_headers: dict | None = None,
    ) -> CustomStreamWrapper:
        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        # Extract use_psc_endpoint_format from optional_params
        use_psc_endpoint_format: Final = optional_params.get("use_psc_endpoint_format", False)

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        headers: Final = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        transform: Final = partial(
            async_transform_request_body,
            **data,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=auth_header,
        )
        rebuild, uncached = cache_rebuilds(transform, messages, optional_params)
        request_body: Final = await build_with_accepted_thought_signatures_async(
            transform, with_placeholder_thought_signatures, uncached
        )

        ## LOGGING
        log_request: Final = partial(_log_generate_request, logging_obj, messages, api_base, headers)
        log_request(request_body)

        open_stream: Final = partial(
            make_call,
            gemini_client=(client if client is not None and isinstance(client, AsyncHTTPHandler) else None),
            api_base=api_base,
            headers=headers,
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            hop_headers=partial(
                self._fresh_headers_async,
                headers,
                extra_headers,
                vertex_credentials,
                vertex_project,
                custom_llm_provider,
            ),
            timeout=timeout,
        )

        async def open_with_fresh_cache(client: AsyncHTTPHandler | None) -> ModelResponseIterator:
            async def post(body: RequestBody) -> ModelResponseIterator:
                return await open_stream(client=client, data=json.dumps(body), request_body=body)

            accepted: Final = partial(
                post_with_accepted_thought_signatures_async,
                placeholders=with_placeholder_thought_signatures,
                uncached=uncached,
                post=post,
                log_request=log_request,
            )
            return (await post_with_fresh_cache_async(request_body, rebuild, accepted, log_request))[1][1]

        streaming_response: Final = CustomStreamWrapper(
            completion_stream=None,
            make_call=open_with_fresh_cache,
            model=model,
            custom_llm_provider="vertex_ai_beta",
            logging_obj=logging_obj,
        )
        return streaming_response

    async def async_completion(
        self,
        model: str,
        messages: list[AllMessageValues],
        model_response: ModelResponse,
        print_verbose: Callable[..., object],
        data: dict,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        timeout: float | httpx.Timeout | None,
        encoding: object,
        logging_obj: LoggingClass,
        stream: bool | None,
        optional_params: dict,
        litellm_params: Mapping[str, object],
        logger_fn: object = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None = None,
        gemini_api_key: str | None = None,
        extra_headers: dict | None = None,
    ) -> ModelResponse | CustomStreamWrapper:
        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        # Extract use_psc_endpoint_format from optional_params
        use_psc_endpoint_format: Final = optional_params.get("use_psc_endpoint_format", False)

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        headers: Final = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        transform: Final = partial(
            async_transform_request_body,
            **data,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=auth_header,
        )
        rebuild, uncached = cache_rebuilds(transform, messages, optional_params)
        request_body: Final = await build_with_accepted_thought_signatures_async(
            transform, with_placeholder_thought_signatures, uncached
        )

        _async_client_params: Final = {}
        if timeout:
            _async_client_params["timeout"] = timeout
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(params=_async_client_params, llm_provider=litellm.LlmProviders.VERTEX_AI)
        else:
            client = client
        ## LOGGING
        log_request: Final = partial(_log_generate_request, logging_obj, messages, api_base, headers)
        log_request(request_body)

        fresh_headers: Final = partial(
            self._fresh_headers_async, headers, extra_headers, vertex_credentials, vertex_project, custom_llm_provider
        )

        async def post(body: RequestBody) -> httpx.Response:
            try:
                response: Final = await client.post(
                    url_for(api_base, body),
                    headers=dict(await fresh_headers()),
                    json=dict(body),
                    logging_obj=logging_obj,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                known_cached_contents.forget_rejected(body.get("cachedContent"), err.response.status_code)
                raise VertexAIError(
                    status_code=err.response.status_code,
                    message=err.response.text,
                    headers=err.response.headers,
                )
            except httpx.TimeoutException:
                raise VertexAIError(
                    status_code=408,
                    message="Timeout error occurred.",
                    headers=None,
                )
            return response

        accepted: Final = partial(
            post_with_accepted_thought_signatures_async,
            placeholders=with_placeholder_thought_signatures,
            uncached=uncached,
            post=post,
            log_request=log_request,
        )
        _, (sent_body, first_response) = await post_with_fresh_cache_async(request_body, rebuild, accepted, log_request)
        return _transform_long_decode_hops(
            model=model,
            hops=await collect_hops_async(parse_body(first_response), sent_body, post),
            first_response=first_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_body=sent_body,
            messages=messages,
        )

    def completion(
        self,
        model: str,
        messages: list[AllMessageValues],
        model_response: ModelResponse,
        print_verbose: Callable[..., object],
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        encoding: object,
        logging_obj: LoggingClass,
        optional_params: dict,
        acompletion: bool,
        timeout: float | httpx.Timeout | None,
        vertex_project: str | None,
        vertex_location: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        gemini_api_key: str | None,
        litellm_params: dict,
        logger_fn: object = None,
        extra_headers: dict | None = None,
        client: AsyncHTTPHandler | HTTPHandler | None = None,
        api_base: str | None = None,
    ) -> ModelResponse | CustomStreamWrapper:
        stream: Final[bool | None] = optional_params.pop("stream", None)
        caller_headers: Final = (
            dict(without_caller_labels_header(_CALLER_HEADERS.validate_python(extra_headers)))
            if extra_headers is not None and custom_llm_provider != "gemini" and vertex_request_label_fields()
            else extra_headers
        )

        transform_request_params: Final = {
            "gemini_api_key": gemini_api_key,
            "messages": messages,
            "api_base": api_base,
            "model": model,
            "client": client,
            "timeout": timeout,
            "extra_headers": caller_headers,
            "optional_params": optional_params,
            "logging_obj": logging_obj,
            "custom_llm_provider": custom_llm_provider,
            "litellm_params": litellm_params,
        }

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            ### ASYNC STREAMING
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,
                    client=client,
                    data=transform_request_params,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    vertex_credentials=vertex_credentials,
                    gemini_api_key=gemini_api_key,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=caller_headers,
                )
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=transform_request_params,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,
                client=client,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                gemini_api_key=gemini_api_key,
                custom_llm_provider=custom_llm_provider,
                extra_headers=caller_headers,
            )

        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        # Extract use_psc_endpoint_format from optional_params
        use_psc_endpoint_format: Final = optional_params.get("use_psc_endpoint_format", False)

        auth_header, url = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            use_psc_endpoint_format=use_psc_endpoint_format,
        )
        headers: Final = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=caller_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## TRANSFORMATION ##
        transform: Final = partial(
            sync_transform_request_body,
            **transform_request_params,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_auth_header=auth_header,
        )
        rebuild, uncached = cache_rebuilds(transform, messages, optional_params)
        data: Final = build_with_accepted_thought_signatures_sync(
            transform, with_placeholder_thought_signatures, uncached
        )

        ## LOGGING
        log_request: Final = partial(_log_generate_request, logging_obj, messages, url, headers)
        log_request(data)

        fresh_headers: Final = partial(
            self._fresh_headers, headers, caller_headers, vertex_credentials, vertex_project, custom_llm_provider
        )

        ## SYNC STREAMING CALL ##
        if stream is True:
            open_stream: Final = partial(
                make_sync_call,
                gemini_client=(client if client is not None and isinstance(client, HTTPHandler) else None),
                api_base=url,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
                headers=headers,
                hop_headers=fresh_headers,
                timeout=timeout,
            )

            def open_with_fresh_cache(client: HTTPHandler | None) -> ModelResponseIterator:
                def post(body: RequestBody) -> ModelResponseIterator:
                    return open_stream(client=client, data=json.dumps(body), request_body=body)

                accepted: Final = partial(
                    post_with_accepted_thought_signatures_sync,
                    placeholders=with_placeholder_thought_signatures,
                    uncached=uncached,
                    post=post,
                    log_request=log_request,
                )
                return post_with_fresh_cache_sync(data, rebuild, accepted, log_request)[1][1]

            streaming_response: Final = CustomStreamWrapper(
                completion_stream=None,
                make_call=open_with_fresh_cache,
                model=model,
                custom_llm_provider="vertex_ai_beta",
                logging_obj=logging_obj,
            )

            return streaming_response
        ## COMPLETION CALL ##

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = get_httpx_client(params=_params)
        else:
            client = client

        def post(body: RequestBody) -> httpx.Response:
            try:
                response: Final = client.post(
                    url=url_for(url, body), headers=dict(fresh_headers()), json=dict(body), logging_obj=logging_obj
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as err:
                known_cached_contents.forget_rejected(body.get("cachedContent"), err.response.status_code)
                raise VertexAIError(
                    status_code=err.response.status_code,
                    message=err.response.text,
                    headers=err.response.headers,
                )
            except httpx.TimeoutException:
                raise VertexAIError(
                    status_code=408,
                    message="Timeout error occurred.",
                    headers=None,
                )
            return response

        accepted: Final = partial(
            post_with_accepted_thought_signatures_sync,
            placeholders=with_placeholder_thought_signatures,
            uncached=uncached,
            post=post,
            log_request=log_request,
        )
        _, (sent_body, first_response) = post_with_fresh_cache_sync(data, rebuild, accepted, log_request)
        return _transform_long_decode_hops(
            model=model,
            hops=collect_hops_sync(parse_body(first_response), sent_body, post),
            first_response=first_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_body=sent_body,
            messages=messages,
        )


def _transform_long_decode_hops(
    model: str,
    hops: LongDecodeHops,
    first_response: httpx.Response,
    model_response: ModelResponse,
    logging_obj: LoggingClass,
    request_body: RequestBody,
    messages: Sequence[AllMessageValues],
) -> ModelResponse:
    body: Final = merge_hops(hops)
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=hops_original_response(hops, body, first_response),
        additional_args={"complete_input_dict": request_body},
    )
    merged: Final = VertexGeminiConfig().transform_google_generate_content_to_openai_model_response(
        completion_response=body,
        model_response=model_response,
        model=model,
        logging_obj=logging_obj,
        raw_response=first_response,
        keeps_requested_audio_format=requests_audio_mime_type(request_body),
    )
    merged_usage: Final = getattr(merged, "usage", None)
    marked: Final = (
        with_initial_prompt_tokens(merged_usage, initial_prompt_tokens_of(hops))
        if isinstance(merged_usage, Usage)
        else merged_usage
    )
    response: Final = merged if marked is merged_usage else merged.model_copy(update={"usage": marked})
    failure: Final = _malformed_without_output_error(response, first_response.headers)
    if failure is None:
        return response
    failed_attempt: Final = logging_obj.served_round(response)
    logging_obj.record_served_rounds_for_failure(() if failed_attempt is None else (failed_attempt,))
    raise failure


def _provider_finish_reasons(
    candidates: Sequence[Candidates], choices: Sequence["StreamingChoices"], blocked_indices: frozenset[int]
) -> tuple[tuple[int, str], ...]:
    """A streamed choice also carries "stop" for Google's FINISH_REASON_UNSPECIFIED, so only a candidate that named how
    it ended, or a blocked prompt, is evidence of it."""
    finished: Final = frozenset(
        candidate.get("index", position)
        for position, candidate in enumerate(candidates)
        if _names_its_finish(candidate)
    )
    finished_or_blocked: Final = finished | blocked_indices
    return tuple(
        (choice.index, choice.finish_reason)
        for choice in choices
        if choice.index in finished_or_blocked and choice.finish_reason
    )


class ModelResponseIterator:
    def __init__(
        self,
        streaming_response,
        sync_stream: bool,
        logging_obj: LoggingClass,
        response_headers: dict[str, str] | None = None,
        response: httpx.Response | None = None,
        resume: StreamResume | None = None,
        returns_provider_audio: bool = False,
    ):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            check_is_function_call,
        )

        self.streaming_response = streaming_response
        self.response = response
        self.chunk_type: Literal["valid_json", "accumulated_json"] = "valid_json"
        self._json_buffer = JSONFragmentAccumulator()
        self.sent_first_chunk = False
        self.logging_obj = logging_obj
        self.response_headers = response_headers or {}
        self.is_function_call = check_is_function_call(logging_obj)
        self.thought_signature_in_tool_call_id: Final = embeds_thought_signature_in_tool_call_id(
            logging_obj.litellm_params  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Logging.litellm_params is untyped
        )
        self.cumulative_tool_call_index: int = 0
        self.has_seen_tool_calls: bool = False
        self._resume = resume
        self._output_cap = continuation_cap_of(resume.request_body) if resume is not None else None
        self._keeps_requested_audio_format: Final = returns_provider_audio or (
            resume is not None and requests_audio_mime_type(resume.request_body)
        )
        self._pending_hop_token: str | None = None
        self._unanswered_hop_token: str | None = None
        self._token_after_unsent_images: str | None = None
        self._stream_end_after_unsent_images: Exception | None = None
        self._first_response_id: str | None = None
        self._usage_offset: UsageMetadata = UsageMetadata()
        self._hop_usage: UsageMetadata = UsageMetadata()
        self._initial_prompt_tokens: int | None = None
        self._grounding_offset: tuple[Mapping[str, object], ...] = ()
        self._hop_grounding: tuple[Mapping[str, object], ...] = ()
        self._search_tool_call_queries: tuple[str, ...] = ()
        self._web_search_calls_by_choice: Mapping[int, Mapping[str, ResponseFunctionWebSearch]] = MappingProxyType({})
        self._answer_offset: int = 0
        self._hop_answer_bytes: int = 0
        self._held_thought_images: Mapping[int, tuple[ImageURLListItem, ...]] = MappingProxyType({})
        self._image_settled_indexes: frozenset[int] = frozenset()

    @property
    def accumulated_json(self) -> str:
        return self._json_buffer.snapshot()

    @accumulated_json.setter
    def accumulated_json(self, value: str) -> None:
        self._json_buffer.set(value)

    @staticmethod
    def _check_streaming_error(chunk: Mapping[str, object]) -> None:
        """Detect embedded errors (e.g. 429 RESOURCE_EXHAUSTED) in streaming chunks and raise VertexAIError."""
        if "error" not in chunk:
            return
        error_data: Final = chunk["error"]
        if not isinstance(error_data, dict):
            raise VertexAIError(
                status_code=500,
                message=f"Unexpected error format in mid-stream chunk: {error_data}",
            )
        raw_code = error_data.get("code", 500)
        if raw_code is None:
            raw_code = 500
        try:
            error_code = int(raw_code)
        except (TypeError, ValueError):
            error_code = 500
        error_message: Final = error_data.get("message", "Unknown error")
        error_status: Final = error_data.get("status", "UNKNOWN")
        raise VertexAIError(
            status_code=error_code,
            message=f"{error_status} - {error_message}",
        )

    def _apply_stream_candidates(
        self,
        _candidates: list[Candidates],
        model_response: "ModelResponseStream",
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        (
            grounding_metadata,
            url_context_metadata,
            safety_ratings,
            citation_metadata,
            self.cumulative_tool_call_index,
        ) = VertexGeminiConfig.process_candidates(
            _candidates,
            model_response,
            self.logging_obj.optional_params,
            cumulative_tool_call_index=self.cumulative_tool_call_index,
            thought_signature_in_tool_call_id=self.thought_signature_in_tool_call_id,
            keeps_requested_audio_format=self._keeps_requested_audio_format,
        )

        # Track whether tool_calls have been seen across streaming chunks.
        # Gemini sends tool_calls and finishReason in separate chunks,
        # so we need to remember if earlier chunks contained tool_calls
        # to correctly set finish_reason="tool_calls" per the OpenAI spec.
        if not self.has_seen_tool_calls:
            for choice in model_response.choices:
                if hasattr(choice, "delta") and choice.delta and choice.delta.tool_calls:
                    self.has_seen_tool_calls = True
                    break

        # _process_candidates skips candidates with neither "content" nor
        # "finishReason", so a metadata-only chunk (grounding/web-search/thought,
        # seen with web_search + reasoning) leaves choices empty and the downstream
        # streaming handler hits IndexError on choices[0]. Emit an empty-delta choice.
        if not model_response.choices and _candidates:
            from litellm.types.utils import Delta, StreamingChoices

            for candidate in _candidates:
                finish_reason_str = candidate.get("finishReason")
                mapped_finish_reason = None
                if finish_reason_str is not None:
                    if self.has_seen_tool_calls:
                        mapped_finish_reason = "tool_calls"
                    else:
                        mapped_finish_reason = VertexGeminiConfig.check_finish_reason(None, finish_reason_str)
                choice = StreamingChoices(
                    finish_reason=mapped_finish_reason,
                    index=candidate.get("index", 0),
                    delta=Delta(content=None, role=None),
                    logprobs=None,
                    enhancements=None,
                )
                model_response.choices.append(choice)

        # Also handle the case where the final chunk has empty
        # content (e.g. text:"") WITH finishReason. In this case
        # _process_candidates DOES create a choice, but maps
        # finishReason="STOP" to "stop" because the current chunk
        # has no tool_calls. Override if we saw tool_calls earlier.
        if self.has_seen_tool_calls:
            for choice in model_response.choices:
                if choice.finish_reason == "stop":
                    choice.finish_reason = "tool_calls"

        VertexGeminiConfig.set_stream_metadata_on_response(
            model_response,
            grounding_metadata,
            url_context_metadata,
            safety_ratings,
            citation_metadata,
        )

        return (
            grounding_metadata,
            url_context_metadata,
            safety_ratings,
            citation_metadata,
        )

    def _marked_stream_usage(self, usage: Usage | None) -> Usage | None:
        if usage is None or not self._usage_offset:
            return usage
        return with_initial_prompt_tokens(usage, self._initial_prompt_tokens)

    def _grounding_through_this_hop(
        self, grounding_metadata: Sequence[Mapping[str, object]]
    ) -> Sequence[Mapping[str, object]]:
        if grounding_metadata:
            self._hop_grounding = tuple(grounding_metadata)
        if not self._grounding_offset:
            return grounding_metadata
        return (*self._grounding_offset, *self._hop_grounding)

    def _apply_stream_usage_metadata(
        self,
        processed_chunk: GenerateContentResponseBody,
        model_response: "ModelResponseStream",
        grounding_metadata: list[dict[str, object]],
    ) -> Usage | None:
        if "usageMetadata" not in processed_chunk:
            return None

        usage: Final = VertexGeminiConfig.calculate_usage(
            completion_response=processed_chunk,
            cache_creation_tokens=_cache_creation_tokens(self.logging_obj),
        )

        VertexGeminiConfig.set_grounding_usage_counters(
            usage, self._grounding_through_this_hop(grounding_metadata), self._search_tool_call_queries
        )

        traffic_type: Final = processed_chunk.get("usageMetadata", {}).get("trafficType")
        if traffic_type:
            model_response.hidden_params.setdefault("provider_specific_fields", {})["traffic_type"] = traffic_type

        service_tier: Final = _response_service_tier(self.response_headers.get("x-gemini-service-tier"), traffic_type)
        if service_tier:
            setattr(model_response, "service_tier", service_tier)

        return usage

    def chunk_parser(self, chunk: dict) -> Optional["ModelResponseStream"]:
        try:
            verbose_logger.debug("RAW GEMINI CHUNK: %s", chunk)

            # Detect mid-stream error chunks (e.g. 429 RESOURCE_EXHAUSTED).
            # Vertex AI can return errors as HTTP 200 but with an "error" field in the SSE body.
            self._check_streaming_error(chunk)
            self._unanswered_hop_token = None

            from litellm.types.utils import ModelResponseStream

            processed_chunk: Final = self._with_answer_offset(
                self._with_usage_offset(GenerateContentResponseBody(**chunk))
            )
            self._first_response_id = self._first_response_id or processed_chunk.get("responseId")
            response_id: Final = self._first_response_id
            served: Final = _served_model_name(processed_chunk.get("modelVersion"))
            model_response = ModelResponseStream(
                choices=[],
                id=response_id,
                model=served,
            )

            # Check if prompt is blocked due to content filtering
            blocked_response: Final = VertexGeminiConfig.check_prompt_level_content_filter(
                processed_chunk=processed_chunk,
                response_id=response_id,
                model=served,
            )
            if blocked_response is not None:
                model_response = blocked_response

            grounding_metadata: list[dict[str, object]] = []
            url_context_metadata: list[dict[str, object]] = []
            safety_ratings: list[dict[str, object]] = []
            citation_metadata: list[dict[str, object]] = []

            _candidates: Final[list[Candidates] | None] = processed_chunk.get("candidates")
            if _candidates:
                (
                    grounding_metadata,
                    url_context_metadata,
                    safety_ratings,
                    citation_metadata,
                ) = self._apply_stream_candidates(_candidates, model_response)
                self._search_tool_call_queries = (
                    *self._search_tool_call_queries,
                    *search_tool_call_queries(_parts_of_candidates(_candidates)),
                )
                self._report_web_search_calls_so_far(model_response)

            usage: Final = self._marked_stream_usage(
                self._apply_stream_usage_metadata(processed_chunk, model_response, grounding_metadata)
            )

            setattr(model_response, "usage", usage)
            self._plan_resume(processed_chunk, model_response)
            self._hold_thought_images(_candidates or ())
            released_response: Final = self._release_held_thought_images(_candidates or (), model_response)

            provider_finish_reasons: Final = _provider_finish_reasons(
                _candidates or (),
                released_response.choices,
                frozenset(choice.index for choice in blocked_response.choices)
                if blocked_response is not None
                else frozenset(),
            )
            if provider_finish_reasons:
                set_hidden_param(released_response, PROVIDER_FINISH_REASONS_HIDDEN_PARAM, provider_finish_reasons)
            released_response.hidden_params["is_finished"] = False
            return released_response

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    def _report_web_search_calls_so_far(self, model_response: "ModelResponseStream") -> None:
        searched: Final = frozenset(
            call.id for call in gemini_web_search_calls(model_response.id, self._search_tool_call_queries)
        )
        for choice in model_response.choices:
            if (fields := self._fields_with_web_search_calls_so_far(choice, searched)) is not None:
                choice.delta.provider_specific_fields = fields

    def _fields_with_web_search_calls_so_far(
        self, choice: "StreamingChoices", searched: frozenset[str]
    ) -> dict[str, object] | None:
        fields: Final[Mapping[str, object]] = choice.delta.provider_specific_fields or {}
        arrived: Final = as_object_list(fields.get("web_search_calls"))
        if not arrived:
            return None
        reported: Final = MappingProxyType(
            {
                **self._web_search_calls_by_choice.get(choice.index, {}),
                **{
                    call.id: call
                    for call in arrived
                    if isinstance(call, ResponseFunctionWebSearch) and (not searched or call.id in searched)
                },
            }
        )
        self._web_search_calls_by_choice = MappingProxyType(
            {**self._web_search_calls_by_choice, choice.index: reported}
        )
        return {**fields, "web_search_calls": list(reported.values())}

    def _with_usage_offset(self, chunk: GenerateContentResponseBody) -> GenerateContentResponseBody:
        if "usageMetadata" not in chunk:
            return chunk
        self._hop_usage = chunk["usageMetadata"]
        if not self._usage_offset:
            return chunk
        return {**chunk, "usageMetadata": sum_usage_metadata(self._usage_offset, self._hop_usage)}

    def _with_answer_offset(self, chunk: GenerateContentResponseBody) -> GenerateContentResponseBody:
        candidates: Final = chunk.get("candidates")
        if self._output_cap is None or not candidates:
            return chunk
        self._hop_answer_bytes += answer_bytes(candidates[0])
        if not self._answer_offset:
            return chunk
        return {**chunk, "candidates": [with_spans_shifted(candidates[0], self._answer_offset), *candidates[1:]]}

    def _hold_thought_images(self, candidates: Sequence[Candidates]) -> None:
        indexed_parts: Final = tuple(
            (candidate.get("index", position), _content_parts(candidate))
            for position, candidate in enumerate(candidates)
        )
        self._image_settled_indexes = self._image_settled_indexes | frozenset(
            index for index, parts in indexed_parts if _final_images(parts)
        )
        self._held_thought_images = MappingProxyType(
            {
                **self._held_thought_images,
                **{
                    index: _last_thought_image(parts) or self._held_thought_images.get(index, ())
                    for index, parts in indexed_parts
                },
            }
        )

    def _release_held_thought_images(
        self, candidates: Sequence[Candidates], model_response: "ModelResponseStream"
    ) -> "ModelResponseStream":
        finishing: Final = MappingProxyType(
            {
                candidate.get("index", position): candidate
                for position, candidate in enumerate(candidates)
                if _names_its_finish(candidate)
            }
        )
        settling: Final = frozenset(
            choice.index
            for choice in model_response.choices
            if choice.index in finishing
            and choice.finish_reason is not None
            and choice.index not in self._image_settled_indexes
        )
        self._image_settled_indexes = self._image_settled_indexes | settling
        released: Final = MappingProxyType(
            {
                index: held
                for index in settling
                if (held := self._held_thought_images.get(index)) and not _blocks_its_output(finishing[index])
            }
        )
        if not released:
            return model_response
        return model_response.model_copy(
            update={
                "choices": [
                    _with_delta_images(choice, released[choice.index]) if choice.index in released else choice
                    for choice in model_response.choices
                ]
            }
        )

    def _take_unsent_thought_images(self) -> Mapping[int, tuple[ImageURLListItem, ...]]:
        unsent: Final = MappingProxyType(
            {
                index: held
                for index, held in self._held_thought_images.items()
                if held and index not in self._image_settled_indexes
            }
        )
        self._image_settled_indexes = self._image_settled_indexes | frozenset(unsent)
        return unsent

    def _unsent_thought_images_end(self) -> Optional["ModelResponseStream"]:
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        unsent: Final = self._take_unsent_thought_images()
        if not unsent:
            return None
        return ModelResponseStream(
            id=self._first_response_id,
            choices=[StreamingChoices(index=index, delta=Delta(images=list(held))) for index, held in unsent.items()],
        )

    def _plan_resume(self, processed_chunk: GenerateContentResponseBody, model_response: "ModelResponseStream") -> None:
        if self._output_cap is None:
            return
        token: Final = next_hop_token(
            processed_chunk,
            hop_usage=self._hop_usage,
            produced_before=output_tokens_of(self._usage_offset),
            output_cap=self._output_cap,
        )
        if token is None:
            return
        self._pending_hop_token = token
        for choice in model_response.choices:
            delta_fields: Mapping[str, object] = choice.delta.provider_specific_fields or {}
            choice.finish_reason = None
            if hasattr(choice, "provider_specific_fields"):
                delattr(choice, "provider_specific_fields")
            choice.delta.provider_specific_fields = {
                key: value for key, value in delta_fields.items() if key != "continuation_token"
            } or None

    def _start_hop(self, response: httpx.Response, token: str) -> None:
        if self._initial_prompt_tokens is None:
            self._initial_prompt_tokens = self._hop_usage.get("promptTokenCount")
        self._usage_offset = sum_usage_metadata(self._usage_offset, self._hop_usage)
        self._hop_usage = UsageMetadata()
        self._grounding_offset = (*self._grounding_offset, *self._hop_grounding)
        self._hop_grounding = ()
        self._answer_offset += self._hop_answer_bytes
        self._hop_answer_bytes = 0
        self._json_buffer = JSONFragmentAccumulator()
        self._unanswered_hop_token = token
        self.response = response

    def _truncated_stream_end(self, token: str) -> "ModelResponseStream":
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        if (images_end := self._unsent_thought_images_end()) is not None:
            self._token_after_unsent_images = token
            return images_end
        return ModelResponseStream(
            id=self._first_response_id,
            choices=[
                StreamingChoices(
                    index=0,
                    finish_reason="length",
                    delta=Delta(provider_specific_fields={"continuation_token": token}),
                )
            ],
        )

    def _end_after_unsent_images(self) -> Optional["ModelResponseStream"]:
        token: Final = self._token_after_unsent_images
        if token is None:
            return None
        self._token_after_unsent_images = None
        return self._truncated_stream_end(token)

    def _unsent_thought_images_before(self, stream_end: Exception) -> "ModelResponseStream":
        images_end: Final = self._unsent_thought_images_end()
        if images_end is None:
            raise stream_end
        self._stream_end_after_unsent_images = stream_end
        return images_end

    def _raise_after_unsent_images(self) -> None:
        stream_end: Final = self._stream_end_after_unsent_images
        if stream_end is None:
            return
        self._stream_end_after_unsent_images = None
        raise stream_end

    def _end_of_unanswered_hop(
        self, reason: Exception | str = "the continuation stream ended without a readable frame"
    ) -> Optional["ModelResponseStream"]:
        token: Final = self._unanswered_hop_token
        if token is None:
            return None
        self._unanswered_hop_token = None
        log_failed_hop(reason)
        return self._truncated_stream_end(token)

    async def _open_next_hop_async(self, token: str) -> Optional["ModelResponseStream"]:
        if not isinstance(self._resume, AsyncStreamResume):
            return None
        await self.aclose()
        try:
            hop_response: Final = await self._resume.open_hop(hop_request(self._resume.request_body, token))
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return self._truncated_stream_end(token)
        self._start_hop(hop_response, token)
        self.streaming_response = aiter_sse_lines(hop_response.aiter_text())
        self.async_response_iterator = self.streaming_response
        return None

    def _open_next_hop_sync(self, token: str) -> Optional["ModelResponseStream"]:
        if not isinstance(self._resume, SyncStreamResume):
            return None
        self.close()
        try:
            hop_response: Final = self._resume.open_hop(hop_request(self._resume.request_body, token))
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return self._truncated_stream_end(token)
        self._start_hop(hop_response, token)
        self.streaming_response = iter_sse_lines(hop_response.iter_text())
        self.response_iterator = self.streaming_response
        return None

    # Sync iterator
    def __iter__(self):
        self.response_iterator = self.streaming_response
        return self

    def handle_valid_json_chunk(self, chunk: str) -> Optional["ModelResponseStream"]:
        chunk = chunk.strip()
        try:
            json_chunk: Final = json.loads(chunk)

        except json.JSONDecodeError:
            # Switch to accumulation mode for partial JSON chunks
            # This can happen at any point due to network fragmentation, not just first chunk
            # See: https://github.com/BerriAI/litellm/issues/16562
            self.chunk_type = "accumulated_json"
            return self.handle_accumulated_json_chunk(chunk=chunk)

        if self.sent_first_chunk is False:
            self.sent_first_chunk = True

        return self.chunk_parser(chunk=json_chunk)

    def handle_accumulated_json_chunk(self, chunk: str, is_final: bool = False) -> Optional["ModelResponseStream"]:
        message: Final = (litellm.CustomStreamWrapper.strip_sse_data_from_chunk(chunk) or "").replace("\n\n", "")
        self._json_buffer.append(message)

        # Mid-stream, defer parsing until the buffer's last byte can close a value:
        # attempting a parse after every fragment of one large object is O(n^2) and
        # holds the GIL, freezing the event loop. At end of stream (is_final) no more
        # data is coming, so drain whatever complete values remain regardless of the
        # trailing byte, otherwise a complete leading value sitting behind a truncated
        # trailing one would be silently dropped.
        if not is_final and not self._json_buffer.could_close_json():
            return None

        # Peel one complete JSON value from the front of the buffer and keep the
        # unconsumed tail. Running json.loads over the whole buffer would fail
        # forever once it held more than one concatenated value ("Extra data") while
        # never resetting the buffer, so the buffer grew without bound and pinned the
        # core. pop_next_value reports where the value ended, so concatenated values
        # drain one call at a time. A leading non-dict value (never emitted by Gemini
        # in practice) is consumed and skipped so it cannot block the dict values
        # behind it.
        while True:
            found, decoded = self._json_buffer.pop_next_value()
            if not found:
                return None
            if isinstance(decoded, dict):
                return self.chunk_parser(chunk=decoded)

    def _common_chunk_parsing_logic(self, chunk: str) -> Optional["ModelResponseStream"]:
        try:
            chunk = litellm.CustomStreamWrapper.strip_sse_data_from_chunk(chunk) or ""
            if len(chunk) > 0:
                """
                Check if initial chunk valid json
                - if partial json -> enter accumulated json logic
                - if valid - continue
                """
                if self.chunk_type == "valid_json":
                    return self.handle_valid_json_chunk(chunk=chunk)
                elif self.chunk_type == "accumulated_json":
                    return self.handle_accumulated_json_chunk(chunk=chunk)

            return None
        except Exception:
            raise

    def __next__(self) -> Optional["ModelResponseStream"]:
        if (end_after_images := self._end_after_unsent_images()) is not None:
            return end_after_images
        self._raise_after_unsent_images()
        try:
            return self._next_chunk()
        except HOP_FAILURES as e:
            if (unanswered_end := self._end_of_unanswered_hop(e)) is None:
                return self._unsent_thought_images_before(e)
            self.close()
            return unanswered_end
        except (StopIteration, RuntimeError) as e:
            return self._unsent_thought_images_before(e)

    def _next_chunk(self) -> Optional["ModelResponseStream"]:
        while True:
            try:
                chunk = self.response_iterator.__next__()  # pyright: ignore[reportAttributeAccessIssue]  # sync stream
            except StopIteration:
                if self.chunk_type == "accumulated_json" and self._json_buffer:
                    result = self.handle_accumulated_json_chunk(chunk="", is_final=True)
                    if result is not None:
                        return result
                if (unanswered_end := self._end_of_unanswered_hop()) is not None:
                    return unanswered_end
                if (token := self._pending_hop_token) is None:
                    raise StopIteration
                self._pending_hop_token = None
                if (truncated_end := self._open_next_hop_sync(token)) is not None:
                    return truncated_end
                continue
            except ValueError as e:
                raise RuntimeError(f"Error receiving chunk from stream: {e}")

            try:
                return self._common_chunk_parsing_logic(chunk=chunk)
            except StopIteration:
                raise StopIteration
            except ValueError as e:
                raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()  # pyright: ignore[reportAttributeAccessIssue]  # async stream
        return self

    async def __anext__(self) -> Optional["ModelResponseStream"]:
        if (end_after_images := self._end_after_unsent_images()) is not None:
            return end_after_images
        self._raise_after_unsent_images()
        try:
            return await self._anext_chunk()
        except HOP_FAILURES as e:
            if (unanswered_end := self._end_of_unanswered_hop(e)) is None:
                return self._unsent_thought_images_before(e)
            await self.aclose()
            return unanswered_end
        except (StopAsyncIteration, RuntimeError) as e:
            return self._unsent_thought_images_before(e)

    async def _anext_chunk(self) -> Optional["ModelResponseStream"]:
        while True:
            try:
                chunk = await self.async_response_iterator.__anext__()
            except StopAsyncIteration:
                if self.chunk_type == "accumulated_json" and self._json_buffer:
                    result = self.handle_accumulated_json_chunk(chunk="", is_final=True)
                    if result is not None:
                        return result
                if (unanswered_end := self._end_of_unanswered_hop()) is not None:
                    return unanswered_end
                if (token := self._pending_hop_token) is None:
                    raise StopAsyncIteration
                self._pending_hop_token = None
                if (truncated_end := await self._open_next_hop_async(token)) is not None:
                    return truncated_end
                continue
            except ValueError as e:
                raise RuntimeError(f"Error receiving chunk from stream: {e}")

            try:
                return self._common_chunk_parsing_logic(chunk=chunk)
            except StopAsyncIteration:
                raise StopAsyncIteration
            except ValueError as e:
                raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    async def aclose(self) -> None:
        iterator: Final = getattr(  # pyright: ignore[reportUnknownVariableType]  # sync or async line iterator
            self,
            "async_response_iterator",
            self.streaming_response,
        )
        if iterator is not None and hasattr(iterator, "aclose"):
            try:
                await iterator.aclose()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # hasattr guard
            except Exception as e:  # noqa: BLE001
                verbose_logger.debug("ModelResponseIterator.aclose: error closing iterator: %s", e)
        if self.response is not None:
            try:
                await self.response.aclose()
            except Exception as e:  # noqa: BLE001
                verbose_logger.debug("ModelResponseIterator.aclose: error closing response: %s", e)

    def close(self) -> None:
        iterator: Final = getattr(self, "response_iterator", self.streaming_response)  # pyright: ignore[reportUnknownVariableType]  # sync or async line iterator
        if iterator is not None and hasattr(iterator, "close"):
            try:
                iterator.close()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # hasattr guard
            except Exception as e:  # noqa: BLE001
                verbose_logger.debug("ModelResponseIterator.close: error closing iterator: %s", e)
        if self.response is not None:
            try:
                self.response.close()
            except Exception as e:  # noqa: BLE001
                verbose_logger.debug("ModelResponseIterator.close: error closing response: %s", e)
