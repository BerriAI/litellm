import copy
import importlib
import inspect
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import pytest
from _pytest.mark.structures import ParameterSet

import litellm
from litellm.llms.elevenlabs.text_to_speech.transformation import ElevenLabsTextToSpeechConfig
from litellm.llms.meta.realtime.transformation import MUSE_MODEL
from litellm.llms.openai_like.dynamic_config import create_config_class, create_responses_config_class
from litellm.llms.openai_like.json_loader import SimpleProviderConfig

API_BASE_WITH_TRAILING_SLASH: Final = "https://gw.example.com/p/"
API_BASE_WITHOUT_TRAILING_SLASH: Final = API_BASE_WITH_TRAILING_SLASH.rstrip("/")
LLMS_ROOT: Final = Path(litellm.__file__).parent / "llms"
PACKAGE_ROOT: Final = LLMS_ROOT.parent.parent

ARGUMENTS_BY_NAME: Final[Mapping[str, object]] = MappingProxyType(
    {
        "api_key": "sk-test",
        "model": "test-model",
        "optional_params": {},
        "litellm_params": {"api_key": "sk-test", "vertex_project": "project", "vertex_location": "us-central1"},
        "stream": False,
        "endpoint": "v1/messages",
        "request_query_params": None,
        "data": None,
        "api_version": "2024-01-01",
        "vector_store_id": "vs_test",
    }
)


SAMPLE_JSON_PROVIDER: Final = SimpleProviderConfig(
    "json-provider", {"base_url": "https://json-provider.example.com", "api_key_env": "UNSET_KEY"}
)
FACTORY_BUILDERS: Final[tuple[type, ...]] = (
    create_config_class(SAMPLE_JSON_PROVIDER),
    create_responses_config_class(SAMPLE_JSON_PROVIDER),
)


async def _resolve_access_token() -> str:
    return "token"


@dataclass(frozen=True, slots=True)
class BuilderInputs:
    constructor: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    call: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    api_base_option: str | None = None


INPUTS_BY_BUILDER: Final[Mapping[str, BuilderInputs]] = MappingProxyType(
    {
        "anthropic.skills.transformation.AnthropicSkillsConfig": BuilderInputs(call={"endpoint": "skills"}),
        "azure_ai.image_edit.transformation.AzureFoundryFluxImageEditConfig": BuilderInputs(
            call={"litellm_params": {"api_version": "2025-04-01-preview"}}
        ),
        "bedrock.chat.agentcore.transformation.AmazonAgentCoreConfig": BuilderInputs(
            call={"model": "agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-runtime"},
            api_base_option="aws_bedrock_runtime_endpoint",
        ),
        "bedrock.chat.invoke_agent.transformation.AmazonInvokeAgentConfig": BuilderInputs(
            call={"model": "agent/AGENT_ID/ALIAS_ID", "optional_params": {"sessionID": "session-id"}}
        ),
        "black_forest_labs.image_edit.transformation.BlackForestLabsImageEditConfig": BuilderInputs(
            call={"model": "flux-kontext-pro"}
        ),
        "black_forest_labs.image_generation.transformation.BlackForestLabsImageGenerationConfig": BuilderInputs(
            call={"model": "flux-pro-1.1"}
        ),
        "elevenlabs.text_to_speech.transformation.ElevenLabsTextToSpeechConfig": BuilderInputs(
            call={"litellm_params": {ElevenLabsTextToSpeechConfig.ELEVENLABS_VOICE_ID_KEY: "voice-id"}}
        ),
        "meta.realtime.transformation.MetaRealtimeConfig": BuilderInputs(call={"model": MUSE_MODEL}),
        "openai.evals.transformation.OpenAIEvalsConfig": BuilderInputs(call={"endpoint": "evals"}),
        "openai_like.messages.transformation.JSONProviderAnthropicMessagesConfig": BuilderInputs(
            constructor={"provider": SAMPLE_JSON_PROVIDER}
        ),
        "predibase.chat.transformation.PredibaseConfig": BuilderInputs(
            call={"litellm_params": {"predibase_tenant_id": "tenant"}}
        ),
        "ragflow.chat.transformation.RAGFlowConfig": BuilderInputs(call={"model": "ragflow/chat/chat-id/model"}),
        "scaledown.chat.transformation.ScaleDownChatConfig": BuilderInputs(call={"model": "classify"}),
        "vertex_ai.audio_transcription.realtime_transformation.VertexChirpRealtimeConfig": BuilderInputs(
            constructor={"resolve_access_token": _resolve_access_token, "project": "project", "location": None},
            call={"model": "chirp_3"},
        ),
        "vertex_ai.interactions.transformation.VertexAIInteractionsConfig": BuilderInputs(
            constructor={"mint_access_token": lambda _credentials, _project: ("token", "project")}
        ),
        "vertex_ai.realtime.transformation.VertexAIRealtimeConfig": BuilderInputs(
            constructor={"access_token": "token", "project": "project", "location": "us-central1"}
        ),
        "vertex_ai.text_to_speech.transformation.VertexAILyriaTextToSpeechConfig": BuilderInputs(
            call={"model": "lyria-002"}
        ),
    }
)

UNCALLABLE_BUILDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "base_llm.search.transformation.BaseSearchConfig": "raises-NotImplementedError-for-providers-to-override",
        "bedrock.image_edit.amazon_nova_canvas_image_edit_transformation.BedrockAmazonNovaCanvasImageEditConfig": (
            "raises-because-BedrockImageEdit-builds-the-url"
        ),
        "sap.chat.transformation.GenAIHubOrchestrationConfig": "url-comes-from-a-live-SAP-deployment-lookup",
        "sap.embed.transformation.GenAIHubEmbeddingConfig": "url-comes-from-a-live-SAP-deployment-lookup",
        "triton.completion.transformation.TritonConfig": "api_base-must-be-the-full-generate-or-infer-url",
        "vertex_ai.rerank.transformation.VertexAIRerankConfig": "ignores-api_base-and-needs-Vertex-credentials",
    }
)


API_BASE_IS_THE_FULL_URL: Final[Mapping[str, str]] = MappingProxyType(
    {
        builder_id: "sends-the-configured-api_base-as-the-request-url"
        for builder_id in (
            "base_llm.audio_transcription.transformation.BaseAudioTranscriptionConfig",
            "base_llm.chat.transformation.BaseConfig",
            "base_llm.completion.transformation.BaseTextCompletionConfig",
            "base_llm.embedding.transformation.BaseEmbeddingConfig",
            "base_llm.image_generation.transformation.BaseImageGenerationConfig",
            "base_llm.image_variations.transformation.BaseImageVariationConfig",
            "base_llm.vector_store.transformation.BaseDirectVectorStoreConfig",
            "bedrock.realtime.transformation.BedrockRealtimeConfig",
            "bedrock.search.transformation.AgentCoreSearchConfig",
            "brave.search.transformation.BraveSearchConfig",
            "cohere.chat.v2_transformation.CohereV2ChatConfig",
            "cohere.embed.transformation.CohereEmbeddingConfig",
            "dashscope.image_generation.transformation.DashScopeImageGenerationConfig",
            "dataforseo.search.transformation.DataForSEOSearchConfig",
            "datarobot.chat.transformation.DataRobotConfig",
            "duckduckgo.search.transformation.DuckDuckGoSearchConfig",
            "google_pse.search.transformation.GooglePSESearchConfig",
            "searchapi.search.transformation.SearchAPIConfig",
            "tinyfish.search.transformation.TinyfishSearchConfig",
            "vertex_ai.text_to_speech.transformation.VertexAITextToSpeechConfig",
            "vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation."
            "VertexAIPartnerModelsAnthropicMessagesConfig",
        )
    }
)


HOST_NAMED_LIKE_A_STRIPPED_SUFFIX_IS_NOT_KEPT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "deepinfra.rerank.transformation.DeepinfraRerankConfig": "removes-every-openai-substring-on-main-too",
        "minimax.videos.transformation.MinimaxVideoConfig": "strips-surface-suffixes-to-the-scheme-on-main-too",
        "volcengine.videos.transformation.VolcEngineVideoConfig": "strips-surface-suffixes-to-the-scheme-on-main-too",
        "vertex_ai.audio_transcription.realtime_transformation.VertexChirpRealtimeConfig": "returns-a-grpc-host",
        "azure.realtime.http_transformation.AzureRealtimeHTTPConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "bedrock.responses.transformation.BedrockOpenAIResponsesConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "bedrock_mantle.chat.claude_transformation.BedrockMantleClaudeChatConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "bedrock_mantle.messages.transformation.BedrockMantleAnthropicMessagesConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "bedrock_mantle.responses.transformation.BedrockMantleResponsesAPIConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "deepseek.messages.transformation.DeepSeekAnthropicMessagesConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "ragflow.chat.transformation.RAGFlowConfig": "replaces-the-host-with-a-route-segment-on-main-too",
        "scaledown.chat.transformation.ScaleDownChatConfig": "replaces-the-host-with-a-route-segment-on-main-too",
    }
)


def _builder_id(cls: type) -> str:
    return f"{cls.__module__.removeprefix('litellm.llms.')}.{cls.__qualname__}"


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)


def _builder_modules() -> tuple[ModuleType, ...]:
    builder_paths: Final = tuple(path for path in LLMS_ROOT.rglob("*.py") if "def get_complete_url" in path.read_text())
    return tuple(importlib.import_module(_module_name(path)) for path in sorted(builder_paths))


def _defined_builders(module: ModuleType) -> tuple[type, ...]:
    classes: Final = (*(cls for _, cls in inspect.getmembers(module, inspect.isclass)), *FACTORY_BUILDERS)
    return tuple(cls for cls in classes if cls.__module__ == module.__name__ and "get_complete_url" in vars(cls))


def _all_builders() -> tuple[type, ...]:
    builders: Final = itertools.chain.from_iterable(_defined_builders(module) for module in _builder_modules())
    return tuple(sorted(builders, key=_builder_id))


def _skip_reason(cls: type) -> str | None:
    if getattr(vars(cls)["get_complete_url"], "__isabstractmethod__", False):
        return "abstract-get_complete_url"
    return UNCALLABLE_BUILDERS.get(_builder_id(cls))


def _builder_param(cls: type) -> ParameterSet:
    reason: Final = _skip_reason(cls)
    if reason is None:
        return pytest.param(cls, id=_builder_id(cls))
    return pytest.param(cls, id=f"{_builder_id(cls)}-skipped-{reason}", marks=pytest.mark.skip(reason=reason))


def _unimplemented(*_args: object, **_kwargs: object) -> object:
    raise NotImplementedError


def _instantiable(cls: type) -> type:
    abstract_methods: Final[frozenset[str]] = getattr(cls, "__abstractmethods__", frozenset())
    if not abstract_methods:
        return cls
    return type(cls.__name__, (cls,), dict.fromkeys(abstract_methods, _unimplemented))


def _call_arguments(builder: object, inputs: BuilderInputs, api_base: str) -> dict[str, object]:
    parameters: Final = inspect.signature(getattr(builder, "get_complete_url")).parameters.values()
    named: Final = tuple(p for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
    api_base_options: Final = (
        {} if inputs.api_base_option is None else {"optional_params": {inputs.api_base_option: api_base}}
    )
    values: Final = {**ARGUMENTS_BY_NAME, "api_base": api_base, **inputs.call, **api_base_options}
    missing: Final = tuple(p.name for p in named if p.default is p.empty and p.name not in values)
    assert not missing, f"add a value for {missing} to ARGUMENTS_BY_NAME or INPUTS_BY_BUILDER"
    return {p.name: copy.deepcopy(values[p.name]) for p in named if p.name in values}


def _built_url(cls: type, api_base: str) -> str:
    inputs: Final = INPUTS_BY_BUILDER.get(_builder_id(cls), BuilderInputs())
    builder: Final = _instantiable(cls)(**inputs.constructor)
    result: Final = getattr(builder, "get_complete_url")(**_call_arguments(builder, inputs, api_base))
    return str(result[0] if isinstance(result, tuple) else result)


def _without_one_trailing_path_slash(url: str) -> str:
    parts: Final = urlsplit(url)
    return urlunsplit(parts._replace(path=parts.path.removesuffix("/")))


@pytest.mark.parametrize("builder_class", [_builder_param(cls) for cls in _all_builders()])
def test_get_complete_url_keeps_one_slash_after_a_trailing_slash_api_base(builder_class: type) -> None:
    url: Final = _built_url(builder_class, API_BASE_WITH_TRAILING_SLASH)

    assert "//" not in urlsplit(url).path, url


@pytest.mark.parametrize("builder_class", [_builder_param(cls) for cls in _all_builders()])
def test_get_complete_url_sends_the_same_url_with_or_without_a_trailing_slash(builder_class: type) -> None:
    with_slash: Final = _built_url(builder_class, API_BASE_WITH_TRAILING_SLASH)
    without_slash: Final = _built_url(builder_class, API_BASE_WITHOUT_TRAILING_SLASH)

    if _builder_id(builder_class) in API_BASE_IS_THE_FULL_URL:
        assert _without_one_trailing_path_slash(with_slash) == _without_one_trailing_path_slash(without_slash)
    else:
        assert with_slash == without_slash


def _host_preserving_builder_param(cls: type) -> ParameterSet:
    reason: Final = HOST_NAMED_LIKE_A_STRIPPED_SUFFIX_IS_NOT_KEPT.get(_builder_id(cls))
    if reason is None:
        return _builder_param(cls)
    return pytest.param(cls, id=f"{_builder_id(cls)}-skipped-{reason}", marks=pytest.mark.skip(reason=reason))


@pytest.mark.parametrize("api_base", ["https://v1", "https://api/v1", "https://openai/v1/", "https://anthropic/v1"])
@pytest.mark.parametrize("builder_class", [_host_preserving_builder_param(cls) for cls in _all_builders()])
def test_get_complete_url_keeps_a_host_named_like_the_suffix_it_strips(builder_class: type, api_base: str) -> None:
    url: Final = _built_url(builder_class, api_base)
    follows_the_api_base_host: Final = (
        urlsplit(_built_url(builder_class, "https://gateway.test/v1")).netloc == "gateway.test"
    )

    assert urlsplit(url).netloc, url
    assert not follows_the_api_base_host or urlsplit(url).netloc == urlsplit(api_base).netloc, url
