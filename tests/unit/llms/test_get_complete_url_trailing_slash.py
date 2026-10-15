import copy
import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final
from urllib.parse import urlsplit

import pytest
from _pytest.mark.structures import ParameterSet

import litellm
from litellm.llms.elevenlabs.text_to_speech.transformation import ElevenLabsTextToSpeechConfig
from litellm.llms.meta.realtime.transformation import MUSE_MODEL
from litellm.llms.openai_like.dynamic_config import create_config_class, create_responses_config_class
from litellm.llms.openai_like.json_loader import SimpleProviderConfig

API_BASE_WITH_TRAILING_SLASH: Final = "https://gw.example.com/p/"
LLMS_ROOT: Final = Path(litellm.__file__).parent / "llms"
PACKAGE_ROOT: Final = LLMS_ROOT.parent.parent

ARGUMENTS_BY_NAME: Final[Mapping[str, object]] = MappingProxyType(
    {
        "api_base": API_BASE_WITH_TRAILING_SLASH,
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


INPUTS_BY_BUILDER: Final[Mapping[str, BuilderInputs]] = MappingProxyType(
    {
        "anthropic.skills.transformation.AnthropicSkillsConfig": BuilderInputs(call={"endpoint": "skills"}),
        "azure_ai.image_edit.transformation.AzureFoundryFluxImageEditConfig": BuilderInputs(
            call={"litellm_params": {"api_version": "2025-04-01-preview"}}
        ),
        "bedrock.chat.agentcore.transformation.AmazonAgentCoreConfig": BuilderInputs(
            call={
                "model": "agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-runtime",
                "optional_params": {"aws_bedrock_runtime_endpoint": API_BASE_WITH_TRAILING_SLASH},
            }
        ),
        "bedrock.chat.invoke_agent.transformation.AmazonInvokeAgentConfig": BuilderInputs(
            call={"model": "agent/AGENT_ID/ALIAS_ID"}
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
    builders: Final = (cls for module in _builder_modules() for cls in _defined_builders(module))
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


def _call_arguments(builder: object, inputs: BuilderInputs) -> dict[str, object]:
    parameters: Final = inspect.signature(getattr(builder, "get_complete_url")).parameters.values()
    named: Final = tuple(p for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
    values: Final = {**ARGUMENTS_BY_NAME, **inputs.call}
    missing: Final = tuple(p.name for p in named if p.default is p.empty and p.name not in values)
    assert not missing, f"add a value for {missing} to ARGUMENTS_BY_NAME or INPUTS_BY_BUILDER"
    return {p.name: copy.deepcopy(values[p.name]) for p in named if p.name in values}


def _built_url(cls: type) -> str:
    inputs: Final = INPUTS_BY_BUILDER.get(_builder_id(cls), BuilderInputs())
    builder: Final = _instantiable(cls)(**inputs.constructor)
    result: Final = getattr(builder, "get_complete_url")(**_call_arguments(builder, inputs))
    return str(result[0] if isinstance(result, tuple) else result)


@pytest.mark.parametrize("builder_class", [_builder_param(cls) for cls in _all_builders()])
def test_get_complete_url_keeps_one_slash_after_a_trailing_slash_api_base(builder_class: type) -> None:
    url: Final = _built_url(builder_class)

    assert "//" not in urlsplit(url).path, url
