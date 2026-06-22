import asyncio
from types import SimpleNamespace

from litellm.llms.antigravity2.chat.transformation import Antigravity2Config, Antigravity2SDK
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import ProviderConfigManager


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeBuiltinTools:
    @classmethod
    def none(cls):
        return []


class _FakeTypes:
    class ThinkingLevel:
        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    GenerationConfig = _Obj
    ModelEntry = _Obj
    ModelConfig = _Obj
    GeminiConfig = _Obj
    CapabilitiesConfig = _Obj
    BuiltinTools = _FakeBuiltinTools


class _FakeLocalAgentConfig(_Obj):
    pass


class _FakeResponse:
    usage_metadata = SimpleNamespace(
        prompt_token_count=3,
        candidates_token_count=2,
        total_token_count=5,
    )

    async def text(self):
        return "hello from ag2"

    def __aiter__(self):
        async def _tokens():
            yield "hel"
            yield "lo"

        return _tokens()


class _FakeAgent:
    last_config = None
    last_prompt = None

    def __init__(self, config):
        _FakeAgent.last_config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def chat(self, prompt):
        _FakeAgent.last_prompt = prompt
        return _FakeResponse()


class _FakeAntigravity2Config(Antigravity2Config):
    def _load_sdk(self):
        return Antigravity2SDK(
            Agent=_FakeAgent,
            LocalAgentConfig=_FakeLocalAgentConfig,
            types=_FakeTypes,
        )


def test_antigravity2_provider_config_registration():
    config = ProviderConfigManager.get_provider_chat_config("gemini-3.1-pro-preview", LlmProviders.ANTIGRAVITY2)

    assert isinstance(config, Antigravity2Config)


def test_build_local_agent_config_uses_official_sdk_contract_and_disables_tools(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY2_APP_DATA_DIR", "/srv/ag2")
    config = _FakeAntigravity2Config()
    sdk = config._load_sdk()

    local_config, prompt = config._build_local_agent_config(
        sdk=sdk,
        model="gemini-3.1-pro-preview",
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        optional_params={"reasoning_effort": "high"},
        api_key=None,
    )

    assert local_config.system_instructions == "be brief"
    assert prompt == "user: hi"
    assert local_config.model == "gemini-3.1-pro-preview"
    assert local_config.capabilities.enabled_tools == []
    assert local_config.capabilities.enable_subagents is False
    assert local_config.app_data_dir == "/srv/ag2"
    assert local_config.gemini_config.models.default.generation.thinking_level == "high"


def test_antigravity2_async_completion_maps_text_and_usage():
    config = _FakeAntigravity2Config()
    model_response = ModelResponse()

    result = asyncio.run(
        config._acompletion(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "say hello"}],
            model_response=model_response,
            optional_params={},
            api_key=None,
        )
    )

    assert _FakeAgent.last_prompt == "user: say hello"
    assert result.choices[0].message.content == "hello from ag2"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 5


def test_antigravity2_stream_yields_generic_chunks():
    config = _FakeAntigravity2Config()

    async def collect():
        return [
            chunk
            async for chunk in config._achat_stream(
                model="gemini-3.1-pro-preview",
                messages=[{"role": "user", "content": "stream"}],
                optional_params={},
                api_key=None,
            )
        ]

    chunks = asyncio.run(collect())

    assert [chunk["text"] for chunk in chunks] == ["hel", "lo", ""]
    assert chunks[-1]["is_finished"] is True
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["usage"].total_tokens == 5
