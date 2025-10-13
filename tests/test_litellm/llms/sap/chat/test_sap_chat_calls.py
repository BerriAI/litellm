# Python
import sys
import types
import contextlib
from dataclasses import dataclass, asdict
from typing import List

import pytest


@pytest.fixture(autouse=True, scope="function")
def fake_gen_ai_hub_modules(monkeypatch):
    """
    Injection of fake gen_ai_hub modules
    """

    # ----- proxy.gen_ai_hub_proxy -----
    proxy_mod = types.ModuleType("gen_ai_hub.proxy.gen_ai_hub_proxy")

    class GenAIHubProxyClient:
        def __init__(self):
            self.request_header = {"X-Fake-Header": "test"}

        def select_deployment(self, model_name: str):
            return type("Dep", (), {"url": "https://fake.sap.ai/v2/v1"})

    @contextlib.contextmanager
    def temporary_headers_addition(headers: dict):
        yield

    proxy_mod.GenAIHubProxyClient = GenAIHubProxyClient
    proxy_mod.temporary_headers_addition = temporary_headers_addition

    # ----- orchestration.exceptions -----
    exc_mod = types.ModuleType("gen_ai_hub.orchestration.exceptions")

    class OrchestrationError(Exception):
        def __init__(self, code=500, message="error"):
            self.code = code
            self.message = message
            super().__init__(message)

    exc_mod.OrchestrationError = OrchestrationError

    # ----- orchestration.models.* -----
    models_config_mod = types.ModuleType("gen_ai_hub.orchestration.models.config")
    models_message_mod = types.ModuleType("gen_ai_hub.orchestration.models.message")
    models_multimodal_items_mod = types.ModuleType(
        "gen_ai_hub.orchestration.models.multimodal_items"
    )
    models_llm_mod = types.ModuleType("gen_ai_hub.orchestration.models.llm")
    models_template_mod = types.ModuleType("gen_ai_hub.orchestration.models.template")
    models_response_format_mod = types.ModuleType(
        "gen_ai_hub.orchestration.models.response_format"
    )
    models_tools_mod = types.ModuleType("gen_ai_hub.orchestration.models.tools")
    native_openai_clients_mod = types.ModuleType(
        "gen_ai_hub.proxy.native.openai.clients"
    )
    dacite_mod = types.ModuleType("dacite")

    @dataclass
    class Message:
        role: str
        content: str

    @dataclass
    class TextPart:
        text: str
        type: str = "text"

        def to_dict(self):
            return {
                "type": self.type,
                "text": self.text,
            }

    @dataclass
    class ImageUrl:
        url: str

    @dataclass
    class ImagePart:
        image_url: ImageUrl
        type: str = "image_url"

        def to_dict(self):
            base = {
                "type": self.type,
                "image_url": {
                    "url": self.image_url.url,
                },
            }

            return base

    @dataclass
    class ToolMessage:
        tool_call_id: str
        content: object

    @dataclass
    class FunctionCall:
        name: str
        arguments: str

    @dataclass
    class MessageToolCall:
        id: str
        type: str
        function: FunctionCall

    @dataclass
    class LLM:
        name: str
        parameters: dict
        version: str = "latest"

    @dataclass
    class Template:
        messages: List[Message]
        response_format: object = None
        tools: list = None

    @dataclass
    class OrchestrationConfig:
        template: Template
        llm: LLM

        def to_dict(self):
            return {
                "template": {
                    "messages": [asdict(m) for m in self.template.messages],
                    "response_format": self.template.response_format,
                    "tools": self.template.tools or [],
                },
                "llm": asdict(self.llm),
            }

    class ResponseFormatJsonSchema(dict):
        pass

    class ResponseFormatJsonObject(dict):
        pass

    class ResponseFormatText(str):
        pass

    @dataclass
    class FunctionTool:
        name: str
        description: str = ""

    def _from_dict(data_class, data):
        if data_class.__name__ == "MessageToolCall":
            fn = data.get("function") or {}
            return MessageToolCall(
                id=data.get("id", ""),
                type=data.get("type", "function"),
                function=FunctionCall(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "")
                    if isinstance(fn.get("arguments", ""), str)
                    else "",
                ),
            )
        return data_class(**data)

    models_config_mod.OrchestrationConfig = OrchestrationConfig
    models_message_mod.Message = Message
    models_message_mod.ToolMessage = ToolMessage
    models_message_mod.MessageToolCall = MessageToolCall
    models_message_mod.FunctionCall = FunctionCall
    models_multimodal_items_mod.TextPart = TextPart
    models_multimodal_items_mod.ImagePart = ImagePart
    models_multimodal_items_mod.ImageUrl = ImageUrl
    models_llm_mod.LLM = LLM
    models_template_mod.Template = Template
    models_response_format_mod.ResponseFormatJsonSchema = ResponseFormatJsonSchema
    models_response_format_mod.ResponseFormatJsonObject = ResponseFormatJsonObject
    models_response_format_mod.ResponseFormatText = ResponseFormatText
    models_tools_mod.FunctionTool = FunctionTool
    native_openai_clients_mod.DEFAULT_API_VERSION = "2024-05-01-preview"
    dacite_mod.from_dict = _from_dict

    # ----- orchestration.service -----
    service_mod = types.ModuleType("gen_ai_hub.orchestration.service")

    @dataclass
    class _FakeLLMUsage:
        prompt_tokens: int = 4
        completion_tokens: int = 3
        total_tokens: int = 7

    @dataclass
    class _FakeLLMChoiceMessage:
        role: str = "assistant"
        content: str = "Hello from SAP!"

    @dataclass
    class _FakeLLMChoice:
        index: int
        message: _FakeLLMChoiceMessage
        finish_reason: str = "stop"

    @dataclass
    class _FakeLLMChoiceSreamDelta:
        content: str

    @dataclass
    class _FakeLLMStreamChoice:
        index: int
        delta: _FakeLLMChoiceSreamDelta
        finish_reason: str = None

    @dataclass
    class _FakeOrchestrationResult:
        choices: List[_FakeLLMStreamChoice]

    @dataclass
    class _FakeLLMStreamResult:
        orchestration_result: _FakeOrchestrationResult

    class _FakeLLMResult:
        def __init__(self):
            self.choices = [_FakeLLMChoice(index=0, message=_FakeLLMChoiceMessage())]
            self.created = 1711111111
            self.model = "gpt-4o"
            self.usage = _FakeLLMUsage()

    class _FakeResponse:
        def __init__(self):
            self.module_results = type("MR", (), {"llm": _FakeLLMResult()})

    class OrchestrationService:
        def __init__(self, proxy_client=None):
            if proxy_client is None:
                proxy_client = GenAIHubProxyClient()
            self.proxy_client = proxy_client
            self.api_url = "https://fake.sap.ai"  # для логов

        def run(self, config):
            return _FakeResponse()

        def stream(self, config):
            return iter(
                [
                    _FakeLLMStreamResult(
                        orchestration_result=_FakeOrchestrationResult(
                            choices=[
                                _FakeLLMStreamChoice(
                                    index=0,
                                    delta=_FakeLLMChoiceSreamDelta(content="Hello "),
                                )
                            ]
                        )
                    ),
                    _FakeLLMStreamResult(
                        orchestration_result=_FakeOrchestrationResult(
                            choices=[
                                _FakeLLMStreamChoice(
                                    index=0,
                                    delta=_FakeLLMChoiceSreamDelta(content="SAP!"),
                                    finish_reason="stop",
                                )
                            ]
                        )
                    ),
                ]
            )

        async def arun(self, config):
            return _FakeResponse()

        async def astream(self, config):
            async def agen():
                yield _FakeLLMStreamResult(
                    orchestration_result=_FakeOrchestrationResult(
                        choices=[
                            _FakeLLMStreamChoice(
                                index=0,
                                delta=_FakeLLMChoiceSreamDelta(content="Hello "),
                            )
                        ]
                    )
                )
                yield _FakeLLMStreamResult(
                    orchestration_result=_FakeOrchestrationResult(
                        choices=[
                            _FakeLLMStreamChoice(
                                index=0,
                                delta=_FakeLLMChoiceSreamDelta(content="SAP!"),
                                finish_reason="stop",
                            )
                        ]
                    )
                )

            return agen()

    service_mod.OrchestrationService = OrchestrationService

    # ----- registration in sys.modules -----
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.gen_ai_hub_proxy", proxy_mod)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.orchestration.exceptions", exc_mod)
    monkeypatch.setitem(
        sys.modules, "gen_ai_hub.orchestration.models.config", models_config_mod
    )
    monkeypatch.setitem(
        sys.modules, "gen_ai_hub.orchestration.models.message", models_message_mod
    )
    monkeypatch.setitem(
        sys.modules,
        "gen_ai_hub.orchestration.models.multimodal_items",
        models_multimodal_items_mod,
    )
    monkeypatch.setitem(
        sys.modules, "gen_ai_hub.orchestration.models.llm", models_llm_mod
    )
    monkeypatch.setitem(
        sys.modules, "gen_ai_hub.orchestration.models.template", models_template_mod
    )
    monkeypatch.setitem(
        sys.modules,
        "gen_ai_hub.orchestration.models.response_format",
        models_response_format_mod,
    )
    monkeypatch.setitem(
        sys.modules, "gen_ai_hub.orchestration.models.tools", models_tools_mod
    )
    monkeypatch.setitem(sys.modules, "gen_ai_hub.orchestration.service", service_mod)
    monkeypatch.setitem(
        sys.modules, "gen_ai_hub.proxy.native.openai.clients", native_openai_clients_mod
    )
    monkeypatch.setitem(sys.modules, "dacite", dacite_mod)

    import importlib

    for name in [
        "litellm.llms.sap.chat.transformation",
        "litellm.llms.sap.chat.handler",
        "litellm.llms.sap.embed.transformation",
        "litellm.llms.sap.embedding.transformation",
    ]:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        mod = sys.modules.get(name)
        if mod is not None:
            monkeypatch.setattr(mod, "_gen_ai_hub_import_error", None, raising=False)

    yield


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_sap_chat_with_fake_gen_ai_hub(
    fake_gen_ai_hub_modules, monkeypatch, sync_mode
):
    import litellm

    model = "sap/gpt-4o"
    messages = [{"role": "user", "content": "Hello"}]
    if sync_mode:
        response = litellm.completion(model=model, messages=messages)
    else:
        response = await litellm.acompletion(model=model, messages=messages)

    assert response.choices[0].message.content == "Hello from SAP!"
    assert response.model == "gpt-4o"
    assert response.usage.total_tokens == 7


def test_sap_streaming_with_fake_gen_ai_hub(fake_gen_ai_hub_modules, monkeypatch):
    import litellm

    stream = litellm.completion(
        model="sap/gpt-4o",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
    )

    full = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None) or ""
        full += delta

    assert full == "Hello SAP!"


@pytest.mark.asyncio
async def test_sap_astreaming_with_fake_gen_ai_hub(
    fake_gen_ai_hub_modules, monkeypatch
):
    import litellm

    stream = await litellm.acompletion(
        model="sap/gpt-4o",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
    )

    full = ""
    async for part in stream:
        delta = getattr(part.choices[0].delta, "content", None) or ""
        full += delta

    assert full == "Hello SAP!"
