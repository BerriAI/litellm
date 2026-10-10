import asyncio, httpx, importlib, json, os
import sys
from typing import Final

import pytest
from pydantic import TypeAdapter

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import TOOL_RESULT_IMAGE_BOUNDARY
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config
from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig
from litellm.utils import _invalidate_model_cost_lowercase_map, get_optional_params
from datetime import datetime
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router import Router
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from respx import MockRouter
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from unittest.mock import AsyncMock

_MAPPED_PARAMS: Final = TypeAdapter(dict[str, object])
_SUPPORTED_PARAMS: Final = TypeAdapter(list[str])


class TestAzureOpenAIConfig:
    def test_is_response_format_supported_model(self):
        config = AzureOpenAIConfig()
        # New logic: Azure deployment names with suffixes and prefixes
        assert config._is_response_format_supported_model("azure/gpt-4.1-suffix")
        assert config._is_response_format_supported_model("gpt-4.1-suffix")
        assert config._is_response_format_supported_model("azure/gpt-4-1-suffix")
        assert config._is_response_format_supported_model("gpt-4-1-suffix")
        # 4o models (should always be supported)
        assert config._is_response_format_supported_model("gpt-4o")
        assert config._is_response_format_supported_model("azure/gpt-4o-custom")
        # Backwards compatibility: base names
        assert config._is_response_format_supported_model("gpt-4.1")
        assert config._is_response_format_supported_model("gpt-4-1")
        # Negative test: clearly unsupported model
        assert not config._is_response_format_supported_model("gpt-3.5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo")
        assert not config._is_response_format_supported_model("gpt-3-5-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo-suffix")
        assert not config._is_response_format_supported_model("gpt-35-turbo")

    def test_prompt_cache_key_supported(self):
        """Test that 'prompt_cache_key' is in supported params for Azure OpenAI chat completion models.

        OpenAI's Chat Completions API supports prompt_cache_key for cache routing optimization.
        """
        config = AzureOpenAIConfig()
        supported_params = config.get_supported_openai_params("gpt-4.1-nano")
        assert "prompt_cache_key" in supported_params

        supported_params = config.get_supported_openai_params("gpt-4.1")
        assert "prompt_cache_key" in supported_params


def test_map_openai_params_with_preview_api_version():
    config = AzureOpenAIConfig()
    non_default_params = {
        "response_format": {"type": "json_object"},
    }
    optional_params = {}
    model = "azure/gpt-4-1"
    drop_params = False
    api_version = "preview"
    assert config.map_openai_params(
        non_default_params, optional_params, model, drop_params, api_version
    )


def test_transform_request_hoists_tool_message_image():
    """Azure builds its request via convert_to_azure_openai_messages without the
    OpenAIGPTConfig._transform_messages pipeline, so transform_request must hoist
    tool-message images itself; Azure rejects non-text tool content."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    messages = [
        {"role": "user", "content": "read the screenshot"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
        },
    ]

    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    transformed = request["messages"]
    assert [m.get("role") for m in transformed] == ["user", "assistant", "tool", "user"]
    assert isinstance(transformed[2]["content"], str)
    assert transformed[3]["content"] == [
        {"type": "text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]


def test_transform_request_drops_tool_reference_parts():
    """Azure's transform_request shares the tool-message sanitizing with OpenAI:
    tool_reference parts are dropped, a reference-only result keeps its tool
    message with empty text (#37462 round trip)."""
    messages = [
        {"role": "user", "content": "load the WebFetch tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ToolSearch", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "tool_reference", "tool_name": "WebFetch"}],
        },
    ]

    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["messages"][2]["content"] == ""


@pytest.mark.parametrize(
    "enabled, expected", [(False, ("hi", "sys", "reply", "more")), (True, ("sys", "hi", "reply", "more"))]
)
def test_transform_request_system_messages_first_follows_global_flag(monkeypatch, enabled, expected):
    """Azure OpenAI shares OpenAI's prefix-matched prompt cache, so the same flag moves
    system messages ahead of the conversation on the Azure request body."""
    monkeypatch.setattr(litellm, "openai_system_messages_first", enabled)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "more"},
    ]

    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert tuple(m["content"] for m in request["messages"]) == expected
    assert [m["content"] for m in messages] == ["hi", "sys", "reply", "more"]


@pytest.mark.parametrize(
    "model, emitted_key, absent_key",
    [
        ("gpt-5-chat", "max_completion_tokens", "max_tokens"),
        ("gpt-5-chat-latest", "max_completion_tokens", "max_tokens"),
        ("gpt-5-chat-2025-08-07", "max_completion_tokens", "max_tokens"),
        ("gpt-5", "max_completion_tokens", "max_tokens"),
        ("o3-mini", "max_completion_tokens", "max_tokens"),
        ("gpt-4o", "max_tokens", "max_completion_tokens"),
    ],
)
def test_azure_max_tokens_rename_covers_gpt_5_chat_family(model: str, emitted_key: str, absent_key: str) -> None:
    """Azure rejects `max_tokens` for the whole gpt-5 name family, gpt-5-chat* included."""
    mapped: Final = _MAPPED_PARAMS.validate_python(
        get_optional_params(model=model, custom_llm_provider="azure", max_tokens=5)
    )
    assert mapped[emitted_key] == 5
    assert absent_key not in mapped


@pytest.mark.parametrize("model", ["gpt-5-chat", "gpt-5-chat-latest"])
def test_azure_gpt_5_chat_stays_off_the_reasoning_path(model: str) -> None:
    """https://github.com/BerriAI/litellm/issues/13781: gpt-5-chat* is a regular chat model."""
    mapped: Final = _MAPPED_PARAMS.validate_python(
        get_optional_params(
            model=model,
            custom_llm_provider="azure",
            max_tokens=5,
            temperature=0.3,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            stop=["stop"],
            logit_bias={"1": 1},
        )
    )
    supported: Final = _SUPPORTED_PARAMS.validate_python(
        litellm.get_supported_openai_params(model=model, custom_llm_provider="azure")
    )
    assert mapped["temperature"] == 0.3
    assert mapped["presence_penalty"] == 0.1
    assert mapped["frequency_penalty"] == 0.2
    assert mapped["stop"] == ["stop"]
    assert mapped["logit_bias"] == {"1": 1}
    assert "reasoning_effort" not in mapped
    assert "reasoning_effort" not in supported


def test_azure_gpt_5_takes_the_reasoning_path() -> None:
    """Positive control for the predicate split: gpt-5 still drops chat-only params."""
    mapped: Final = _MAPPED_PARAMS.validate_python(
        get_optional_params(
            model="gpt-5",
            custom_llm_provider="azure",
            presence_penalty=0.1,
            logit_bias={"1": 1},
            drop_params=True,
        )
    )
    supported: Final = _SUPPORTED_PARAMS.validate_python(
        litellm.get_supported_openai_params(model="gpt-5", custom_llm_provider="azure")
    )
    assert "presence_penalty" not in mapped
    assert "logit_bias" not in mapped
    assert "reasoning_effort" in supported


_ARTIFACT_FIELD_PATTERN: Final = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'


class TestAzureToolSchemaCombinatorFlattening:
    """
    Regression tests for LIT-6510: Azure's chat completions validator rejects
    tool parameters carrying a top-level anyOf/oneOf/allOf for every model
    family, so AzureOpenAIConfig.transform_request must flatten them.
    """

    @staticmethod
    def _anyof_tool():
        return {
            "type": "function",
            "function": {
                "name": "automation_update",
                "description": "Update an automation",
                "parameters": {
                    "type": "object",
                    "anyOf": [
                        {
                            "properties": {"id": {"type": "string"}, "enabled": {"type": "boolean"}},
                            "required": ["id", "enabled"],
                        },
                        {
                            "properties": {"id": {"type": "string"}, "schedule": {"type": "string"}},
                            "required": ["id", "schedule"],
                        },
                    ],
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        }

    def _transform(self, config, model, tools):
        return config.transform_request(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": tools},
            litellm_params={"custom_llm_provider": "azure"},
            headers={},
        )

    def test_transform_request_flattens_top_level_anyof(self):
        request = self._transform(AzureOpenAIConfig(), "gpt-4o", [self._anyof_tool()])
        parameters = request["tools"][0]["function"]["parameters"]
        assert "anyOf" not in parameters
        assert parameters["type"] == "object"
        assert set(parameters["properties"]) == {"id", "enabled", "schedule"}
        assert parameters["required"] == ["id"]
        assert request["tools"][0]["function"]["name"] == "automation_update"

    def test_gpt5_config_flattens_via_shared_transform(self):
        request = self._transform(AzureOpenAIGPT5Config(), "gpt-5.4-mini", [self._anyof_tool()])
        parameters = request["tools"][0]["function"]["parameters"]
        assert "anyOf" not in parameters
        assert set(parameters["properties"]) == {"id", "enabled", "schedule"}

    def test_caller_tool_dict_is_not_mutated(self):
        tool = self._anyof_tool()
        self._transform(AzureOpenAIConfig(), "gpt-4o", [tool])
        assert tool == self._anyof_tool()

    def test_transform_request_drops_non_python_regex_pattern(self):
        tool = {
            "type": "function",
            "function": {
                "name": "Artifact",
                "parameters": {
                    "type": "object",
                    "properties": {"field": {"type": "string", "pattern": _ARTIFACT_FIELD_PATTERN}},
                },
            },
        }

        request = self._transform(AzureOpenAIConfig(), "gpt-4o", [tool])

        assert request["tools"][0]["function"]["parameters"] == {
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }
        assert tool["function"]["parameters"]["properties"]["field"]["pattern"] == _ARTIFACT_FIELD_PATTERN

    def test_clean_object_schema_passes_through_as_same_object(self):
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
        }
        request = self._transform(AzureOpenAIConfig(), "gpt-4o", [tool])
        assert request["tools"][0] is tool

    def test_non_dict_tool_entries_pass_through_unchanged(self):
        request = self._transform(AzureOpenAIConfig(), "gpt-4o", ["not-a-tool"])
        assert request["tools"] == ["not-a-tool"]

    def test_request_without_tools_is_unchanged(self):
        request = AzureOpenAIConfig().transform_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"temperature": 0.2},
            litellm_params={"custom_llm_provider": "azure"},
            headers={},
        )
        assert "tools" not in request
        assert request["temperature"] == 0.2


@pytest.mark.parametrize("tool_choice", ["none", "auto"])
def test_azure_drops_tool_choice_without_tools_or_functions(tool_choice: str) -> None:
    optional_params = {"tool_choice": tool_choice, "temperature": 0.2}
    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert "tool_choice" not in request
    assert request["temperature"] == 0.2
    assert optional_params["tool_choice"] == tool_choice


def test_azure_tools_empty_drops_tool_choice() -> None:
    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"tools": [], "tool_choice": "auto"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["tools"] == []
    assert "tool_choice" not in request


def test_azure_functions_empty_drops_tool_choice() -> None:
    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"functions": [], "tool_choice": "none"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["functions"] == []
    assert "tool_choice" not in request


def test_azure_preserves_tool_choice_with_tools() -> None:
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"tools": tools, "tool_choice": "auto"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["tools"] == tools
    assert request["tool_choice"] == "auto"


def test_azure_preserves_tool_choice_with_legacy_functions() -> None:
    functions = [{"name": "get_weather", "parameters": {}}]
    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"functions": functions, "tool_choice": "auto"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["functions"] == functions
    assert request["tool_choice"] == "auto"


def test_azure_preserves_function_call_without_tools() -> None:
    request = AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"function_call": "none", "tool_choice": "auto"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["function_call"] == "none"
    assert "tool_choice" not in request


def test_azure_gpt5_drops_tool_choice_without_tools() -> None:
    request = AzureOpenAIGPT5Config().transform_request(
        model="gpt5_series/gpt-5.6-sol",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"tool_choice": "none"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["model"] == "gpt-5.6-sol"
    assert "tool_choice" not in request


@pytest.mark.asyncio
async def test_azure_async_transform_drops_tool_choice_without_tools() -> None:
    request = await AzureOpenAIConfig().async_transform_request(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"tool_choice": "none"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert "tool_choice" not in request


@pytest.mark.asyncio
async def test_azure_gpt5_async_transform_drops_tool_choice_without_tools() -> None:
    request = await AzureOpenAIGPT5Config().async_transform_request(
        model="gpt5_series/gpt-5.6-sol",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"tool_choice": "auto"},
        litellm_params={"custom_llm_provider": "azure"},
        headers={},
    )

    assert request["model"] == "gpt-5.6-sol"
    assert "tool_choice" not in request


def test_transform_request_strips_litellm_format_from_managed_file_id():
    import base64

    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_messages_with_model_file_ids,
    )

    managed_file_id: Final = base64.b64encode(
        b"litellm_proxy:application/pdf;unified_id,abc123;llm_output_file_id,assistant-xyz;target_model_names,azure-gpt"
    ).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this file"},
                {"type": "file", "file": {"file_id": managed_file_id}},
            ],
        }
    ]
    updated_messages = update_messages_with_model_file_ids(messages, None, {})

    request = AzureOpenAIConfig().transform_request(
        model="gpt-5.4",
        messages=updated_messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    file_part = request["messages"][0]["content"][1]["file"]
    assert "format" not in file_part
    assert file_part["file_id"] == "assistant-xyz"


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}

@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield

@pytest.fixture
def _pr4_azure_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "pr4-test-azure-openai-key")
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://azure-openai.example.invalid")
    monkeypatch.setenv("AZURE_TENANT_ID", "pr4-test-tenant-id")
    monkeypatch.setenv("AZURE_CLIENT_ID", "pr4-test-client-id")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "pr4-test-client-secret")

@pytest.mark.usefixtures(
    "_pr4_azure_openai_env",
    "_vcr_outcome_gate",
    "isolate_litellm_state",
    "setup_and_teardown",
)
@pytest.mark.asyncio()
@pytest.mark.respx()
async def test_aaaaazure_tenant_id_auth(respx_mock: MockRouter):
    """

    Tests when we set  tenant_id, client_id, client_secret they don't get sent with the request

    PROD Test
    """
    litellm.disable_aiohttp_transport = True  # since this uses respx, we need to set use_aiohttp_transport to False

    # Clear the HTTP client cache to ensure respx mocking works
    # This is critical because respx only intercepts clients created AFTER mocking is active
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                    "tenant_id": os.getenv("AZURE_TENANT_ID"),
                    "client_id": os.getenv("AZURE_CLIENT_ID"),
                    "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
                },
            },
        ],
    )

    mock_response = AsyncMock()
    obj = ChatCompletion(
        id="foo",
        model="gpt-4",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Hello world!",
                    role="assistant",
                ),
            )
        ],
        created=int(datetime.now().timestamp()),
    )
    litellm.set_verbose = True

    mock_request = respx_mock.post(url__regex=r".*/chat/completions.*").mock(
        return_value=httpx.Response(200, json=obj.model_dump(mode="json"))
    )

    await router.acompletion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])

    # Ensure all mocks were called
    respx_mock.assert_all_called()

    for call in mock_request.calls:
        print(call)
        print(call.request.content)

        json_body = json.loads(call.request.content)
        print(json_body)

        assert json_body == {
            "messages": [{"role": "user", "content": "Hello world!"}],
            "model": "gpt-4.1-mini",
            "stream": False,
        }


@pytest.mark.respx(assert_all_called=True)
def test_azure_chat_completion_preserves_safety_result(respx_mock: MockRouter) -> None:
    url: Final = "https://example-resource.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions"
    content_filter_results: Final = {
        "hate": {"filtered": False, "severity": "safe"},
        "violence": {"filtered": False, "severity": "safe"},
    }
    payload: Final = {
        "id": "chatcmpl-safety",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
                "content_filter_results": content_filter_results,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    route: Final = respx_mock.post(url__startswith=url).mock(return_value=httpx.Response(200, json=payload))
    response: Final = litellm.completion(
        model="azure/gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://example-resource.openai.azure.com",
        api_key="azure-test-key",
        api_version="2024-12-01-preview",
    )

    assert route.call_count == 1
    assert response.choices[0].message.content == "Hello"
    assert response.choices[0].provider_specific_fields == {"content_filter_results": content_filter_results}


@pytest.mark.respx(assert_all_called=True)
def test_azure_deployment_id_selects_chat_completions_endpoint(respx_mock: MockRouter) -> None:
    url: Final = "https://example-resource.openai.azure.com/openai/deployments/deployment-gpt-4o/chat/completions"
    payload: Final = {
        "id": "chatcmpl-deployment",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "deployment-gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    route: Final = respx_mock.post(url__startswith=url).mock(return_value=httpx.Response(200, json=payload))
    response: Final = litellm.completion(
        model="gpt-3.5-turbo",
        deployment_id="deployment-gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        api_base="https://example-resource.openai.azure.com",
        api_key="azure-test-key",
        api_version="2024-12-01-preview",
    )

    assert route.call_count == 1
    assert route.calls[0].request.url.path.endswith("/deployments/deployment-gpt-4o/chat/completions")
    assert response.choices[0].message.content == "Hello"
