"""Unit tests for litellm.proxy.guardrails.auto_router_shunt."""

import json
from typing import Any

import pytest

from litellm.proxy.guardrails.auto_router_shunt import (
    BULK_READ_TOOL_NAME,
    CODE_WRITE_TOOL_NAME,
    ShuntConfig,
    ShuntGuardrail,
    shunt_config_for_model,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

# A real UserAPIKeyAuth is required once a request actually reaches the rewrite path: it mints a
# capability token identifying the caller (auto_router_shunt.py's _mint_caller_capability_token),
# which needs a real api_key hash to seal. `None` still works for every test that stays on the
# unarmed/unchanged path, since that path returns before ever touching user_api_key_dict.
_FAKE_USER_API_KEY_DICT = UserAPIKeyAuth(api_key="fakehash1234567890")


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    """Minting a capability token needs a signing key; see shunt_capability_token.py."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234-test-salt-key")


class _FakeRouter:
    """Minimal stand-in for litellm.Router.get_model_list, mirroring
    test_auto_router_compression.py's _FakeRouter."""

    def __init__(self, deployments: list[dict[str, Any]]):
        self._deployments = deployments

    def get_model_list(self, model_name, team_id=None):
        return [d for d in self._deployments if d.get("model_name") == model_name]


def _marker(shunt_fields: dict[str, Any], tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_name": "shunt",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_default_model": "claude-haiku-4-5",
            **shunt_fields,
            **({"tags": tags} if tags is not None else {}),
        },
    }


def _tiered_marker(shunt_fields: dict[str, Any], simple: list[str] | None) -> dict[str, Any]:
    """A marker with a SIMPLE tier and no default model, to isolate the tier fallback."""
    return {
        "model_name": "shunt",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {"SIMPLE": simple} if simple is not None else {}},
            **shunt_fields,
        },
    }


# Regression: an unset worker model fell through to "" when the marker had no default model, so
# the delegated call ran against an empty model name. The UI and preset docs promise SIMPLE.
class TestWorkerModelFallback:
    def test_unset_worker_models_fall_back_to_the_simple_tier(self):
        router = _FakeRouter([_tiered_marker({"auto_router_shunt_min_lines": 350}, ["claude-haiku-4-5"])])
        config = shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=())
        assert config.bulk_read_model == "claude-haiku-4-5"
        assert config.code_write_model == "claude-haiku-4-5"

    def test_simple_tier_wins_over_the_default_model(self):
        marker = _marker({"auto_router_shunt_min_lines": 350})
        marker["litellm_params"]["complexity_router_config"] = {"tiers": {"SIMPLE": ["gpt-5.6-luna"]}}
        config = shunt_config_for_model(
            llm_router=_FakeRouter([marker]), model_alias="shunt", team_id=None, request_tags=()
        )
        assert config.bulk_read_model == "gpt-5.6-luna"

    def test_no_tier_and_no_default_model_is_unarmed_rather_than_empty(self):
        router = _FakeRouter([_tiered_marker({"auto_router_shunt_min_lines": 350}, None)])
        assert shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=()) is None

    def test_empty_tier_list_and_no_default_model_is_unarmed(self):
        router = _FakeRouter([_tiered_marker({"auto_router_shunt_min_lines": 350}, [])])
        assert shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=()) is None


class TestShuntConfigForModel:
    def test_no_router_returns_none(self):
        assert shunt_config_for_model(llm_router=None, model_alias="shunt", team_id=None, request_tags=()) is None

    def test_no_marker_deployment_returns_none(self):
        router = _FakeRouter([{"model_name": "shunt", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
        assert shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=()) is None

    def test_marker_without_min_lines_is_unarmed(self):
        router = _FakeRouter([_marker({})])
        assert shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=()) is None

    def test_armed_marker_falls_back_to_default_model_for_worker_models(self):
        router = _FakeRouter([_marker({"auto_router_shunt_min_lines": 350})])
        config = shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=())
        assert config == ShuntConfig(
            min_lines=350, bulk_read_model="claude-haiku-4-5", code_write_model="claude-haiku-4-5"
        )

    def test_explicit_worker_models_override_the_default(self):
        router = _FakeRouter(
            [
                _marker(
                    {
                        "auto_router_shunt_min_lines": 200,
                        "auto_router_shunt_bulk_read_model": "gpt-5.6-luna",
                    }
                )
            ]
        )
        config = shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=())
        assert config.bulk_read_model == "gpt-5.6-luna"
        assert config.code_write_model == "claude-haiku-4-5"

    def test_picks_the_marker_whose_tags_the_request_carries(self):
        router = _FakeRouter(
            [
                _marker({"auto_router_shunt_min_lines": 100}, tags=["eu"]),
                _marker({"auto_router_shunt_min_lines": 999}, tags=["us"]),
            ]
        )
        eu = shunt_config_for_model(llm_router=router, model_alias="shunt", team_id=None, request_tags=("eu",))
        assert eu.min_lines == 100


class TestAsyncPreCallHook:
    @pytest.mark.asyncio
    async def test_unarmed_returns_none(self):
        guardrail = ShuntGuardrail()
        data = {"model": "some-model", "messages": []}
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="acompletion"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_wrong_call_type_returns_none_without_touching_data(self, monkeypatch):
        guardrail = ShuntGuardrail()
        data = {"model": "shunt", "tools": []}
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="amoderation"
        )
        assert result is None
        assert data["tools"] == []

    @pytest.mark.asyncio
    async def test_armed_injects_anthropic_shape_tools(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        router = _FakeRouter([_marker({"auto_router_shunt_min_lines": 350})])
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)

        guardrail = mod.ShuntGuardrail()
        data = {"model": "shunt", "messages": []}
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="anthropic_messages"
        )
        assert result is not None
        tool_names = {tool["name"] for tool in result["tools"]}
        assert tool_names == {BULK_READ_TOOL_NAME, CODE_WRITE_TOOL_NAME}

    @pytest.mark.asyncio
    async def test_armed_injects_openai_shape_tools_for_completion(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        router = _FakeRouter([_marker({"auto_router_shunt_min_lines": 350})])
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)

        guardrail = mod.ShuntGuardrail()
        data = {"model": "shunt", "messages": []}
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="acompletion"
        )
        assert result is not None
        function_names = {tool["function"]["name"] for tool in result["tools"]}
        assert function_names == {BULK_READ_TOOL_NAME, CODE_WRITE_TOOL_NAME}

    @pytest.mark.asyncio
    async def test_appends_to_existing_tools_rather_than_replacing(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        router = _FakeRouter([_marker({"auto_router_shunt_min_lines": 350})])
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)

        guardrail = mod.ShuntGuardrail()
        existing_tool = {"name": "some_other_tool", "description": "x", "input_schema": {}}
        data = {"model": "shunt", "tools": [existing_tool]}
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="anthropic_messages"
        )
        tool_names = {tool["name"] for tool in result["tools"]}
        assert "some_other_tool" in tool_names
        assert BULK_READ_TOOL_NAME in tool_names


# Regression: a caller that already had its own `bulk_read`/`code_write` tool got a duplicate
# definition injected, and its own tool calls silently rewritten into shunt's curl.
class TestCallerOwnedToolNamesAreLeftAlone:
    @pytest.mark.asyncio
    async def test_pre_call_declines_to_inject_over_an_anthropic_shape_collision(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", _FakeRouter([_marker({"auto_router_shunt_min_lines": 350})]))
        caller_tool = {"name": BULK_READ_TOOL_NAME, "description": "the caller's own", "input_schema": {}}
        data = {"model": "shunt", "tools": [caller_tool]}
        result = await mod.ShuntGuardrail().async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="anthropic_messages"
        )
        assert result is None
        assert data["tools"] == [caller_tool]

    @pytest.mark.asyncio
    async def test_pre_call_declines_to_inject_over_an_openai_shape_collision(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", _FakeRouter([_marker({"auto_router_shunt_min_lines": 350})]))
        caller_tool = {"type": "function", "function": {"name": CODE_WRITE_TOOL_NAME, "parameters": {}}}
        data = {"model": "shunt", "tools": [caller_tool]}
        result = await mod.ShuntGuardrail().async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="acompletion"
        )
        assert result is None
        assert data["tools"] == [caller_tool]

    @pytest.mark.asyncio
    async def test_post_call_leaves_the_callers_own_tool_call_unrewritten(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        data = {
            "model": "shunt",
            "proxy_server_request": {"url": "http://localhost:4000/v1/messages"},
            "tools": [{"name": BULK_READ_TOOL_NAME, "input_schema": {}}],
        }
        guardrail = mod.ShuntGuardrail()
        await guardrail.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="anthropic_messages"
        )
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": BULK_READ_TOOL_NAME,
                    "input": {"question": "q", "paths": ["a.py"]},
                }
            ]
        }
        result = await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=None, response=response
        )
        assert result["content"][0]["name"] == BULK_READ_TOOL_NAME

    def _config(self) -> ShuntConfig:
        return ShuntConfig(min_lines=350, bulk_read_model="claude-haiku-4-5", code_write_model="claude-haiku-4-5")


class TestAsyncPostCallSuccessHookAnthropicShape:
    def _config(self) -> ShuntConfig:
        return ShuntConfig(min_lines=350, bulk_read_model="claude-haiku-4-5", code_write_model="claude-haiku-4-5")

    def _armed_request_data(self, model="shunt") -> dict:
        return {
            "model": model,
            "proxy_server_request": {"url": "http://localhost:4000/v1/messages"},
            "secret_fields": {"raw_headers": {"authorization": "Bearer sk-1234"}},
        }

    @pytest.mark.asyncio
    async def test_unarmed_returns_response_untouched(self):
        guardrail = ShuntGuardrail()
        response = {"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x.py"}}]}
        result = await guardrail.async_post_call_success_hook(
            data={"model": "some-model"}, user_api_key_dict=None, response=response
        )
        assert result["content"][0]["name"] == "Read"

    @pytest.mark.asyncio
    async def test_rewrites_untargeted_read_to_bash(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "litellm/router.py"}}]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        block = result["content"][0]
        assert block["name"] == "Bash"
        assert "curl" in block["input"]["command"]
        assert "litellm/router.py" in block["input"]["command"]

    @pytest.mark.asyncio
    async def test_leaves_targeted_read_untouched(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x.py", "offset": 10}}
            ]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        assert result["content"][0]["name"] == "Read"

    @pytest.mark.asyncio
    async def test_leaves_endpoints_unresolvable_response_untouched(self, monkeypatch):
        """No base_url/auth_header recoverable -> the rewrite is skipped rather than shipped broken."""
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "litellm/router.py"}}]
        }
        result = await guardrail.async_post_call_success_hook(
            data={"model": "shunt"}, user_api_key_dict=None, response=response
        )
        assert result["content"][0]["name"] == "Read"

    @pytest.mark.asyncio
    async def test_rewrites_explicit_bulk_read_call(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": BULK_READ_TOOL_NAME,
                    "input": {"question": "what does this do", "paths": ["a.py", "b.py"]},
                }
            ]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        block = result["content"][0]
        assert block["name"] == "Bash"
        assert "paths=@a.py" in block["input"]["command"]
        assert "paths=@b.py" in block["input"]["command"]

    @pytest.mark.asyncio
    async def test_rewrites_explicit_code_write_call_with_target(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": CODE_WRITE_TOOL_NAME,
                    "input": {"spec": "write tests", "reference": "tests/y_test.py", "target": "tests/x_test.py"},
                }
            ]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        block = result["content"][0]
        assert block["name"] == "Bash"
        assert block["input"]["command"].endswith("> tests/x_test.py")

    @pytest.mark.asyncio
    async def test_rewrites_bare_bash_read_command(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat litellm/router.py"}}
            ]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        assert "wc -l" in result["content"][0]["input"]["command"]

    @pytest.mark.asyncio
    async def test_leaves_piped_bash_command_untouched(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        original_command = "cat litellm/router.py | grep foo"
        response = {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": original_command}}]}
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        assert result["content"][0]["input"]["command"] == original_command

    @pytest.mark.asyncio
    async def test_leaves_non_tool_use_block_untouched(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = {"content": [{"type": "text", "text": "hello"}]}
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"


# Regression: the generated command used to embed the caller's raw Authorization header, which
# put the real key in the model's response and conversation history. It now carries a sealed,
# short-lived capability token that identifies the caller by reference instead.
class TestRewriteNeverCarriesTheCallersRealCredential:
    def _config(self) -> ShuntConfig:
        return ShuntConfig(min_lines=350, bulk_read_model="claude-haiku-4-5", code_write_model="claude-haiku-4-5")

    def _armed_request_data(self) -> dict:
        return {
            "model": "shunt",
            "proxy_server_request": {"url": "http://localhost:4000/v1/messages"},
            # A real caller's Authorization header may still be present on the request (secret_
            # fields is populated regardless of shunt), but the rewrite must never read it now.
            "secret_fields": {"raw_headers": {"authorization": "Bearer sk-the-callers-real-key"}},
        }

    @pytest.mark.asyncio
    async def test_rewritten_command_never_contains_the_callers_real_key(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        real_key_holder = UserAPIKeyAuth(api_key="fakehash1234567890")
        response = {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "litellm/router.py"}}]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=real_key_holder, response=response
        )
        command = result["content"][0]["input"]["command"]
        assert "sk-the-callers-real-key" not in command
        assert "shunt_cap_v1:" in command

    @pytest.mark.asyncio
    async def test_master_key_caller_gets_a_token_too(self, monkeypatch):
        """A master-key caller has no DB-backed key hash (LITELLM_PROXY_MASTER_KEY_ALIAS instead
        of a real hash), so the mint path must handle it without raising."""
        import litellm.proxy.guardrails.auto_router_shunt as mod
        from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-real-master-key")
        guardrail = mod.ShuntGuardrail()
        master_key_holder = UserAPIKeyAuth(api_key=LITELLM_PROXY_MASTER_KEY_ALIAS)
        response = {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "litellm/router.py"}}]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=master_key_holder, response=response
        )
        command = result["content"][0]["input"]["command"]
        assert "sk-the-real-master-key" not in command
        assert "shunt_cap_v1:" in command

    @pytest.mark.asyncio
    async def test_token_is_carried_in_the_authorization_header_not_a_query_string(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        holder = UserAPIKeyAuth(api_key="fakehash1234567890")
        response = {
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "litellm/router.py"}}]
        }
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=holder, response=response
        )
        command = result["content"][0]["input"]["command"]
        assert "-H " in command
        before_header, _, after_header = command.partition("-H ")
        assert "shunt_cap_v1:" not in before_header
        assert "shunt_cap_v1:" in after_header


class TestAsyncPostCallSuccessHookOpenAIShape:
    def _config(self) -> ShuntConfig:
        return ShuntConfig(min_lines=350, bulk_read_model="claude-haiku-4-5", code_write_model="claude-haiku-4-5")

    def _armed_request_data(self) -> dict:
        return {
            "model": "shunt",
            "proxy_server_request": {"url": "http://localhost:4000/v1/chat/completions"},
            "secret_fields": {"raw_headers": {"authorization": "Bearer sk-1234"}},
        }

    @pytest.mark.asyncio
    async def test_rewrites_untargeted_read_tool_call(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        tool_call = ChatCompletionMessageToolCall(
            id="call_1", function=Function(name="Read", arguments=json.dumps({"file_path": "litellm/router.py"}))
        )
        response = ModelResponse(
            choices=[Choices(index=0, message=Message(role="assistant", tool_calls=[tool_call]))]
        )
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        rewritten = result.choices[0].message.tool_calls[0]
        assert rewritten.function.name == "Bash"
        parsed = json.loads(rewritten.function.arguments)
        assert "curl" in parsed["command"]

    @pytest.mark.asyncio
    async def test_leaves_response_with_no_tool_calls_untouched(self, monkeypatch):
        import litellm.proxy.guardrails.auto_router_shunt as mod

        monkeypatch.setattr(mod, "_resolve_shunt_config", lambda data: self._config())
        guardrail = mod.ShuntGuardrail()
        response = ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content="hi"))])
        result = await guardrail.async_post_call_success_hook(
            data=self._armed_request_data(), user_api_key_dict=_FAKE_USER_API_KEY_DICT, response=response
        )
        assert result.choices[0].message.content == "hi"
