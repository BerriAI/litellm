import asyncio
from typing import Final

import pytest

import litellm
from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
from litellm.router import Router
from litellm.types.prompts.init_prompts import PromptInfo, PromptLiteLLMParams, PromptSpec


@pytest.fixture
def clean_registry():
    IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.clear()
    IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt.clear()
    yield IN_MEMORY_PROMPT_REGISTRY
    IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.clear()
    IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt.clear()


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    async def mock_acompletion(*args, **kwargs):
        calls.append(dict(kwargs))
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="mock reply", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)
    return calls


def test_router_serves_specified_prompt_version_instead_of_first_registered(
    clean_registry, captured_calls: list[dict]
) -> None:
    # 1. Register version 1 first
    spec_v1: Final = PromptSpec(
        prompt_id="assistant.v1",
        version=1,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="assistant",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: You are Assistant V1\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.initialize_prompt(spec_v1)

    # 2. Register version 2 second
    spec_v2: Final = PromptSpec(
        prompt_id="assistant.v2",
        version=2,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="assistant",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: You are Assistant V2\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.initialize_prompt(spec_v2)

    # 3. Router deployment configured with prompt_version=2
    router: Final = Router(
        model_list=[
            {
                "model_name": "agent-v2",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "assistant",
                    "prompt_version": 2,
                },
            }
        ]
    )

    response: Final = asyncio.run(
        router.acompletion(
            model="agent-v2",
            messages=[{"role": "user", "content": "hello"}],
        )
    )
    assert response.choices[0].message.content == "mock reply"
    assert len(captured_calls) == 1
    call_messages: Final = captured_calls[0]["messages"]
    system_messages: Final = [m for m in call_messages if m.get("role") == "system"]
    assert len(system_messages) == 1
    assert "You are Assistant V2" in system_messages[0]["content"]
    assert "You are Assistant V1" not in system_messages[0]["content"]


def test_router_prompt_management_prevents_double_rendering(clean_registry, captured_calls: list[dict]) -> None:
    spec: Final = PromptSpec(
        prompt_id="greet.v1",
        version=1,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greet",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: Single greeting template\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.initialize_prompt(spec)

    router: Final = Router(
        model_list=[
            {
                "model_name": "greet-model",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "greet",
                    "prompt_version": 1,
                },
            }
        ]
    )

    asyncio.run(
        router.acompletion(
            model="greet-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert len(captured_calls) == 1
    captured: Final = captured_calls[0]
    call_messages: Final = captured["messages"]
    # Only 1 system message rendered, not two
    system_messages: Final = [m for m in call_messages if m.get("role") == "system"]
    assert len(system_messages) == 1
    # prompt_id, prompt_variables, prompt_version should be stripped from downstream kwargs
    assert "prompt_id" not in captured
    assert "prompt_variables" not in captured
    assert "prompt_version" not in captured
    assert "prompt_environment" not in captured


def test_router_serves_prompt_version_passed_as_string(clean_registry, captured_calls: list[dict]) -> None:
    spec_v1: Final = PromptSpec(
        prompt_id="bot.v1",
        version=1,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="bot",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: Bot V1\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.initialize_prompt(spec_v1)
    spec_v2: Final = PromptSpec(
        prompt_id="bot.v2",
        version=2,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="bot",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: Bot V2\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.initialize_prompt(spec_v2)

    router: Final = Router(
        model_list=[
            {
                "model_name": "bot-model",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "bot",
                    "prompt_version": "2",  # passed as string (e.g. from YAML or JSON)
                },
            }
        ]
    )

    asyncio.run(
        router.acompletion(
            model="bot-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert len(captured_calls) == 1
    call_messages: Final = captured_calls[0]["messages"]
    system_messages: Final = [m for m in call_messages if m.get("role") == "system"]
    assert len(system_messages) == 1
    assert "Bot V2" in system_messages[0]["content"]


def test_router_serves_reloaded_prompt_version_immediately(clean_registry, captured_calls: list[dict]) -> None:
    spec_v1: Final = PromptSpec(
        prompt_id="dyn.v1",
        version=1,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="dyn",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: Initial text\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.initialize_prompt(spec_v1)

    router: Final = Router(
        model_list=[
            {
                "model_name": "dyn-model",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "dyn",
                    "prompt_version": 1,
                },
            }
        ]
    )

    asyncio.run(router.acompletion(model="dyn-model", messages=[{"role": "user", "content": "1"}]))
    assert "Initial text" in captured_calls[-1]["messages"][0]["content"]

    # Now reload with updated content
    updated_spec: Final = PromptSpec(
        prompt_id="dyn.v1",
        version=1,
        environment="production",
        litellm_params=PromptLiteLLMParams(
            prompt_id="dyn",
            prompt_integration="dotprompt",
            prompt_data={"content": "System: Updated text\n\nUser: {{input}}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )
    clean_registry.reload_prompt(updated_spec)

    asyncio.run(router.acompletion(model="dyn-model", messages=[{"role": "user", "content": "2"}]))
    assert "Updated text" in captured_calls[-1]["messages"][0]["content"]


def test_dotprompt_managers_with_different_versions_have_distinct_logger_keys() -> None:
    from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager

    mgr_v1: Final = DotpromptManager(
        prompt_id="greeting",
        prompt_version=1,
        prompt_data={"content": "v1"},
    )
    mgr_v2: Final = DotpromptManager(
        prompt_id="greeting",
        prompt_version=2,
        prompt_data={"content": "v2"},
    )

    key_v1: Final = litellm.logging_callback_manager._get_custom_logger_key(mgr_v1)
    key_v2: Final = litellm.logging_callback_manager._get_custom_logger_key(mgr_v2)

    assert key_v1 != key_v2
    assert "prompt_version=1" in key_v1
    assert "prompt_version=2" in key_v2
