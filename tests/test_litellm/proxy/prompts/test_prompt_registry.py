import asyncio
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.proxy.prompts.prompt_registry import InMemoryPromptRegistry, parse_prompt_version
from litellm.types.prompts.init_prompts import PromptInfo, PromptLiteLLMParams, PromptSpec


def _db_prompt_spec(content: str, environment: str = "development", version: int = 1) -> PromptSpec:
    return PromptSpec(
        prompt_id=f"greeting.v{version}",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greeting",
            prompt_integration="dotprompt",
            prompt_data={"content": content, "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
        version=version,
        environment=environment,
    )


def _resolved_callback(registry: InMemoryPromptRegistry, environment: str | None = None) -> CustomPromptManagement:
    spec = registry.resolve_prompt_spec("greeting", environment=environment)
    assert spec is not None
    callback = registry.get_prompt_callback_for_prompt(prompt=spec)
    assert callback is not None
    return callback


def _served_content(registry: InMemoryPromptRegistry, environment: str | None = None) -> str:
    return _resolved_callback(registry, environment=environment).prompt_manager.get_prompt("greeting").content


@pytest.fixture
def isolated_callbacks(monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setattr(litellm, "callbacks", [])
    return litellm.callbacks


def test_sync_prompt_from_db_reloads_row_edited_elsewhere(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY"))
    stale_callback = _resolved_callback(registry)
    assert _served_content(registry) == "begin every reply with AHOY"

    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY"))

    assert _served_content(registry) == "begin every reply with HOWDY"
    reloaded_spec = registry.resolve_prompt_spec("greeting", environment="development")
    assert reloaded_spec is not None
    assert reloaded_spec.litellm_params.prompt_data["content"] == "begin every reply with HOWDY"
    assert stale_callback not in isolated_callbacks
    assert isolated_callbacks == [_resolved_callback(registry)]


def test_sync_prompt_from_db_keeps_unchanged_row_in_place(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY"))
    first_callback = _resolved_callback(registry)

    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY"))

    assert _resolved_callback(registry) is first_callback
    assert isolated_callbacks == [first_callback]


def test_reload_prompt_replaces_callback_without_leaking_the_old_one(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with AHOY"))
    stale_callback = _resolved_callback(registry)

    reloaded = registry.reload_prompt(prompt=_db_prompt_spec("begin every reply with HOWDY"))

    assert reloaded is not None
    assert _served_content(registry) == "begin every reply with HOWDY"
    assert stale_callback not in isolated_callbacks
    assert len(isolated_callbacks) == 1


def test_reload_prompt_keeps_the_old_template_when_the_replacement_fails(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with AHOY"))
    old_callback = _resolved_callback(registry)

    broken = PromptSpec(
        prompt_id="greeting.v1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greeting",
            prompt_integration="does_not_exist",
            prompt_data={"content": "begin every reply with HOWDY", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
        version=1,
        environment="development",
    )

    with pytest.raises(ValueError, match="Unsupported prompt"):
        registry.reload_prompt(prompt=broken)

    assert _resolved_callback(registry) is old_callback
    assert _served_content(registry) == "begin every reply with AHOY"
    assert isolated_callbacks == [old_callback]


def test_environments_sharing_a_prompt_id_keep_separate_templates(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY", environment="development"))
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY", environment="production"))

    assert _served_content(registry, environment="development") == "begin every reply with AHOY"
    assert _served_content(registry, environment="production") == "begin every reply with HOWDY"
    assert _resolved_callback(registry, environment="development") is not _resolved_callback(
        registry, environment="production"
    )


def test_default_resolution_prefers_production(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY", environment="development"))
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY", environment="production"))

    assert _served_content(registry) == "begin every reply with HOWDY"


@pytest.mark.parametrize("environment", ["staging", "qa"])
def test_default_resolution_serves_the_only_environment_present(isolated_callbacks: list, environment: str) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY", environment=environment))

    assert _served_content(registry) == "begin every reply with AHOY"


def test_resolution_picks_exact_version_and_latest_within_an_environment(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(
        prompt=_db_prompt_spec("begin every reply with AHOY", environment="development", version=1)
    )
    registry.sync_prompt_from_db(
        prompt=_db_prompt_spec("begin every reply with YO", environment="development", version=2)
    )
    registry.sync_prompt_from_db(
        prompt=_db_prompt_spec("begin every reply with HOWDY", environment="production", version=1)
    )

    exact = registry.resolve_prompt_spec("greeting", version=1, environment="development")
    assert exact is not None
    assert exact.litellm_params.prompt_data["content"] == "begin every reply with AHOY"

    latest = registry.resolve_prompt_spec("greeting", environment="development")
    assert latest is not None
    assert latest.litellm_params.prompt_data["content"] == "begin every reply with YO"

    assert registry.resolve_prompt_spec("greeting", version=3, environment="development") is None


def test_resolution_returns_none_for_unknown_environment_or_prompt(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY", environment="development"))

    assert registry.resolve_prompt_spec("greeting", environment="production") is None
    assert registry.resolve_prompt_spec("no_such_prompt") is None


def test_delete_prompts_by_base_id_scoped_to_one_environment(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY", environment="development"))
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY", environment="production"))
    production_callback = _resolved_callback(registry, environment="production")

    deleted = registry.delete_prompts_by_base_id(base_prompt_id="greeting", environment="development")

    assert deleted == ["greeting.v1::development"]
    assert registry.resolve_prompt_spec("greeting", environment="development") is None
    assert _resolved_callback(registry, environment="production") is production_callback
    assert _served_content(registry, environment="production") == "begin every reply with HOWDY"

    deleted_rest = registry.delete_prompts_by_base_id(base_prompt_id="greeting")

    assert deleted_rest == ["greeting.v1::production"]
    assert registry.resolve_prompt_spec("greeting") is None


def test_has_config_prompt_matches_any_version_of_the_base_id(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    config_spec = PromptSpec(
        prompt_id="greeting",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greeting",
            prompt_integration="dotprompt",
            prompt_data={"content": "begin every reply with AHOY", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="config"),
    )
    registry.initialize_prompt(prompt=config_spec)
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY", environment="production"))

    assert registry.has_config_prompt(base_prompt_id="greeting") is True
    assert registry.has_config_prompt(base_prompt_id="other_prompt") is False


def test_delete_prompts_by_base_id_removes_the_callbacks_from_litellm_callbacks(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with AHOY", version=1))
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with YO", version=2))
    assert len(isolated_callbacks) == 2

    deleted = registry.delete_prompts_by_base_id(base_prompt_id="greeting")

    assert sorted(deleted) == ["greeting.v1::development", "greeting.v2::development"]
    assert registry.resolve_prompt_spec("greeting") is None
    assert isolated_callbacks == []


def test_remove_prompt_is_a_no_op_for_an_unknown_registry_key(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with AHOY"))

    registry.remove_prompt(registry_key="not_there.v1::development")

    assert registry.resolve_prompt_spec("greeting") is not None
    assert len(isolated_callbacks) == 1


@pytest.mark.parametrize(
    ("raw_version", "expected"),
    [(2, 2), ("2", 2), (None, None), ("v2", None), (True, None), (2.0, None)],
)
def test_parse_prompt_version_accepts_integers_and_json_strings(raw_version: object, expected: int | None) -> None:
    assert parse_prompt_version(raw_version) == expected


def test_strip_version_suffix_handles_both_dot_and_underscore() -> None:
    from litellm.integrations.dotprompt.prompt_manager import strip_version_suffix

    assert strip_version_suffix("greeting.v1") == "greeting"
    assert strip_version_suffix("greeting_v2") == "greeting"
    assert strip_version_suffix("greeting") is None
    assert strip_version_suffix("greeting.vabc") is None
    assert strip_version_suffix("greeting_vabc") is None


def test_prompt_manager_version_indexing_and_retrieval() -> None:
    from litellm.integrations.dotprompt.prompt_manager import PromptManager

    manager: Final = PromptManager(
        prompt_data={"prompt_one": {"content": "Hello {{name}}", "metadata": {}}},
        prompt_version=3,
    )
    assert manager.get_prompt("prompt_one", version=3) is not None
    assert manager.get_prompt("prompt_one.v3") is not None
    assert manager.get_prompt("prompt_one_v3") is not None

    manager.add_prompt(prompt_id="prompt_two", content="Hi there")
    assert manager.get_prompt("prompt_two", version=3) is not None
    assert manager.get_prompt("prompt_two_v3") is not None


def test_dotprompt_manager_version_matching_and_fallback() -> None:
    from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager

    dot_mgr: Final = DotpromptManager(
        prompt_data={"greeting": {"content": "Hello {{name}}"}},
        prompt_id="greeting",
        prompt_version=2,
    )
    mismatched_spec: Final = _db_prompt_spec(content="Test", version=1)

    from litellm.types.utils import StandardCallbackDynamicParams

    dynamic_params: Final = StandardCallbackDynamicParams()

    assert (
        dot_mgr.should_run_prompt_management(
            prompt_id="greeting", dynamic_callback_params=dynamic_params, prompt_spec=mismatched_spec
        )
        is False
    )
    assert (
        dot_mgr.should_run_prompt_management(
            prompt_id="greeting_v2", dynamic_callback_params=dynamic_params, prompt_spec=None
        )
        is True
    )
    assert (
        dot_mgr.should_run_prompt_management(
            prompt_id="other_prompt_v2", dynamic_callback_params=dynamic_params, prompt_spec=None
        )
        is False
    )

    client: Final = dot_mgr._compile_prompt_helper(
        prompt_id="greeting",
        prompt_spec=None,
        prompt_variables={"name": "Alice"},
        dynamic_callback_params=dynamic_params,
    )
    assert client.get("prompt_template") is not None
    assert any("Alice" in str(msg.get("content", "")) for msg in client.get("prompt_template", []))


def test_logging_prompt_management_version_filtering(isolated_callbacks: list) -> None:
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    registry: Final = IN_MEMORY_PROMPT_REGISTRY
    registry.initialize_prompt(prompt=_db_prompt_spec("Prompt V1", version=1))
    registry.initialize_prompt(prompt=_db_prompt_spec("Prompt V2", version=2))

    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="call-123",
        start_time=datetime.now(),
        function_id="fn-123",
    )

    resolved_v1: Final = logging_obj.get_custom_logger_for_prompt_management(
        model="gpt-4o",
        non_default_params={},
        prompt_id="greeting",
        prompt_version=1,
    )
    assert resolved_v1 is not None
    assert getattr(resolved_v1, "prompt_version", None) == 1

    resolved_v2: Final = logging_obj.get_custom_logger_for_prompt_management(
        model="gpt-4o",
        non_default_params={},
        prompt_id="greeting",
        prompt_version=2,
    )
    assert resolved_v2 is not None
    assert getattr(resolved_v2, "prompt_version", None) == 2


def test_restore_fallback_prompt_state_allowlist() -> None:
    from litellm.router_utils.fallback_event_handlers import _restore_fallback_prompt_state

    kwargs: Final[dict[str, object]] = {
        "_unrendered_messages": [{"role": "user", "content": "original"}],
        "_original_prompt_params": {
            "prompt_id": "test_p",
            "prompt_variables": {"a": 1},
            "prompt_label": "prod",
            "prompt_version": 2,
            "prompt_environment": "staging",
            "malicious_key": "injected",
        },
        "_in_prompt_factory": True,
        "messages": [{"role": "user", "content": "rendered"}],
    }

    _restore_fallback_prompt_state(kwargs)

    assert kwargs["messages"] == [{"role": "user", "content": "original"}]
    assert kwargs["prompt_id"] == "test_p"
    assert kwargs["prompt_variables"] == {"a": 1}
    assert kwargs["prompt_label"] == "prod"
    assert kwargs["prompt_version"] == 2
    assert kwargs["prompt_environment"] == "staging"
    assert "malicious_key" not in kwargs
    assert "_in_prompt_factory" not in kwargs


def test_prompt_manager_version_indexing_with_prompt_file(tmp_path: Path) -> None:
    from litellm.integrations.dotprompt.prompt_manager import PromptManager

    prompt_path: Final = tmp_path / "greeting_test.prompt"
    prompt_path.write_text("---\nmodel: gpt-4o\n---\nHello {{name}}")

    manager: Final = PromptManager(
        prompt_id="greeting_test",
        prompt_file=str(prompt_path),
        prompt_version=2,
    )
    assert manager.get_prompt("greeting_test", version=2) is not None
    assert manager.get_prompt("greeting_test.v2") is not None
    assert manager.get_prompt("greeting_test_v2") is not None


def test_dotprompt_manager_async_get_chat_completion_prompt() -> None:
    from datetime import datetime

    from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import StandardCallbackDynamicParams

    dot_mgr: Final = DotpromptManager(
        prompt_data={"greeting": {"content": "Hello {{name}}"}},
        prompt_id="greeting",
        prompt_version=2,
    )
    dynamic_params: Final = StandardCallbackDynamicParams()
    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="call-async",
        start_time=datetime.now(),
        function_id="fn-async",
    )
    result: Final = asyncio.run(
        dot_mgr.async_get_chat_completion_prompt(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            non_default_params={},
            prompt_id="greeting",
            prompt_variables={"name": "Bob"},
            dynamic_callback_params=dynamic_params,
            litellm_logging_obj=logging_obj,
        )
    )
    rendered_messages: Final = result[1]
    assert any("Bob" in str(msg.get("content", "")) for msg in rendered_messages)


def test_logging_prompt_management_error_and_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    def mock_broken_resolve(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated registry failure")

    monkeypatch.setattr(IN_MEMORY_PROMPT_REGISTRY, "resolve_prompt_spec", mock_broken_resolve)

    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="call-err",
        start_time=datetime.now(),
        function_id="fn-err",
    )

    detected: Final = logging_obj._auto_detect_prompt_management_logger(
        prompt_id="unknown_prompt",
        prompt_spec=None,
        dynamic_callback_params=logging_obj.standard_callback_dynamic_params,
        prompt_version=1,
    )
    assert detected is None

    mismatched_logger: Final = DotpromptManager(
        prompt_data={"test": {"content": "Test"}},
        prompt_id="test",
        prompt_version=1,
    )
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        lambda callback_type: [mismatched_logger],
    )

    res: Final = logging_obj.get_custom_logger_for_prompt_management(
        model="gpt-4o",
        non_default_params={},
        prompt_id="test",
        dynamic_callback_params=logging_obj.standard_callback_dynamic_params,
        prompt_version=2,
    )
    assert res is None


def test_router_prompt_version_acompletion(
    isolated_callbacks: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.router import Router

    async def mock_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="router reply", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
        prompt=_db_prompt_spec(content="System: Hello\n\nUser: {{input}}", version=2)
    )

    router: Final = Router(
        model_list=[
            {
                "model_name": "p-model",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "greeting",
                    "prompt_version": 2,
                },
            }
        ]
    )

    response: Final = asyncio.run(
        router.acompletion(
            model="p-model",
            messages=[{"role": "user", "content": "hi"}],
            prompt_variables={"input": "world"},
        )
    )
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "router reply"


def test_router_async_function_with_fallbacks_prompt_management(
    isolated_callbacks: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.router import Router

    async def mock_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="fallback reply", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
        prompt=_db_prompt_spec(content="System: Hello\n\nUser: {{input}}", version=1)
    )

    router: Final = Router(
        model_list=[
            {
                "model_name": "pm-model",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "greeting",
                    "prompt_version": 1,
                },
            }
        ]
    )

    res_kw: Final = asyncio.run(
        router.async_function_with_fallbacks(
            model="pm-model",
            messages=[{"role": "user", "content": "hi"}],
            original_function=router._acompletion,
        )
    )
    assert isinstance(res_kw, litellm.ModelResponse)

    res_pos: Final = asyncio.run(
        router.async_function_with_fallbacks(
            [{"role": "user", "content": "hi"}],
            model="pm-model",
            original_function=router._acompletion,
        )
    )
    assert isinstance(res_pos, litellm.ModelResponse)


def test_router_dispatch_prompt_completion_direct_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Final[list[str]] = []

    async def mock_failover_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        model: Final = str(kwargs.get("model"))
        calls.append(model)
        if model == "unlisted-direct":
            raise RuntimeError("primary direct failure")
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="failover ok", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_failover_acompletion)

    from litellm.router import Router

    router: Final = Router(
        model_list=[
            {
                "model_name": "fallback-target",
                "litellm_params": {"model": "openai/gpt-4o-fallback"},
            }
        ],
        fallbacks=[{"unlisted-direct": ["fallback-target"]}],
        num_retries=0,
    )

    response: Final = asyncio.run(
        router._dispatch_prompt_completion(
            model="unlisted-direct",
            original_model_name="unlisted-direct",
            unrendered_messages=[{"role": "user", "content": "hi"}],
            original_prompt_params={"prompt_id": "test"},
            original_function=router._acompletion,
            prompt_management_params={"prompt_id", "prompt_version"},
            kwargs={"model": "unlisted-direct", "messages": [{"role": "user", "content": "hi"}], "metadata": {}},
        )
    )
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "failover ok"
    assert len(calls) == 2

    async def mock_success_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="direct success", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_success_acompletion)

    direct_res: Final = asyncio.run(
        router._dispatch_prompt_completion(
            model="unlisted-direct-2",
            original_model_name="unlisted-direct-2",
            unrendered_messages=[{"role": "user", "content": "hi"}],
            original_prompt_params={"prompt_id": "test"},
            original_function=router._acompletion,
            prompt_management_params={"prompt_id", "prompt_version"},
            kwargs={"model": "unlisted-direct-2", "messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert isinstance(direct_res, litellm.ModelResponse)
    assert direct_res.choices[0].message.content == "direct success"

    async def mock_error_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        raise RuntimeError("unhandled direct error")

    monkeypatch.setattr(litellm, "acompletion", mock_error_acompletion)

    empty_router: Final = Router(model_list=[])
    with pytest.raises(RuntimeError, match="unhandled direct error"):
        asyncio.run(
            empty_router._dispatch_prompt_completion(
                model="unlisted-err",
                original_model_name="unlisted-err",
                unrendered_messages=[{"role": "user", "content": "hi"}],
                original_prompt_params={"prompt_id": "test"},
                original_function=empty_router._acompletion,
                prompt_management_params={"prompt_id", "prompt_version"},
                kwargs={"model": "unlisted-err", "messages": [{"role": "user", "content": "hi"}]},
            )
        )


def test_router_prompt_management_factory_registry_lookup_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.router import Router

    def mock_failing_resolve(*args: object, **kwargs: object) -> None:
        raise RuntimeError("registry exception")

    monkeypatch.setattr(IN_MEMORY_PROMPT_REGISTRY, "resolve_prompt_spec", mock_failing_resolve)

    async def mock_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="fallback reply", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    router: Final = Router(
        model_list=[
            {
                "model_name": "direct-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "prompt_id": "test-prompt",
                    "prompt_version": 1,
                },
            }
        ]
    )

    response: Final = asyncio.run(
        router._prompt_management_factory(
            model="direct-model",
            messages=[{"role": "user", "content": "hi"}],
            kwargs={"model": "direct-model", "original_function": router._acompletion},
        )
    )
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "fallback reply"


def test_router_fallback_hop_restores_prompt_parameters_and_messages(
    isolated_callbacks: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.router import Router

    IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
        prompt=_db_prompt_spec(content="System: Primary V1\n\nUser: {{input}}", version=1)
    )
    IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
        prompt=_db_prompt_spec(content="System: Fallback V2\n\nUser: {{input}}", version=2)
    )

    calls: Final[list[dict]] = []

    async def mock_acompletion(*args: object, **kwargs: object) -> litellm.ModelResponse:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("primary prompt failure")
        return litellm.ModelResponse(
            choices=[litellm.utils.Choices(message=litellm.utils.Message(content="hop ok", role="assistant"))]
        )

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    router: Final = Router(
        model_list=[
            {
                "model_name": "primary-pm",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "greeting",
                    "prompt_version": 1,
                },
            },
            {
                "model_name": "fallback-pm",
                "litellm_params": {
                    "model": "dotprompt/gpt-4o",
                    "prompt_id": "greeting",
                    "prompt_version": 2,
                },
            },
        ],
        fallbacks=[{"primary-pm": ["fallback-pm"]}],
        num_retries=0,
    )

    response: Final = asyncio.run(
        router.acompletion(
            model="primary-pm",
            messages=[{"role": "user", "content": "hi"}],
            prompt_variables={"input": "test-val"},
        )
    )
    assert isinstance(response, litellm.ModelResponse)
    assert response.choices[0].message.content == "hop ok"
    assert len(calls) == 2
