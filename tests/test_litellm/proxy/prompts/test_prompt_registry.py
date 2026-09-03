import pytest

import litellm
from litellm.proxy.prompts.prompt_registry import InMemoryPromptRegistry
from litellm.types.prompts.init_prompts import PromptInfo, PromptLiteLLMParams, PromptSpec


def _db_prompt_spec(content: str) -> PromptSpec:
    return PromptSpec(
        prompt_id="greeting.v1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greeting",
            prompt_integration="dotprompt",
            prompt_data={"content": content, "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )


def _served_content(registry: InMemoryPromptRegistry) -> str:
    callback = registry.get_prompt_callback_by_id("greeting.v1")
    assert callback is not None
    return callback.prompt_manager.get_prompt("greeting").content


@pytest.fixture
def isolated_callbacks(monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setattr(litellm, "callbacks", [])
    return litellm.callbacks


def test_sync_prompt_from_db_reloads_row_edited_elsewhere(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY"))
    stale_callback = registry.get_prompt_callback_by_id("greeting.v1")
    assert _served_content(registry) == "begin every reply with AHOY"

    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY"))

    assert _served_content(registry) == "begin every reply with HOWDY"
    assert registry.get_prompt_by_id("greeting.v1").litellm_params.prompt_data["content"] == "begin every reply with HOWDY"
    assert stale_callback not in isolated_callbacks
    assert isolated_callbacks == [registry.get_prompt_callback_by_id("greeting.v1")]


def test_sync_prompt_from_db_keeps_unchanged_row_in_place(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY"))
    first_callback = registry.get_prompt_callback_by_id("greeting.v1")

    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY"))

    assert registry.get_prompt_callback_by_id("greeting.v1") is first_callback
    assert isolated_callbacks == [first_callback]


def test_reload_prompt_replaces_callback_without_leaking_the_old_one(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with AHOY"))
    stale_callback = registry.get_prompt_callback_by_id("greeting.v1")

    reloaded = registry.reload_prompt(prompt=_db_prompt_spec("begin every reply with HOWDY"))

    assert reloaded is not None
    assert _served_content(registry) == "begin every reply with HOWDY"
    assert stale_callback not in isolated_callbacks
    assert len(isolated_callbacks) == 1


def test_reload_prompt_keeps_the_old_template_when_the_replacement_fails(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_db_prompt_spec("begin every reply with AHOY"))
    old_callback = registry.get_prompt_callback_by_id("greeting.v1")

    broken = PromptSpec(
        prompt_id="greeting.v1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greeting",
            prompt_integration="does_not_exist",
            prompt_data={"content": "begin every reply with HOWDY", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db"),
    )

    with pytest.raises(ValueError, match="Unsupported prompt"):
        registry.reload_prompt(prompt=broken)

    assert registry.get_prompt_callback_by_id("greeting.v1") is old_callback
    assert _served_content(registry) == "begin every reply with AHOY"
    assert isolated_callbacks == [old_callback]


def _versioned_prompt_spec(version: int, environment: str) -> PromptSpec:
    return PromptSpec(
        prompt_id=f"greeting.v{version}",
        litellm_params=PromptLiteLLMParams(
            prompt_id="greeting",
            prompt_integration="dotprompt",
            prompt_data={"content": f"begin every reply with AHOY v{version}", "metadata": {}},
        ),
        prompt_info=PromptInfo(prompt_type="db", environment=environment),
        version=version,
        environment=environment,
    )


def test_delete_prompts_by_base_id_removes_the_callbacks_from_litellm_callbacks(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_versioned_prompt_spec(1, "development"))
    registry.initialize_prompt(prompt=_versioned_prompt_spec(2, "development"))
    assert len(isolated_callbacks) == 1

    deleted = registry.delete_prompts_by_base_id("greeting")

    assert sorted(deleted) == ["greeting.v1", "greeting.v2"]
    assert registry.get_prompt_by_id("greeting.v1") is None
    assert registry.get_prompt_callback_by_id("greeting.v2") is None
    assert isolated_callbacks == []


def test_delete_prompts_by_base_id_environment_scope_keeps_other_environments(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_versioned_prompt_spec(1, "development"))
    registry.initialize_prompt(prompt=_versioned_prompt_spec(2, "production"))
    production_callback = registry.get_prompt_callback_by_id("greeting.v2")

    deleted = registry.delete_prompts_by_base_id("greeting", environment="development")

    assert deleted == ["greeting.v1"]
    assert registry.get_prompt_by_id("greeting.v1") is None
    assert registry.get_prompt_by_id("greeting.v2") is not None
    assert registry.get_prompt_callback_by_id("greeting.v2") is production_callback


def test_remove_prompt_is_a_no_op_for_an_unknown_id(isolated_callbacks: list) -> None:
    registry = InMemoryPromptRegistry()
    registry.initialize_prompt(prompt=_versioned_prompt_spec(1, "development"))

    registry.remove_prompt(prompt_id="not_there.v1")

    assert registry.get_prompt_by_id("greeting.v1") is not None
    assert len(isolated_callbacks) == 1
