import pytest

import litellm
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.proxy.prompts.prompt_registry import InMemoryPromptRegistry
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
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with AHOY", environment="development", version=1))
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with YO", environment="development", version=2))
    registry.sync_prompt_from_db(prompt=_db_prompt_spec("begin every reply with HOWDY", environment="production", version=1))

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
    assert len(isolated_callbacks) == 1

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
