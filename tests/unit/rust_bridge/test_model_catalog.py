import json
import threading
from typing import Final

import pytest

from litellm.rust_bridge._native import ModelCatalog
from litellm.rust_bridge.model_catalog import (
    ModelCatalogSlot,
    ModelCostView,
    RegistryRejected,
    bundled_registry_bytes,
    parse_registry,
)


def _parsed(entries: dict[str, dict[str, object]]) -> ModelCatalog:
    parsed: Final = parse_registry(json.dumps(entries).encode(), source="local")
    assert not isinstance(parsed, RegistryRejected)
    return parsed


def _registry(entries: dict[str, dict[str, object]]) -> ModelCatalogSlot:
    return ModelCatalogSlot(_parsed(entries))


def _view(slot: ModelCatalogSlot) -> ModelCostView:
    return ModelCostView(lambda: slot.current)


def test_the_bundled_registry_loads() -> None:
    bundled: Final = parse_registry(bundled_registry_bytes(), source="local")
    assert not isinstance(bundled, RegistryRejected)
    assert bundled.registry_model_count > 0


@pytest.mark.parametrize("body", [b"{", b"{}", b'{"gpt-4o": 1}'])
def test_an_invalid_registry_is_rejected_as_a_value(body: bytes) -> None:
    assert isinstance(parse_registry(body, source="remote"), RegistryRejected)


def test_the_view_serves_whatever_the_slot_serves_now() -> None:
    slot: Final = _registry({"gpt-4o": {"input_cost_per_token": 1.0}})
    view: Final = _view(slot)
    slot.register("sdk", {"my-model": {"mode": "chat"}})
    assert dict(view) == {"gpt-4o": {"input_cost_per_token": 1.0}, "my-model": {"mode": "chat"}}
    assert len(view) == 2
    assert "my-model" in view
    assert 1 not in view


def test_registrations_survive_adopting_a_new_registry() -> None:
    slot: Final = _registry({"gpt-4o": {"input_cost_per_token": 1.0}})
    slot.register("sdk", {"my-model": {"mode": "chat"}})
    slot.adopt(_parsed({"claude": {"mode": "chat"}}))
    assert dict(_view(slot)) == {"claude": {"mode": "chat"}, "my-model": {"mode": "chat"}}


def test_a_reload_changes_every_field_a_registration_did_not_set() -> None:
    slot: Final = _registry(
        {"claude-x": {"litellm_provider": "bedrock_converse", "input_cost_per_token": 3.0, "output_cost_per_token": 15.0}}
    )
    slot.register("sdk", {"claude-x": {"input_cost_per_token": 1.0}})
    slot.adopt(
        _parsed({"claude-x": {"litellm_provider": "bedrock", "input_cost_per_token": 4.0, "output_cost_per_token": 20.0}})
    )
    assert slot.current.get("claude-x") == {
        "litellm_provider": "bedrock",
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 20.0,
    }


def test_unregistering_an_owner_removes_only_what_it_added() -> None:
    slot: Final = _registry({"hosted_vllm/shared": {"litellm_provider": "hosted_vllm", "mode": "chat"}})
    slot.register("deployment:a", {"a": {"max_input_tokens": 111}, "hosted_vllm/shared": {"supports_vision": True}})
    slot.register("deployment:b", {"b": {"max_input_tokens": 222}, "hosted_vllm/shared": {"max_tokens": 10}})
    slot.unregister("deployment:a")
    assert dict(_view(slot)) == {
        "hosted_vllm/shared": {"litellm_provider": "hosted_vllm", "mode": "chat", "max_tokens": 10},
        "b": {"max_input_tokens": 222},
    }


@pytest.mark.parametrize(
    ("unregistered", "price"),
    [
        ((), 4.0),
        (("sdk",), 3.0),
        (("deployment:a",), 4.0),
        (("sdk", "deployment:a"), 1.0),
        (("deployment:missing",), 4.0),
    ],
)
def test_the_latest_remaining_write_wins(unregistered: tuple[str, ...], price: float) -> None:
    slot: Final = _registry({"gpt-4o": {"input_cost_per_token": 1.0}})
    slot.register("sdk", {"gpt-4o": {"input_cost_per_token": 2.0}})
    slot.register("deployment:a", {"gpt-4o": {"input_cost_per_token": 3.0}})
    slot.register("sdk", {"gpt-4o": {"input_cost_per_token": 4.0}})
    for owner in unregistered:
        slot.unregister(owner)
    assert slot.current.get("gpt-4o") == {"input_cost_per_token": price}


def test_an_owner_registered_before_a_reload_can_unregister_after_it() -> None:
    slot: Final = _registry({"gpt-4o": {"input_cost_per_token": 1.0}})
    slot.register("deployment:a", {"a": {"mode": "chat"}, "claude": {"max_tokens": 10}})
    slot.adopt(_parsed({"claude": {"mode": "chat"}}))
    slot.unregister("deployment:a")
    assert dict(_view(slot)) == {"claude": {"mode": "chat"}}


def test_concurrent_registrations_are_never_lost() -> None:
    slot: Final = _registry({"gpt-4o": {"input_cost_per_token": 1.0}})
    names: Final = tuple(f"model-{index}" for index in range(64))
    threads: Final = tuple(
        threading.Thread(target=slot.register, args=("sdk", {name: {"mode": "chat"}})) for name in names
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(name in slot.current for name in names)


def test_the_view_refuses_writes() -> None:
    view: Final = _view(_registry({"gpt-4o": {"input_cost_per_token": 1.0}}))
    with pytest.raises(TypeError):
        view["gpt-4o"]["input_cost_per_token"] = 2.0  # pyright: ignore[reportIndexIssue]  # the write must fail
    with pytest.raises(TypeError):
        view["gpt-4o"] = {}  # pyright: ignore[reportIndexIssue]  # the write must fail
