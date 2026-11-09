from collections.abc import Mapping
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Final, Literal, cast

import pytest

import litellm
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import grouped_model_access
from litellm.proxy._types import ModelAccessDeniedProxyException
from litellm.proxy.auth import model_access_adapter, model_projection_adapter
from litellm.router_api import RegistryReader
from tests.unit._auth.support import model_registry_reference as reference


@dataclass(frozen=True, slots=True)
class _Info:
    team_id: str | None = "team"
    team_public_model_name: str | None = "public"


@dataclass(frozen=True, slots=True)
class _Deployment:
    model_info: _Info = field(default_factory=_Info)


@dataclass(frozen=True, slots=True)
class _Registry:
    events: SimpleQueue[tuple[str, str | None]]
    entries: tuple[str, ...] = ("registered",)
    router_aliases: Mapping[str, str] = field(default_factory=lambda: {"alias": "target"})
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {"group": ("m",)})
    deployment: _Deployment | None = field(default_factory=_Deployment)
    fail: str | None = None

    def observe(self, kind: str, model: str | None = None) -> None:
        self.events.put((kind, model))
        if self.fail == kind:
            raise RuntimeError(f"failed {kind}")

    def names(self) -> list[str]:  # mutable-ok: the legacy sentinel resolver concatenates the registry's concrete list
        self.observe("names")
        return list(self.entries)

    def aliases(self) -> Mapping[str, str]:
        self.observe("aliases")
        return self.router_aliases

    def resolve(self, model: str, *, kind: Literal["alias"]) -> str | None:
        self.observe("resolve", model)
        return self.router_aliases.get(model)

    def access_groups(self, model_name: str, team_id: str | None) -> Mapping[str, tuple[str, ...]]:
        self.observe("groups", model_name)
        return self.groups

    def deployments(self, model: str, *, source: Literal["first_in_group"]) -> _Deployment | None:
        self.observe("deployment", model)
        return self.deployment


@dataclass(frozen=True, slots=True)
class _Router:
    view: _Registry

    def registry(self) -> _Registry:
        self.view.observe("registry")
        return self.view


def _reader(view: _Registry) -> RegistryReader:
    return cast(RegistryReader, _Router(view))  # noqa: TID251  # cast-ok: typed IO double implements only the registry methods consumed by auth


def _events(events: SimpleQueue[tuple[str, str | None]]) -> tuple[tuple[str, str | None], ...]:
    return tuple(events.get() for _ in range(events.qsize()))


def _run(native: bool, model: str | tuple[str, ...], grants: tuple[str, ...], registry: _Registry) -> str:
    router: Final = _reader(registry)
    aliases: Final = {"z": "model_name_team_deploy", "a": "model_name_team_deploy", "team-alias": "target"}
    try:
        if native:
            model_access_adapter.can_object_call_model(model, router, grants, aliases, "team", {"key": "target"})
        else:
            reference.can_object_call_model(
                model if isinstance(model, str) else list(model),
                router,
                list(grants),
                aliases,
                "team",
                {"key": "target"},
            )
        return "allowed"
    except ModelAccessDeniedProxyException as error:
        return str((error.message, error.internal_message, error.type, error.param, error.code))


@pytest.mark.parametrize(
    "model", ("m", "alias", "global", "key", "team-alias", "model_name_team_deploy", ("m", "m"), ("denied", "m"), ())
)
@pytest.mark.parametrize(
    "grants",
    (
        (),
        ("m",),
        ("target",),
        ("group",),
        ("all-team-models",),
        ("all-team-models", "other"),
        ("*",),
        ("other/*", "provider/*"),
        ("a",),
        ("public",),
    ),
)
def test_registry_observation_order_and_denials_match_legacy(
    monkeypatch: pytest.MonkeyPatch, model: str | tuple[str, ...], grants: tuple[str, ...]
) -> None:
    events: Final[SimpleQueue[tuple[str, str | None]]] = SimpleQueue()
    registry: Final = _Registry(events)

    def provider(model: str) -> tuple[str, str, None, None]:
        events.put(("provider", model))
        return "leaf", "provider", None, None

    monkeypatch.setattr(litellm, "get_llm_provider", provider)
    monkeypatch.setattr(litellm, "model_alias_map", {"global": "target"})
    expected: Final = _run(False, model, grants, registry)
    expected_events: Final = _events(events)
    assert _run(True, model, grants, registry) == expected
    assert _events(events) == expected_events


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("failed", ("aliases", "groups", "names", "deployment"))
def test_registry_failure_stops_before_later_reads(native: bool, failed: str) -> None:
    events: Final[SimpleQueue[tuple[str, str | None]]] = SimpleQueue()
    registry: Final = _Registry(events, fail=failed)
    with pytest.raises(RuntimeError, match=f"failed {failed}") as error:
        _run(native, "model_name_team_deploy", ("all-team-models",), registry)
    assert not isinstance(error.value, NativeAuthFailure)
    assert _events(events)[-1][0] == failed


@pytest.mark.parametrize("native", (False, True))
def test_depth_denial_never_reads_registry(native: bool) -> None:
    events: Final[SimpleQueue[tuple[str, str | None]]] = SimpleQueue()
    router: Final = _reader(_Registry(events, fail="registry"))
    policy: Final = model_access_adapter.can_object_call_model if native else reference.can_object_call_model
    with pytest.raises(Exception, match="max fallback depth exceeded"):
        policy("m", router, (), fallback_depth=100)
    assert _events(events) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("group_models", (("other",), ("other/*",), ("m",)))
async def test_group_retry_gets_fresh_registry_reads_even_for_duplicate_grants(group_models: tuple[str, ...]) -> None:
    events: Final[SimpleQueue[tuple[str, str | None]]] = SimpleQueue()
    router: Final = _reader(_Registry(events))
    facts: Final = model_access_adapter.project_model_facts("m", router, ("other",))
    assert _events(events) == (("registry", None), ("aliases", None), ("registry", None), ("groups", "m"))
    scope: Final = await model_projection_adapter.read_grouped_scope(
        GroupedModelScope(facts, ("group",), group_models), router, merge=False
    )
    assert _events(events)[:4] == (("registry", None), ("aliases", None), ("registry", None), ("groups", "m"))
    assert grouped_model_access(scope.facts, scope.group_ids, scope.group_models).kind == (
        "Allow" if group_models == ("m",) else "Deny"
    )


def test_native_provider_failure_is_not_a_missing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def provider(model: str) -> tuple[str, str, None, None]:
        raise NativeAuthRuntimeError("terminal provider failure")

    monkeypatch.setattr(litellm, "get_llm_provider", provider)
    with pytest.raises(NativeAuthRuntimeError, match="terminal provider failure"):
        model_access_adapter.can_object_call_model("model", None, ("provider/*",))


@pytest.mark.parametrize("native", (False, True))
def test_compaction_parent_is_read_only_after_child_denial(native: bool) -> None:
    from litellm.router_strategy.complexity_router.context_compaction import native_compaction_call

    events: Final[SimpleQueue[tuple[str, str | None]]] = SimpleQueue()
    with native_compaction_call("parent", "child"):
        assert _run(native, "child", ("parent",), _Registry(events)) == "allowed"
    assert _events(events) == (
        ("registry", None),
        ("aliases", None),
        ("registry", None),
        ("groups", "child"),
        ("registry", None),
        ("groups", "parent"),
    )


def test_large_model_lists_do_not_consume_python_recursion_depth() -> None:
    events: Final[SimpleQueue[tuple[str, str | None]]] = SimpleQueue()
    assert _run(True, ("m",) * 180, ("m",), _Registry(events)) == "allowed"
    assert _events(events) == (("registry", None), ("aliases", None), ("registry", None), ("groups", "m")) * 180
