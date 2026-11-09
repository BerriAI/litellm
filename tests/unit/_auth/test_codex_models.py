from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, cast

import pytest

import litellm
from litellm._auth.failure import NativeAuthRuntimeError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth import codex_model_adapter as native
from litellm.router_api import RouterRegistry
from litellm.rust_bridge import _native
from litellm.types.router import Deployment
from tests.unit._auth.support import codex_reference as reference


@dataclass(frozen=True, slots=True)
class Registry:
    deployments: tuple[Deployment | None, ...]
    alias: str | None = None
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    calls: list[tuple[object, ...]] = field(default_factory=list)  # mutable-ok: ordered raw registry IO trace

    def names(self, model: str | None = None, *, kind: str = "all", team_id: str | None = None) -> tuple[str, ...]:
        self.calls.append(("names", model, kind, team_id))
        return (
            tuple(str(index) for index in range(len(self.deployments)))
            if kind == "route_candidates"
            else ("model", "chatgpt/model")
        )

    def deployment(self, model_id: str) -> Deployment | None:
        self.calls.append(("deployment", model_id))
        return self.deployments[int(model_id)]

    def resolve(self, model: str, *, kind: str) -> str | None:
        self.calls.append(("resolve", model, kind))
        return self.alias

    def access_groups(self, model_name: str, team_id: str | None = None) -> tuple[str, ...]:
        self.calls.append(("groups", model_name, team_id))
        return self.groups.get(model_name, ())


@dataclass(frozen=True, slots=True)
class Router:
    reader: Registry

    def registry(self) -> RouterRegistry:
        self.reader.calls.append(("registry",))
        return cast(RouterRegistry, self.reader)  # cast-ok: raw registry double supplies the selected IO methods


def deployment(
    name: str, backend: str = "openai/backend", provider: str | None = None, public: str | None = None
) -> Deployment:
    return Deployment.model_validate(
        {
            "model_name": name,
            "litellm_params": {"model": backend, "custom_llm_provider": provider},
            "model_info": {"team_public_model_name": public},
        }
    )


_DEPLOYMENTS: Final = (
    (),
    (None,),
    (deployment("model"),),
    (deployment("target"),),
    (deployment("internal", public="model"),),
    (deployment("other"),),
    (deployment("model", "chatgpt/backend"),),
    (deployment("model", provider="chatgpt"),),
    (deployment("model"), deployment("model", "chatgpt/backend")),
    (None, deployment("model")),
)


@pytest.mark.parametrize("model", (None, [], 5, "", "model", "bundled", "priced", "chatgpt/model"))
@pytest.mark.parametrize("alias", (None, "target", ""))
@pytest.mark.parametrize("deployments", _DEPLOYMENTS)
@pytest.mark.parametrize("router_present", (False, True))
def test_route_candidates_match_frozen_main_and_read_order(
    monkeypatch: pytest.MonkeyPatch,
    model: object,
    alias: str | None,
    deployments: tuple[Deployment | None, ...],
    router_present: bool,
) -> None:
    monkeypatch.setattr(native, "bundled_codex_models", lambda: {"bundled": None})
    prices: Final[Mapping[str, object]] = {"chatgpt/priced": {}}
    monkeypatch.setattr(litellm, "model_cost", prices)
    old: Final = Router(Registry(deployments, alias))
    new: Final = Router(Registry(deployments, alias))
    expected: Final = reference.routes_to_litellm(model, old if router_present else None, "team", ("bundled",), prices)
    assert native.routes_to_litellm(model, new if router_present else None, "team") == expected
    assert new.reader.calls == old.reader.calls


@pytest.mark.parametrize("model", ("model", "bundled", "chatgpt/model", ""))
@pytest.mark.parametrize(
    "key", ((), ("model",), ("chatgpt/model",), ("other",), ("*",), ("all-team-models",), ("group",))
)
@pytest.mark.parametrize("team", ((), ("model",), ("chatgpt/model",), ("other",)))
@pytest.mark.parametrize("deployments", _DEPLOYMENTS)
def test_chatgpt_prefix_matches_frozen_main_and_grant_read_order(
    model: str, key: tuple[str, ...], team: tuple[str, ...], deployments: tuple[Deployment | None, ...]
) -> None:
    groups: Final = {"model": ("group",), "chatgpt/model": ("group",)}
    old: Final = Router(Registry(deployments, groups=groups))
    new: Final = Router(Registry(deployments, groups=groups))
    token: Final = UserAPIKeyAuth(models=list(key), team_models=list(team), team_id="team")
    assert native.native_chatgpt_model(model, new, token) == reference.native_chatgpt_model(model, old, token)
    assert new.reader.calls == old.reader.calls


def test_native_model_failure_is_terminal_before_registry_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = NativeAuthRuntimeError("injected native failure")

    def fail(_facts: object) -> object:
        raise failure

    monkeypatch.setattr(_native, "request_auth_codex_model", fail)
    router: Final = Router(Registry((deployment("model"),)))
    with pytest.raises(NativeAuthRuntimeError) as error:
        native.routes_to_litellm("model", router, "team")
    assert error.value is failure
    assert router.reader.calls == []


@pytest.mark.parametrize("route", ("/backend-api/codex/responses", "/v1/responses", "/messages"))
def test_native_route_guard_precedes_all_model_reads(route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "bundled_codex_models", lambda: dict[str, object]())
    monkeypatch.setattr(litellm, "model_cost", {})
    router: Final = Router(Registry((deployment("model"),)))
    assert native.is_codex_native_litellm_turn(route, "model", router, "team") == (
        route == "/backend-api/codex/responses"
    )
    assert bool(router.reader.calls) == (route == "/backend-api/codex/responses")
