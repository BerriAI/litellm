from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final, cast

import pytest
from starlette.requests import Request

import litellm
from litellm._auth.claude_normalization import ClaudeNormalizationFacts, claude_membership, claude_normalization
from litellm._auth.failure import NativeAuthRuntimeError
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.claude_normalization_adapter import normalize_claude_model, read_claude_membership_facts
from litellm.router_api import RouterRegistry
from litellm.rust_bridge import _native
from tests.unit._auth.support import claude_reference as reference


@dataclass(frozen=True, slots=True)
class Trace:
    fail_at: str | None = None
    calls: list[tuple[object, ...]] = field(default_factory=list)  # mutable-ok: ordered raw IO observations

    def read(self, operation: str, *values: object) -> None:
        self.calls.append((operation, *values))
        if operation == self.fail_at:
            raise ValueError(operation)


@dataclass(frozen=True, slots=True)
class Registry:
    trace: Trace
    alias_names: tuple[str, ...] = ()
    deployments: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()

    def aliases(self) -> Mapping[str, object]:
        self.trace.read("aliases")
        return dict.fromkeys(self.alias_names)

    def has_deployment(self, name: str) -> bool:
        self.trace.read("deployment", name)
        return name in self.deployments

    def names(self, name: str, *, kind: str, team_id: str | None) -> tuple[str, ...]:
        self.trace.read("names", name, kind, team_id)
        return ("deployment-id",) if name in self.candidates else ()


@dataclass(frozen=True, slots=True)
class Router:
    reader: Registry

    def registry(self) -> RouterRegistry:
        self.reader.trace.read("registry")
        return cast(RouterRegistry, self.reader)  # cast-ok: raw registry fixture supplies the selected IO


@dataclass(frozen=True, slots=True)
class Settings:
    trace: Trace
    aliases: object = None

    async def get_hierarchical_router_settings(
        self, *, user_api_key_dict: UserAPIKeyAuth, prisma_client: object, proxy_logging_obj: object
    ) -> Mapping[str, object]:
        self.trace.read("settings")
        return {"model_group_alias": self.aliases}


@dataclass(frozen=True, slots=True)
class LookupOnly(Mapping[str, object]):
    entries: Mapping[str, object]
    trace: Trace

    def __getitem__(self, key: str) -> object:
        self.trace.read("lookup", key)
        return self.entries[key]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("unrequested whole-map read")


@pytest.mark.parametrize("present", (False, True))
def test_membership_only_reads_the_native_selected_alias_name(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {})
    trace: Final = Trace()
    aliases: Final = LookupOnly({"picker": None} if present else {}, trace)
    facts: Final = read_claude_membership_facts("picker", None, None, (aliases,))
    assert claude_membership(facts).present is present
    assert trace.calls == [("lookup", "picker")]


_PICKER: Final = "claude-router-" + "model".encode().hex()
_MODEL_NAMES: Final = (
    "model",
    "model[1m]",
    "model[1M]",
    _PICKER,
    _PICKER + "[1m]",
    "claude-router-6D6F64656C",
    "claude-router-6d 6f64656c",
    "claude-router-ff",
    "claude-router-1",
    "claude-router-",
    "claude-router-zz[1m]",
    "claude-router-" + "模型".encode().hex(),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _MODEL_NAMES)
@pytest.mark.parametrize("source", ("global", "key", "team", "settings", "registry", "deployment", "candidates"))
@pytest.mark.parametrize("claim", ("requested", "target", "missing"))
async def test_normalization_matches_frozen_main_rewrite_scope_and_read_order(
    monkeypatch: pytest.MonkeyPatch, model: str, source: str, claim: str
) -> None:
    name: Final = model if claim == "requested" else (reference.decoded(model) or reference.unmarked(model))
    names: Final = () if claim == "missing" else (name,)
    monkeypatch.setattr(litellm, "model_alias_map", dict.fromkeys(names, "target") if source == "global" else {})
    token: Final = UserAPIKeyAuth(
        team_id="team",
        aliases=dict.fromkeys(names, "target") if source == "key" else {},
        team_model_aliases=dict.fromkeys(names, "target") if source == "team" else {},
    )

    async def run(native: bool) -> tuple[Mapping[str, object], object, tuple[tuple[object, ...], ...]]:
        trace: Final = Trace()
        router: Final = Router(
            Registry(
                trace,
                names if source == "registry" else (),
                names if source == "deployment" else (),
                names if source == "candidates" else (),
            )
        )
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(
            proxy_server, "proxy_config", Settings(trace, dict.fromkeys(names) if source == "settings" else None)
        )
        request: Final = Request({"type": "http", "path": "/v1/messages", "method": "POST", "headers": []})
        data: Final[dict[str, object]] = {"model": model, "messages": ()}  # mutable-ok: real request rewrite target
        await (normalize_claude_model if native else reference.normalize)(data, token, request, "/v1/messages")
        return data, request.scope.get("litellm.claude_model_normalized"), tuple(trace.calls)

    assert await run(True) == await run(False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    (
        "/messages",
        "/v1/messages",
        "/chat/completions",
        "/v1/chat/completions",
        "/responses",
        "/v1/responses",
        "/models",
        "/backend-api/codex/responses",
    ),
)
@pytest.mark.parametrize("router_present", (False, True))
@pytest.mark.parametrize("normalized", (False, True, 1))
@pytest.mark.parametrize("request_present", (False, True))
async def test_normalization_early_exits_and_idempotency_match_main(
    monkeypatch: pytest.MonkeyPatch, route: str, router_present: bool, normalized: object, request_present: bool
) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {})

    async def run(native: bool) -> tuple[Mapping[str, object], object, tuple[tuple[object, ...], ...]]:
        trace: Final = Trace()
        monkeypatch.setattr(
            proxy_server, "llm_router", Router(Registry(trace, deployments=("model",))) if router_present else None
        )
        monkeypatch.setattr(proxy_server, "proxy_config", Settings(trace))
        request: Final = (
            Request(
                {
                    "type": "http",
                    "path": route,
                    "method": "POST",
                    "headers": [],
                    "litellm.claude_model_normalized": normalized,
                }
            )
            if request_present
            else None
        )
        data: Final[dict[str, object]] = {"model": _PICKER}  # mutable-ok: real request rewrite target
        await (normalize_claude_model if native else reference.normalize)(data, UserAPIKeyAuth(), request, route)
        return (
            data,
            request.scope.get("litellm.claude_model_normalized") if request is not None else None,
            tuple(trace.calls),
        )

    assert await run(True) == await run(False)


@pytest.mark.parametrize("aliases", (None, [], {"picker": None}, {"other": "picker"}, {"picker": 42}))
@pytest.mark.parametrize("position", (0, 1, 2))
def test_raw_alias_membership_preserves_keys_not_targets(
    monkeypatch: pytest.MonkeyPatch, aliases: object, position: int
) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {})
    maps: Final = tuple(aliases if index == position else None for index in range(3))
    first: Final = Trace()
    second: Final = Trace()
    facts: Final = read_claude_membership_facts("picker", Router(Registry(first)), "team", maps)
    expected: Final = "picker" in reference.RoutingNames(Router(Registry(second)), "team", maps)
    assert claude_membership(facts).present == expected
    assert first.calls == second.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ("settings", "registry", "aliases", "deployment", "names"))
async def test_raw_read_errors_preserve_order_and_do_not_mark_request(
    monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {})

    async def run(native: bool) -> tuple[tuple[object, ...], ...]:
        trace: Final = Trace(fail_at)
        monkeypatch.setattr(proxy_server, "llm_router", Router(Registry(trace)))
        monkeypatch.setattr(proxy_server, "proxy_config", Settings(trace))
        request: Final = Request({"type": "http", "path": "/messages", "method": "POST", "headers": []})
        data: Final[dict[str, object]] = {"model": _PICKER}  # mutable-ok: real request rewrite target
        with pytest.raises(ValueError, match=fail_at):
            await (normalize_claude_model if native else reference.normalize)(
                data, UserAPIKeyAuth(), request, "/messages"
            )
        assert data == {"model": _PICKER}
        assert "litellm.claude_model_normalized" not in request.scope
        return tuple(trace.calls)

    assert await run(True) == await run(False)


def test_non_string_request_never_reads_body_model_or_settings() -> None:
    facts: Final = ClaudeNormalizationFacts("/messages", False, router_present=True, model_loaded=True)
    assert claude_normalization(facts).kind == "Complete"


@pytest.mark.asyncio
async def test_native_membership_failure_is_terminal_before_registry_io(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = NativeAuthRuntimeError("native membership failed")
    trace: Final = Trace()
    monkeypatch.setattr(proxy_server, "llm_router", Router(Registry(trace)))
    monkeypatch.setattr(proxy_server, "proxy_config", Settings(trace))

    def fail(_facts: object) -> object:
        raise failure

    monkeypatch.setattr(_native, "request_auth_claude_membership", fail)
    data: Final[dict[str, object]] = {"model": _PICKER}  # mutable-ok: real request rewrite target
    with pytest.raises(NativeAuthRuntimeError) as raised:
        await normalize_claude_model(data, UserAPIKeyAuth(), None, "/messages")
    assert raised.value is failure
    assert data == {"model": _PICKER}
    assert trace.calls == [("settings",)]
