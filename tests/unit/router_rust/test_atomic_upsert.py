from collections.abc import Callable
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.proxy import proxy_server
from litellm.proxy._types import ReconcileOutcome
from litellm.proxy.management_endpoints.model_management_endpoints import clear_cache
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.router_python_host import Endpoint
from tests.unit.test_router.core.registry_helpers import typed
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rust_core


class ObservingHost(ScriptedHost):
    def __init__(self) -> None:
        super().__init__()
        self.observe: Callable[[], None] | None = None

    def invoke(self, op: str) -> Endpoint:
        endpoint: Final = super().invoke(op)
        if op != "get_llm_provider":
            return endpoint

        def provider(*args: object, **kwargs: object) -> object:  # kwargs-ok: provider resolver arguments
            if self.observe is not None:
                self.observe()
            return endpoint(*args, **kwargs)

        return provider


@pytest.mark.asyncio
async def test_upsert_keeps_the_previous_deployment_serving_while_resolving_its_replacement() -> None:
    host: Final = ObservingHost()
    core: Final[RouterCore] = rust_core(host, model_list=[deployment("group", "only")])
    seen: Final[list[tuple[str, ...]]] = []  # mutable-ok: observations made by the injected provider resolver

    def observe() -> None:
        seen.append(tuple(str(item["model_info"]["id"]) for item in core.registry().deployments("group")))

    host.observe = observe
    result: Final = core.apply(ChangeSet(upsert=[typed("group", "only", api_base="https://replacement.example")]))

    assert seen and set(seen) == {("only",)}, seen
    assert list(result.upserted) == [typed("group", "only", api_base="https://replacement.example")]
    await core.acompletion(model="group", messages=[{"role": "user", "content": "after replacement"}])  # pyright: ignore[reportAttributeAccessIssue]  # the core's completion endpoint
    assert host.deployments_called("acompletion") == ["only"]


@pytest.mark.asyncio
async def test_clear_cache_keeps_a_db_classifier_serving_while_the_reload_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host: Final = ScriptedHost()
    core: Final = rust_core(
        host,
        model_list=[
            deployment("tier", "tier"),
            deployment(
                "smart",
                "router",
                model="auto_router/complexity_router",
                complexity_router_config={
                    "classifier_type": "heuristic_v2",
                    "tiers": {"SIMPLE": "tier", "MEDIUM": "tier", "COMPLEX": "tier", "REASONING": "tier"},
                },
                model_info={"db_model": True},
            ),
        ],
    )
    seen: Final[list[tuple[str, ...]]] = []  # mutable-ok: snapshots at the reload boundary

    async def reload(**kwargs: object) -> ReconcileOutcome:  # kwargs-ok: the config reload's injected I/O boundary
        seen.append(tuple(str(item["model_info"]["id"]) for item in core.registry().deployments("smart")))
        await core.acompletion(model="smart", messages=[{"role": "user", "content": "hello"}])  # pyright: ignore[reportAttributeAccessIssue]  # the core's completion endpoint
        return ReconcileOutcome(still_desired=frozenset({"router"}), live_after=frozenset({"router"}))

    monkeypatch.setattr(proxy_server, "llm_router", core)
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "proxy_config", SimpleNamespace(add_deployment_locked=reload))

    outcome: Final = await clear_cache()

    assert seen == [("router",)]
    assert outcome == ReconcileOutcome(still_desired=frozenset({"router"}), live_after=frozenset({"router"}))
    assert host.deployments_called("acompletion") == ["tier"]
