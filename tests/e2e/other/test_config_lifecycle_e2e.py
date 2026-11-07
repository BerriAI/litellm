"""Live e2e: the proxy's configuration lifecycle - what the running process took
from its config file at boot, and what it accepts at runtime without a restart.

Boot is only observable through the state the process now exposes, so each test
keys off something that exists *only* because the startup config loader ran:

- the config file's `model_list` is live in the routing catalog. /model/info marks
  a deployment the process read from its file with `db_model: false` (a DB-stored
  deployment is `true`), so a `db_model: false` entry that serves a real
  completion is the file's model_list loaded, credentials and all. The file's
  `general_settings` are proven live the same way: /model/new only persists a
  deployment when `store_model_in_db` came off that file, and the deployment comes
  back from the catalog flagged `db_model: true`
- the `os.environ/` references in that file were resolved into real values. A
  credential reference is a poor witness: the router resolves `os.environ/` refs
  again at call time, so a completion would succeed even if boot-time resolution
  had never happened. The cache block is not re-resolved anywhere - the response
  cache is built once at startup from `cache_params.host` / `.port` - so a
  redis-typed cache that actually serves a repeated request can only exist if
  those two references resolved at boot; an unresolved literal would leave the
  cache dialling the hostname "os.environ/REDIS_HOST" and nothing would ever hit
- /config/update reaches the running process, not just the DB. The test adds a
  uniquely named model-group alias, reads it back off the live router through
  /get/config/callbacks (process state, not a DB row), drives a completion through
  the alias, then removes it and watches the alias stop resolving. The alias is
  namespaced per run, and every write goes through `_write_alias`, which compares
  the aliases the test does not own against a baseline taken at the start, names
  and targets both, before writing and again after. Anything else appearing,
  vanishing or being repointed stops the test with a message naming the
  difference, and nothing is written, so a concurrent writer's change is never
  reverted with a stale view. A second alias of the test's own stands across the
  whole exchange and is asserted to survive. /config/update rewrites the alias map
  wholesale and offers neither a per-key write nor a version to compare against,
  so this is the honest ceiling: the exposure becomes a loud, diagnosable failure
  rather than silent data loss

These tests assume the proxy under test was booted from a config file that wires
the example models and a redis cache through `os.environ/` references, which is
the setup tests/e2e/CONTRIBUTING.md prescribes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict

from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import NoBody, StreamingResponse, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

CACHE_HIT_HEADER = "x-litellm-cache-key"
CACHE_HIT_DEADLINE_SECONDS = 30.0


class CatalogModelInfo(BaseModel):
    """The /model/info `model_info` block, narrowed to the flag that says where the
    deployment came from: `db_model` is false for a deployment read off the config
    file at startup and true for one stored in the DB."""

    db_model: bool = False


class CatalogEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_info: CatalogModelInfo = CatalogModelInfo()


class CatalogResponse(BaseModel):
    data: list[CatalogEntry] = []


class CacheReadiness(BaseModel):
    """GET /health/readiness/details, narrowed to the cache the process built at
    startup (`litellm.cache.type`)."""

    status: str
    cache: str | None = None


class LiveRouterSettings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_group_alias: dict[str, str] = {}


class ConfigCallbacksResponse(BaseModel):
    """GET /get/config/callbacks. `router_settings` is read straight off the live
    Router object, so it reports what the running process is using rather than
    what any config row holds."""

    router_settings: LiveRouterSettings


class RouterSettingsUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_group_alias: dict[str, str]


class ConfigUpdateBody(BaseModel):
    router_settings: RouterSettingsUpdate


class ConfigUpdateResult(BaseModel):
    message: str


class CompletionId(BaseModel):
    id: str


@dataclass(frozen=True, slots=True)
class ConfigClient:
    """The catalog, health and config routes these tests read the proxy's own
    configuration state back through."""

    proxy: ProxyClient

    def catalog(self) -> list[CatalogEntry]:
        return unwrap(
            self.proxy.transport.get(
                "/model/info",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=CatalogResponse,
            )
        ).data

    def cache_readiness(self) -> CacheReadiness:
        return unwrap(
            self.proxy.transport.get(
                "/health/readiness/details",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=CacheReadiness,
            )
        )

    def live_router_settings(self) -> LiveRouterSettings:
        return unwrap(
            self.proxy.transport.get(
                "/get/config/callbacks",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=ConfigCallbacksResponse,
            )
        ).router_settings

    def set_model_group_alias(self, aliases: dict[str, str]) -> None:
        _ = unwrap(
            self.proxy.transport.post(
                "/config/update",
                headers=self.proxy.transport.master,
                json=ConfigUpdateBody(router_settings=RouterSettingsUpdate(model_group_alias=aliases)),
                response_type=ConfigUpdateResult,
            )
        )

    def chat_status(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=8,
            ),
        )


@pytest.fixture
def config(proxy: ProxyClient) -> ConfigClient:
    return ConfigClient(proxy=proxy)


def _poll[T](attempt: Callable[[], T | None], *, deadline_seconds: float, interval: float, failure: str) -> T:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(interval)
    pytest.fail(failure)


def _entry(catalog: list[CatalogEntry], model_name: str) -> CatalogEntry | None:
    return next((entry for entry in catalog if entry.model_name == model_name), None)


def _foreign_aliases(live: dict[str, str], owned: tuple[str, ...]) -> dict[str, str]:
    """Every alias the test does not own, names and targets both."""
    return {name: target for name, target in live.items() if name not in owned}


def _write_alias(
    config: ConfigClient,
    *,
    alias: str,
    target: str | None,
    owned: tuple[str, ...],
    baseline: dict[str, str],
) -> None:
    """Set or remove one owned alias, refusing to write at all if anything else in
    the map has moved.

    /config/update is the only route that writes model_group_alias, it takes the
    whole map, and it offers neither a per-key write nor a version to compare
    against, so every writer is a read-modify-write that can revert a concurrent
    change. Rather than paper over that, the aliases the test does not own are
    compared against `baseline` as full name/target pairs before the write and
    again after it: a name that appeared or vanished, or an existing alias
    repointed at a different model group, stops the test with a message naming the
    difference instead of being quietly overwritten with a stale value. Nothing is
    written once the map has moved, so the other writer's state stands.
    """
    live = config.live_router_settings().model_group_alias
    _assert_unmoved(_foreign_aliases(live, owned), baseline, when=f"before writing {alias!r}")

    keep = _foreign_aliases(live, owned) | {name: live[name] for name in owned if name in live and name != alias}
    config.set_model_group_alias(keep if target is None else {**keep, alias: target})

    settled = config.live_router_settings().model_group_alias
    _assert_unmoved(_foreign_aliases(settled, owned), baseline, when=f"after writing {alias!r}")


def _release_alias(config: ConfigClient, alias: str) -> None:
    """Teardown safety net: drop one alias from the map as it stands right now.

    Deliberately assertion-free and built from a fresh read, so it removes the
    test's own entry without restoring anything else to an older value; teardown
    swallows failures, so a stop-the-test check here would be invisible anyway.
    """
    live = config.live_router_settings().model_group_alias
    config.set_model_group_alias({name: target for name, target in live.items() if name != alias})


def _assert_unmoved(current: dict[str, str], baseline: dict[str, str], *, when: str) -> None:
    added = sorted(name for name in current if name not in baseline)
    removed = sorted(name for name in baseline if name not in current)
    repointed = sorted(name for name, target in baseline.items() if name in current and current[name] != target)
    assert current == baseline, (
        f"the model_group_alias map moved under the test {when}: added {added}, removed {removed}, "
        f"repointed {repointed}. /config/update rewrites the whole map, so continuing would write a stale "
        "view back over another writer's change; nothing was written"
    )


class TestStartupConfigLoad:
    @pytest.mark.covers("other.lifecycle.startup.config_loads")
    def test_config_file_model_list_and_general_settings_are_live(
        self, config: ConfigClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        from_file = _entry(config.catalog(), CHEAP_OPENAI_MODEL)
        assert from_file is not None, (
            f"the proxy serves no {CHEAP_OPENAI_MODEL!r} deployment, so its config file's model_list never loaded"
        )
        assert not from_file.model_info.db_model, (
            f"{CHEAP_OPENAI_MODEL!r} is flagged db_model=true, so it came from the DB rather than the config "
            "file the process booted with; the file's model_list is not what is being served"
        )

        served = config.chat_status(scoped_key, CHEAP_OPENAI_MODEL, f"reply with one word {unique_marker()}")
        assert served.status_code == 200, (
            f"the deployment loaded from the config file must serve a real completion, got "
            f"{served.status_code}: {served.body[:300]}"
        )

        stored_name = f"e2e-boot-cfg-{unique_marker()}"
        model_id = config.proxy.create_model(
            stored_name,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="e2e-dummy-key"),
        )
        resources.defer(lambda: config.proxy.delete_model(model_id))

        stored = _entry(config.catalog(), stored_name)
        assert stored is not None, f"{stored_name!r} is absent from /model/info right after /model/new"
        assert stored.model_info.db_model, (
            f"{stored_name!r} came back flagged db_model=false; the config file's general_settings "
            "(store_model_in_db) are not in effect on the running process"
        )

    @pytest.mark.covers("other.lifecycle.startup.env_vars_resolved")
    def test_env_referenced_cache_block_resolved_at_startup(self, config: ConfigClient, scoped_key: str) -> None:
        readiness = config.cache_readiness()
        assert readiness.cache == "redis", (
            f"the process reports cache {readiness.cache!r}; the config file's redis cache_params block, "
            "whose host and port are os.environ/ references, is not the cache the proxy built at startup"
        )

        prompt = f"reply with one word {unique_marker()}"
        first = config.chat_status(scoped_key, CHEAP_OPENAI_MODEL, prompt)
        assert first.status_code == 200, (
            f"the call being cached must succeed first, got {first.status_code}: {first.body[:300]}"
        )
        assert CACHE_HIT_HEADER not in first.headers, (
            f"a first-of-its-kind prompt came back as a cache hit ({first.headers.get(CACHE_HIT_HEADER)}), "
            "so the repeat below would prove nothing"
        )

        repeated = _poll(
            lambda: (lambda outcome: outcome if CACHE_HIT_HEADER in outcome.headers else None)(
                config.chat_status(scoped_key, CHEAP_OPENAI_MODEL, prompt)
            ),
            deadline_seconds=CACHE_HIT_DEADLINE_SECONDS,
            interval=5.0,
            failure=(
                "an identical repeat call was never served from the redis response cache, so the "
                "os.environ/ host and port that cache was configured with never resolved into a "
                "reachable redis at startup"
            ),
        )
        assert CompletionId.model_validate_json(repeated.body).id == CompletionId.model_validate_json(first.body).id, (
            "the repeat call carried a cache-key header but returned a different completion, so it was "
            "not the stored response coming back"
        )


class TestRuntimeConfigUpdate:
    @pytest.mark.covers("other.config.runtime_update.applies_at_runtime")
    def test_config_update_reaches_the_running_router_and_can_be_taken_back(
        self, config: ConfigClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        alias = f"e2e-cfg-alias-{unique_marker()}"
        bystander = f"e2e-cfg-bystander-{unique_marker()}"
        owned = (alias, bystander)
        baseline = _foreign_aliases(config.live_router_settings().model_group_alias, owned)
        assert alias not in baseline and bystander not in baseline, (
            f"the run's aliases {owned} are somehow already configured"
        )

        resources.defer(lambda: _release_alias(config, bystander))
        _write_alias(config, alias=bystander, target=CHEAP_OPENAI_MODEL, owned=owned, baseline=baseline)

        unknown = config.chat_status(scoped_key, alias, "should not route yet")
        assert unknown.status_code == 400, (
            f"an unconfigured model group must be rejected 400 before the update, got {unknown.status_code}: "
            f"{unknown.body[:300]}"
        )

        resources.defer(lambda: _release_alias(config, alias))
        _write_alias(config, alias=alias, target=CHEAP_OPENAI_MODEL, owned=owned, baseline=baseline)

        applied = _poll(
            lambda: (lambda live: live if live.model_group_alias.get(alias) == CHEAP_OPENAI_MODEL else None)(
                config.live_router_settings()
            ),
            deadline_seconds=config.proxy.poll_timeout,
            interval=5.0,
            failure=(
                f"the live router never picked up model_group_alias {alias!r} after /config/update, so the "
                "update only reached the DB and would need a restart to take effect"
            ),
        )
        assert applied.model_group_alias[alias] == CHEAP_OPENAI_MODEL

        routed = config.chat_status(scoped_key, alias, f"reply with one word {unique_marker()}")
        assert routed.status_code == 200, (
            f"the alias added at runtime must route to {CHEAP_OPENAI_MODEL!r}, got {routed.status_code}: "
            f"{routed.body[:300]}"
        )

        _write_alias(config, alias=alias, target=None, owned=owned, baseline=baseline)

        _poll(
            lambda: (
                True
                if config.chat_status(scoped_key, alias, "should not route any more").status_code == 400
                else None
            ),
            deadline_seconds=config.proxy.poll_timeout,
            interval=5.0,
            failure=f"the alias {alias!r} still routed after being removed at runtime",
        )
        settled = config.live_router_settings().model_group_alias
        assert alias not in settled, f"the live router still carries {alias!r} after the removing /config/update"
        assert settled.get(bystander) == CHEAP_OPENAI_MODEL, (
            f"removing {alias!r} also took {bystander!r} with it; /config/update replaces the alias map "
            "wholesale, so a caller that writes back anything other than the map as it currently stands "
            "wipes aliases it never touched"
        )
