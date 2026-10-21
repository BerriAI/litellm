import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.redis_process import owned_redis

pytestmark: Final = pytest.mark.timeout(300)

PROVIDER_KEY: Final = "integration-provider-key"
EVICTION_WARNING: Final = "config cache eviction of router_settings failed"
REFUSED: Final = frozenset({401, 403})


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as client:
        client.get("/__observations").raise_for_status()
        yield client


def _observed_requests(upstream: httpx.Client) -> list[JsonValue]:
    observed: Final = upstream.get("/__observations")
    observed.raise_for_status()
    requests: Final = object_value(observed.json())["requests"]
    assert isinstance(requests, list)
    return requests


def _calls_to(observed: list[JsonValue], provider_model: str) -> int:
    return sum(object_value(object_value(request)["body"]).get("model") == provider_model for request in observed)


def _fallback_body(model: str, fallback_models: list[str], fallback_type: str = "general") -> dict[str, JsonValue]:
    return {"model": model, "fallback_models": list(fallback_models), "fallback_type": fallback_type}


def _forget_fallback(gateway: Gateway, model: str, fallback_type: str) -> None:
    gateway.request("DELETE", f"/fallback/{model}", params={"fallback_type": fallback_type})


def _create_fallback(
    gateway: Gateway, scenario: Scenario, model: str, fallback_models: list[str], fallback_type: str = "general"
) -> httpx.Response:
    scenario.cleanups.callback(_forget_fallback, gateway, model, fallback_type)
    return eventually(
        lambda: gateway.request("POST", "/fallback", _fallback_body(model, fallback_models, fallback_type)),
        lambda response: response.status_code == 200,
        seconds=30,
        return_last_on_timeout=True,
    )


def _fallback_models(response: httpx.Response) -> list[str]:
    body: Final = response.json()
    models: Final = body.get("fallback_models") if isinstance(body, dict) else None
    return [string_value(entry) for entry in models] if isinstance(models, list) else []


def _available_models(response: httpx.Response) -> list[str]:
    body: Final = response.json()
    if not isinstance(body, dict):
        return []
    detail: Final = body.get("detail")
    source: Final = detail if isinstance(detail, dict) else body
    models: Final = source.get("available_models")
    return [string_value(entry) for entry in models] if isinstance(models, list) else []


def _stored_fallbacks(fallback_key: str = "fallbacks") -> list[JsonValue]:
    rows: Final = read_rows('SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s', ("router_settings",))
    if not rows:
        return []
    settings: Final = object_value(rows[0]["param_value"])
    entries: Final = settings.get(fallback_key) or []
    assert isinstance(entries, list), entries
    return entries


def _covered_models(entries: list[JsonValue]) -> frozenset[str]:
    dict_entries: Final = (object_value(entry) for entry in entries if isinstance(entry, dict))
    return frozenset(chain.from_iterable(dict_entries))


def _models_over_a_fresh_connection(gateway: Gateway, _: int) -> frozenset[str]:
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        listed: Final = client.get("/v1/models", headers={"Authorization": f"Bearer {gateway.key}"})
    assert listed.status_code == 200, listed.text
    data: Final = object_value(listed.json())["data"]
    assert isinstance(data, list), listed.text
    return frozenset(string_value(object_value(entry)["id"]) for entry in data)


def _every_worker_serves(gateway: Gateway, model: str) -> bool:
    with ThreadPoolExecutor(max_workers=16) as pool:
        rounds: Final = tuple(
            tuple(pool.map(partial(_models_over_a_fresh_connection, gateway), range(16))) for _ in range(2)
        )
    return all(model in seen for seen in chain.from_iterable(rounds))


def _wait_until_served(gateway: Gateway, model: str) -> None:
    eventually(lambda: _every_worker_serves(gateway, model), lambda served: served, seconds=90)


def _team_model(gateway: Gateway, scenario: Scenario, team: str, provider_model: str) -> tuple[str, str]:
    public: Final = f"ipub-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": public,
            "litellm_params": {
                "model": f"openai/{provider_model}",
                "api_key": PROVIDER_KEY,
                "api_base": f"{gateway.upstream_url}/v1",
                "num_retries": 0,
            },
            "model_info": {"team_id": team},
        },
    )
    info: Final = object_value(created["model_info"])
    scenario.cleanups.callback(scenario.delete_model, string_value(info["id"]))
    return public, string_value(created["model_name"])


def _workers(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    return tuple(child for child in psutil.Process(owned.process.pid).children() if _is_worker(child))


def _is_worker(child: psutil.Process) -> bool:
    try:
        return "spawn_main" in " ".join(child.cmdline()) and child.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def test_post_by_team_public_name_creates_reads_back_and_peer_converges(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        primary: Final = scenario.model(model=f"openai/n1p-{uuid.uuid4().hex}", model_info={"team_id": team})
        fallback: Final = scenario.model(model=f"openai/n1f-{uuid.uuid4().hex}", model_info={"team_id": team})
        created: Final = _create_fallback(gateway, scenario, primary, [fallback])
        assert created.status_code == 200, created.text
        here: Final = eventually(
            lambda: gateway.request("GET", f"/fallback/{primary}", params={"fallback_type": "general"}),
            lambda response: response.status_code == 200 and fallback in _fallback_models(response),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert here.status_code == 200 and fallback in _fallback_models(here), here.text
        there: Final = eventually(
            lambda: peer.request("GET", f"/fallback/{primary}", params={"fallback_type": "general"}),
            lambda response: response.status_code == 200 and fallback in _fallback_models(response),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert there.status_code == 200 and fallback in _fallback_models(there), there.text


def test_rule_by_team_public_name_fires_on_chat_completions(gateway: Gateway, upstream: httpx.Client) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        primary_provider: Final = f"n2prim-{uuid.uuid4().hex}"
        fallback_provider: Final = f"n2fbk-{uuid.uuid4().hex}"
        primary: Final = scenario.model(model=f"openai/{primary_provider}", num_retries=0, model_info={"team_id": team})
        fallback: Final = scenario.model(model=f"openai/{fallback_provider}", model_info={"team_id": team})
        team_key: Final = scenario.key(team_id=team)
        created: Final = _create_fallback(gateway, scenario, primary, [fallback])
        assert created.status_code == 200, created.text
        upstream.post(f"/__scripts/{primary_provider}", json={"statuses": [500]}).raise_for_status()
        upstream.get("/__observations").raise_for_status()
        answered: Final = eventually(
            lambda: gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": primary, "messages": [{"role": "user", "content": f"fire {uuid.uuid4().hex}"}]},
                key=team_key,
            ),
            lambda response: response.status_code == 200,
            seconds=60,
            return_last_on_timeout=True,
        )
        assert answered.status_code == 200, answered.text
        observed: Final = _observed_requests(upstream)
        assert _calls_to(observed, primary_provider) >= 1, observed
        assert _calls_to(observed, fallback_provider) >= 1, observed


def test_consecutive_creates_within_the_cache_window_keep_every_rule(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        target: Final = scenario.model(model=f"openai/n5t-{uuid.uuid4().hex}")
        primaries: Final = tuple(scenario.model(model=f"openai/n5{tag}-{uuid.uuid4().hex}") for tag in "abc")
        for model in (target, *primaries):
            _wait_until_served(gateway, model)
        created: Final = tuple(_create_fallback(gateway, scenario, primary, [target]) for primary in primaries)
        assert all(response.status_code == 200 for response in created), [response.text for response in created]
        covered: Final = _covered_models(_stored_fallbacks())
        assert frozenset(primaries) <= covered, (primaries, covered)


def test_unknown_model_404_lists_team_public_and_internal_names(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        public, internal = _team_model(gateway, scenario, team, f"n6-{uuid.uuid4().hex}")
        unknown: Final = f"n6-unknown-{uuid.uuid4().hex}"
        refused: Final = eventually(
            lambda: gateway.request("POST", "/fallback", _fallback_body(unknown, [public])),
            lambda response: response.status_code == 404 and internal in _available_models(response),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert refused.status_code == 404, refused.text
        available: Final = _available_models(refused)
        assert internal in available, (internal, available)
        assert public in available, (public, available)


def test_create_by_generated_internal_name_still_works(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        _, primary_internal = _team_model(gateway, scenario, team, f"c1p-{uuid.uuid4().hex}")
        _, fallback_internal = _team_model(gateway, scenario, team, f"c1f-{uuid.uuid4().hex}")
        created: Final = _create_fallback(gateway, scenario, primary_internal, [fallback_internal])
        assert created.status_code == 200, created.text
        here: Final = eventually(
            lambda: gateway.request("GET", f"/fallback/{primary_internal}", params={"fallback_type": "general"}),
            lambda response: response.status_code == 200 and fallback_internal in _fallback_models(response),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert here.status_code == 200 and fallback_internal in _fallback_models(here), here.text


def test_delete_reads_fresh_rules_and_keeps_the_others(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        target: Final = scenario.model(model=f"openai/c2t-{uuid.uuid4().hex}")
        keep: Final = scenario.model(model=f"openai/c2k-{uuid.uuid4().hex}")
        drop: Final = scenario.model(model=f"openai/c2d-{uuid.uuid4().hex}")
        _wait_until_served(gateway, target)
        _wait_until_served(gateway, keep)
        _wait_until_served(gateway, drop)
        assert _create_fallback(gateway, scenario, keep, [target]).status_code == 200
        assert _create_fallback(gateway, scenario, drop, [target]).status_code == 200
        removed: Final = gateway.request("DELETE", f"/fallback/{drop}", params={"fallback_type": "general"})
        assert removed.status_code == 200, removed.text
        covered: Final = _covered_models(_stored_fallbacks())
        assert keep in covered, covered
        assert drop not in covered, covered
        gone: Final = gateway.request("GET", f"/fallback/{drop}", params={"fallback_type": "general"})
        assert gone.status_code == 404, gone.text


def test_self_fallback_and_unknown_fallback_models_are_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/c3-{uuid.uuid4().hex}")
        _wait_until_served(gateway, model)
        itself: Final = gateway.request("POST", "/fallback", _fallback_body(model, [model]))
        assert itself.status_code == 400, itself.text
        unknown_target: Final = gateway.request(
            "POST", "/fallback", _fallback_body(model, [f"c3-missing-{uuid.uuid4().hex}"])
        )
        assert unknown_target.status_code == 400, unknown_target.text


def test_malformed_requests_are_rejected_and_the_proxy_stays_up(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/c4-{uuid.uuid4().hex}")
        fallback: Final = scenario.model(model=f"openai/c4f-{uuid.uuid4().hex}")
        _wait_until_served(gateway, model)
        _wait_until_served(gateway, fallback)
        malformed: Final = (
            {"model": model, "fallback_models": [], "fallback_type": "general"},
            {"model": model, "fallback_type": "general"},
            {"model": model, "fallback_models": [fallback], "fallback_type": "nonsense"},
            {"model": 123, "fallback_models": [fallback], "fallback_type": "general"},
        )
        for body in malformed:
            assert gateway.request("POST", "/fallback", body).status_code in (400, 422), body
        healthy: Final = _create_fallback(gateway, scenario, model, [fallback])
        assert healthy.status_code == 200, healthy.text
        listed: Final = gateway.request("GET", "/v1/models")
        assert listed.status_code == 200, listed.text


def test_team_key_is_forbidden_on_create_and_delete(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        model: Final = scenario.model(model=f"openai/c5-{uuid.uuid4().hex}", model_info={"team_id": team})
        fallback: Final = scenario.model(model=f"openai/c5f-{uuid.uuid4().hex}", model_info={"team_id": team})
        team_key: Final = scenario.key(team_id=team)
        creating: Final = gateway.request("POST", "/fallback", _fallback_body(model, [fallback]), key=team_key)
        assert creating.status_code in REFUSED, creating.text
        assert model not in _covered_models(_stored_fallbacks())
        admitted: Final = _create_fallback(gateway, scenario, model, [fallback])
        assert admitted.status_code == 200, admitted.text
        deleting: Final = gateway.request(
            "DELETE", f"/fallback/{model}", params={"fallback_type": "general"}, key=team_key
        )
        assert deleting.status_code in REFUSED, deleting.text
        assert model in _covered_models(_stored_fallbacks())


def test_context_window_and_content_policy_types_persist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        target: Final = scenario.model(model=f"openai/c6t-{uuid.uuid4().hex}")
        windowed: Final = scenario.model(model=f"openai/c6w-{uuid.uuid4().hex}")
        policy: Final = scenario.model(model=f"openai/c6p-{uuid.uuid4().hex}")
        _wait_until_served(gateway, target)
        _wait_until_served(gateway, windowed)
        _wait_until_served(gateway, policy)
        window_create: Final = _create_fallback(gateway, scenario, windowed, [target], "context_window")
        assert window_create.status_code == 200, window_create.text
        window_read: Final = eventually(
            lambda: gateway.request("GET", f"/fallback/{windowed}", params={"fallback_type": "context_window"}),
            lambda response: response.status_code == 200 and target in _fallback_models(response),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert window_read.status_code == 200 and target in _fallback_models(window_read), window_read.text
        policy_create: Final = _create_fallback(gateway, scenario, policy, [target], "content_policy")
        assert policy_create.status_code == 200, policy_create.text
        covered: Final = _covered_models(_stored_fallbacks("content_policy_fallbacks"))
        assert policy in covered, covered


def test_fallback_write_survives_a_redis_outage(gateway: Gateway, tmp_path: Path) -> None:
    with owned_redis(tmp_path) as cache:
        overrides: Final = {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}
        with owned_proxy_process(gateway, tmp_path, overrides, workers=2, database_setup=()) as owned:
            with owned.gateway.scenario() as scenario:
                target: Final = scenario.model(model=f"openai/x1t-{uuid.uuid4().hex}")
                first: Final = scenario.model(model=f"openai/x1a-{uuid.uuid4().hex}")
                second: Final = scenario.model(model=f"openai/x1b-{uuid.uuid4().hex}")
                third: Final = scenario.model(model=f"openai/x1c-{uuid.uuid4().hex}")
                for model in (target, first, second, third):
                    _wait_until_served(owned.gateway, model)
                assert _create_fallback(owned.gateway, scenario, first, [target]).status_code == 200
                cache.stop()
                degraded: Final = tuple(
                    _create_fallback(owned.gateway, scenario, model, [target]) for model in (second, third)
                )
                assert all(response.status_code == 200 for response in degraded), [r.text for r in degraded]
                covered: Final = _covered_models(_stored_fallbacks())
                assert {first, second, third} <= covered, covered
                assert EVICTION_WARNING in owned.log.read_text(), owned.log.read_text()[-3000:]
                cache.start()


def test_fallback_write_survives_a_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, workers=2, database_setup=()) as owned:
        with owned.gateway.scenario() as scenario:
            target: Final = scenario.model(model=f"openai/x2t-{uuid.uuid4().hex}")
            before: Final = scenario.model(model=f"openai/x2a-{uuid.uuid4().hex}")
            after: Final = scenario.model(model=f"openai/x2b-{uuid.uuid4().hex}")
            last: Final = scenario.model(model=f"openai/x2c-{uuid.uuid4().hex}")
            for model in (target, before, after, last):
                _wait_until_served(owned.gateway, model)
            assert _create_fallback(owned.gateway, scenario, before, [target]).status_code == 200
            victim: Final = eventually(lambda: _workers(owned), lambda workers: len(workers) == 2)[0]
            victim.kill()
            fresh: Final = scenario.cleanups.enter_context(
                httpx.Client(base_url=owned.gateway.client.base_url, timeout=15, trust_env=False)
            )
            survivor: Final = Gateway(fresh, owned.gateway.key, owned.gateway.upstream_url)
            degraded: Final = tuple(_create_fallback(survivor, scenario, model, [target]) for model in (after, last))
            assert all(response.status_code == 200 for response in degraded), [r.text for r in degraded]
            covered: Final = _covered_models(_stored_fallbacks())
            assert {before, after, last} <= covered, covered
