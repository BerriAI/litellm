"""The probe surface an orchestrator and an on-call engineer rely on: liveness, readiness, details and /health.

Contract (litellm/proxy/health_endpoints/_health_endpoints.py):

* `GET /health/liveliness` and `/health/liveness` take no credential and answer the JSON string `"I'm alive!"`.
  A bad `Authorization` header must not turn them into a 401, because kubelets send none and some load
  balancers forward whatever the client sent.
* `GET /health/readiness` takes no credential and answers exactly `{"status": "healthy", "db": "connected"}`
  while the configured database answers; the detailed payload needs a key (`/health/readiness/details`).
* `OPTIONS` on the probes answers 200 with `Allow` and CORS headers for preflighting balancers.
* `GET /health` and `GET /health/services` need a key. `/health` probes every deployment, answers 200 with
  per-endpoint lists and counts, hides routing fields from non-admin keys behind a
  `Litellm-Health-Field-Notice` header, and answers 503 when a caller asks about one model and none of its
  deployments is healthy.
* `GET /health/backlog` needs a key and reports this worker's admission counters.

When the HTTP front door moves to Rust these routes must keep the same auth split, bodies and status codes,
because they are wired into Helm charts, load balancer health checks and dashboards that nobody re-reads.
Readiness during a database outage is in `database/test_dec1_failures_postgres_*`, the drain endpoint in
`management/test_health_routing.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration.configuration.dec1_deploy_support import (
    ALIVE,
    CHAT,
    READY,
    Provider,
    deploy_config,
    deployment,
    held_chat,
    provider,
    readiness,
    write_config,
)

pytestmark: Final = pytest.mark.timeout(600)

DEAD: Final = "dec1-deploy-dead"
DEAD_BASE: Final = "http://127.0.0.1:9/v1"
DETAIL_FIELDS: Final = frozenset(
    {
        "status",
        "db",
        "cache",
        "litellm_version",
        "success_callbacks",
        "use_aiohttp_transport",
        "log_level",
        "is_detailed_debug",
        "show_no_redis_warning",
        "show_env_credential_login_warning",
    }
)
NOTICE: Final = "Litellm-Health-Field-Notice"


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    upstream: Provider

    @property
    def proxy(self) -> Gateway:
        return self.owned.gateway


@pytest.fixture(scope="module")
def rig(gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-deploy-probes")
    with provider() as upstream:
        config: Final = write_config(
            directory, deploy_config(upstream.api_base, extra_models=(deployment(DEAD, DEAD_BASE),))
        )
        with owned_proxy_process(gateway, directory / "proxy", {}, config=config) as owned:
            yield _Rig(owned, upstream)


def _virtual_key(rig: _Rig) -> str:
    return str(rig.proxy.post("/key/generate", {"models": [CHAT, DEAD]})["key"])


def test_readiness_answers_exactly_healthy_and_connected_without_a_key(rig: _Rig) -> None:
    """A ready proxy with a reachable database gets `GET /health/readiness` with no `Authorization` header.
    It must answer 200 with exactly `{"status": "healthy", "db": "connected"}` and nothing more: the public
    probe must not leak versions or callbacks (`health_readiness`, _health_endpoints.py:2336)."""
    assert readiness(rig.proxy) == (200, dict(READY))


def test_readiness_ignores_a_garbage_bearer_token(rig: _Rig) -> None:
    """`GET /health/readiness` with `Authorization: Bearer not-a-real-key`. It must still answer 200 with the
    two field body: the probe has no auth dependency, so a balancer that forwards client headers cannot make
    the pod look unready."""
    response: Final = rig.proxy.client.get("/health/readiness", headers={"Authorization": "Bearer not-a-real-key"})
    assert (response.status_code, response.json()) == (200, dict(READY))


@pytest.mark.parametrize("path", ["/health/liveliness", "/health/liveness"])
def test_liveness_ignores_a_garbage_bearer_token(rig: _Rig, path: str) -> None:
    """Both liveness spellings get `Authorization: Bearer not-a-real-key`. Each must answer 200 with the JSON
    string `"I'm alive!"`: both routes are in `LiteLLMRoutes.public_routes` (litellm/proxy/_types.py:797)."""
    response: Final = rig.proxy.client.get(path, headers={"Authorization": "Bearer not-a-real-key"})
    assert (response.status_code, response.json()) == (200, ALIVE)


@pytest.mark.parametrize("path", ["/health/liveliness", "/health/liveness"])
def test_liveness_options_advertises_get_options_and_any_header(rig: _Rig, path: str) -> None:
    """`OPTIONS` on either liveness spelling must answer 200 with `Allow: GET, OPTIONS`,
    `Access-Control-Allow-Methods: GET, OPTIONS` and `Access-Control-Allow-Headers: *`
    (`health_liveliness_options`, _health_endpoints.py:2484)."""
    response: Final = rig.proxy.client.options(path)
    assert response.status_code == 200, response.text
    assert response.headers["allow"] == "GET, OPTIONS"
    assert response.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert response.headers["access-control-allow-headers"] == "*"


def test_readiness_options_allows_any_request_header(rig: _Rig) -> None:
    """`OPTIONS /health/readiness` must answer 200 with `Access-Control-Allow-Headers: *`, so a browser based
    status page can preflight the probe with custom headers."""
    response: Final = rig.proxy.client.options("/health/readiness")
    assert response.status_code == 200, response.text
    assert response.headers["access-control-allow-headers"] == "*"


def test_readiness_details_without_a_key_is_401(rig: _Rig) -> None:
    """`GET /health/readiness/details` with no credential must answer 401: unlike the public probe it reports
    the version, callbacks and log level, which are reconnaissance for an attacker."""
    assert rig.proxy.client.get("/health/readiness/details").status_code == 401


def test_readiness_details_for_the_master_key_has_exactly_the_documented_fields(rig: _Rig) -> None:
    """The master key calls `GET /health/readiness/details`. It must answer 200 with exactly the ten fields
    `status, db, cache, litellm_version, success_callbacks, use_aiohttp_transport, log_level,
    is_detailed_debug, show_no_redis_warning, show_env_credential_login_warning`, with `status: healthy` and
    `db: connected` (`_get_health_readiness_details`, _health_endpoints.py:2166)."""
    body: Final = rig.proxy.get("/health/readiness/details")
    assert frozenset(body) == DETAIL_FIELDS, sorted(body)
    assert (body["status"], body["db"]) == ("healthy", "connected")


def test_readiness_details_reports_no_cache_when_none_is_configured(rig: _Rig) -> None:
    """The config sets no `litellm_settings.cache`. `GET /health/readiness/details` must report `cache: null`,
    so a dashboard can tell an uncached deployment from a cache that is down."""
    assert rig.proxy.get("/health/readiness/details")["cache"] is None


def test_readiness_details_refuses_a_plain_virtual_key(rig: _Rig) -> None:
    """A virtual key with no admin role calls `GET /health/readiness/details`. It must be refused with 401 or
    403, never served: the route is in no non-admin route list in litellm/proxy/_types.py.
    UNVERIFIED: the exact status for a key without a user; read only that the route is absent from
    `info_routes` and `public_routes`."""
    response: Final = rig.proxy.request("GET", "/health/readiness/details", key=_virtual_key(rig))
    assert response.status_code in (401, 403), response.text


def test_health_without_a_key_is_401(rig: _Rig) -> None:
    """`GET /health` with no credential must answer 401: it makes real calls to every provider, so an
    anonymous caller could use it to burn provider quota."""
    assert rig.proxy.client.get("/health").status_code == 401


def test_health_for_the_master_key_counts_one_healthy_and_one_unhealthy_deployment(rig: _Rig) -> None:
    """The config has one deployment on the scripted provider and one on a port nothing listens on. The
    master key's `GET /health` must answer 200 with `healthy_count: 1` and `unhealthy_count: 1`: an overall
    probe stays 200 while any deployment is up, the counts carry the truth."""
    body: Final = rig.proxy.get("/health")
    assert (body["healthy_count"], body["unhealthy_count"]) == (1, 1), body


def test_health_for_the_master_key_shows_the_api_base_of_a_healthy_endpoint(rig: _Rig) -> None:
    """The master key's `GET /health?model=dec1-deploy-chat` must list one healthy endpoint whose `api_base`
    is the scripted provider's `/v1` URL: admins need it to tell which region or host failed."""
    body: Final = rig.proxy.get("/health", params={"model": CHAT})
    endpoints: Final = body["healthy_endpoints"]
    assert isinstance(endpoints, list) and len(endpoints) == 1, body
    assert JSON_OBJECT.validate_python(endpoints[0])["api_base"] == rig.upstream.api_base


def test_health_for_one_model_with_no_healthy_deployment_is_503(rig: _Rig) -> None:
    """The master key asks `GET /health?model=dec1-deploy-dead`, whose only deployment is unreachable. It must
    answer 503 with `healthy_count: 0` and `unhealthy_count: 1` in the body, so monitoring can alert on the
    status code alone (_health_endpoints.py:1622)."""
    response: Final = rig.proxy.request("GET", "/health", params={"model": DEAD})
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert response.status_code == 503, response.text
    assert (body["healthy_count"], body["unhealthy_count"]) == (0, 1), body


def test_health_for_an_unknown_model_id_is_404(rig: _Rig) -> None:
    """`GET /health?model_id=dec1-deploy-no-such-id` from the master key must answer 404 and name
    `Model with ID dec1-deploy-no-such-id not found` (`_health_endpoint_resolve_target_model_name`,
    _health_endpoints.py:1530)."""
    response: Final = rig.proxy.request("GET", "/health", params={"model_id": "dec1-deploy-no-such-id"})
    assert response.status_code == 404, response.text
    assert "Model with ID dec1-deploy-no-such-id not found" in response.text


def test_health_for_a_virtual_key_hides_the_api_base_and_says_so_in_a_header(rig: _Rig) -> None:
    """A virtual key allowed on both models calls `GET /health?model=dec1-deploy-chat`. The answer must carry
    the `Litellm-Health-Field-Notice` header starting `api_base, api_version, aws_bedrock_runtime_endpoint`
    and ending `are admin-only on this endpoint`, and the healthy endpoint must have no `api_base`
    (litellm/proxy/health_check.py:42)."""
    response: Final = rig.proxy.request("GET", "/health", params={"model": CHAT}, key=_virtual_key(rig))
    assert response.status_code == 200, response.text
    notice: Final = response.headers[NOTICE]
    assert notice.startswith("api_base, api_version, aws_bedrock_runtime_endpoint"), notice
    assert notice.endswith("are admin-only on this endpoint"), notice
    endpoints: Final = JSON_OBJECT.validate_json(response.content)["healthy_endpoints"]
    assert isinstance(endpoints, list) and endpoints, response.text
    assert "api_base" not in JSON_OBJECT.validate_python(endpoints[0]), endpoints


def test_health_for_the_master_key_carries_no_field_notice(rig: _Rig) -> None:
    """The master key's `GET /health?model=dec1-deploy-chat` must not carry `Litellm-Health-Field-Notice`:
    clients use the header's presence to detect a stripped body."""
    response: Final = rig.proxy.request("GET", "/health", params={"model": CHAT})
    assert NOTICE.lower() not in {name.lower() for name in response.headers}


def test_health_services_without_a_key_is_401(rig: _Rig) -> None:
    """`GET /health/services?service=slack` with no credential must answer 401."""
    assert rig.proxy.client.get("/health/services", params={"service": "slack"}).status_code == 401


def test_health_services_without_the_service_parameter_is_rejected(rig: _Rig) -> None:
    """The master key calls `GET /health/services` with no `service` query parameter. It must answer 422:
    the parameter is required by the route signature, so no alerting integration is fired by accident.
    UNVERIFIED: FastAPI's 422 for a missing required query parameter, not a 400 from the body of the handler."""
    assert rig.proxy.request("GET", "/health/services").status_code == 422


def test_health_services_rejects_a_service_name_it_does_not_know(rig: _Rig) -> None:
    """The master key calls `GET /health/services?service=dec1-not-a-service`. It must be rejected with a 4xx
    and must not answer 200: the name is outside the `services` literal type.
    UNVERIFIED: 422 from query validation is expected; a 400 `Service must be in list` is also accepted."""
    response: Final = rig.proxy.request("GET", "/health/services", params={"service": "dec1-not-a-service"})
    assert response.status_code in (400, 422), response.text


def test_health_backlog_without_a_key_is_401(rig: _Rig) -> None:
    """`GET /health/backlog` with no credential must answer 401: queue depth is operational data."""
    assert rig.proxy.client.get("/health/backlog").status_code == 401


def test_health_backlog_reports_the_four_admission_counters_and_counts_itself(rig: _Rig) -> None:
    """The master key calls `GET /health/backlog`. It must answer 200 with exactly `in_flight_requests`,
    `admitted_requests`, `queued_requests` and `rejected_requests`, all integers, and `in_flight_requests`
    must be at least 1 because the probe itself is in flight (_health_endpoints.py:2372)."""
    body: Final[dict[str, JsonValue]] = rig.proxy.get("/health/backlog")
    assert frozenset(body) == {"in_flight_requests", "admitted_requests", "queued_requests", "rejected_requests"}
    assert all(isinstance(value, int) for value in body.values()), body
    in_flight: Final = body["in_flight_requests"]
    assert isinstance(in_flight, int) and in_flight >= 1, body


def test_readiness_stays_200_while_a_slow_call_is_in_flight(gateway: Gateway, tmp_path: Path) -> None:
    """A chat call is held at the provider. `GET /health/readiness` must still answer 200 with the two field
    body: a busy pod is not an unready pod, and a rewrite that ties readiness to in-flight work would empty
    the load balancer under load. The call finishes with 200 once released."""
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(gateway, tmp_path / "proxy", {}, config=config) as owned:
            pending: Final = held_chat(owned.gateway, upstream)
            assert readiness(owned.gateway) == (200, dict(READY))
            upstream.release.set()
            answered: Final = pending.result(timeout=120)
            assert getattr(answered, "status_code", None) == 200, answered
