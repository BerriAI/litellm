"""
Static checks on docker-compose.yml to verify the UI/gateway split.

Test matrix
-----------

| Scenario                                           | Expected                                  |
|----------------------------------------------------|-------------------------------------------|
| `ui` service exists                                | key present in services dict              |
| `gateway` service exists                           | key present in services dict              |
| Legacy `litellm` service is gone                   | key absent from services dict             |
| `ui` builds from `ui/Dockerfile`                  | build.dockerfile == "ui/Dockerfile"       |
| `gateway` builds from `gateway/Dockerfile`         | build.dockerfile == "gateway/Dockerfile"  |
| `ui` exposes port 3000                             | "3000:3000" in ports list                 |
| `gateway` exposes port 4000                        | "4000:4000" in ports list                 |
| `gateway` has DATABASE_URL env var                 | key present in environment dict           |
| `gateway` has STORE_MODEL_IN_DB env var            | key present in environment dict           |
| `gateway` health check is configured               | healthcheck.test is non-empty             |
| `ui` health check is configured                    | healthcheck.test is non-empty             |
| `gateway` depends on `db`                          | "db" in depends_on                        |
| `db` service exists                                | key present in services dict              |
| `db` health check is configured                    | healthcheck.test is non-empty             |
| `prometheus` service exists                        | key present in services dict              |
| `prometheus.yml` scrapes `gateway`, not `litellm`  | target contains "gateway:4000"            |
| `prometheus.yml` does not reference old service    | target does not contain "litellm:4000"    |
| Named volume `postgres_data` is declared           | key present in top-level volumes dict     |
"""

import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
PROMETHEUS_PATH = REPO_ROOT / "prometheus.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    with COMPOSE_PATH.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def services(compose) -> dict:
    return compose["services"]


# ---------------------------------------------------------------------------
# Service presence
# ---------------------------------------------------------------------------


def test_ui_service_exists(services):
    assert "ui" in services, "'ui' service missing from docker-compose.yml"


def test_gateway_service_exists(services):
    assert "gateway" in services, "'gateway' service missing from docker-compose.yml"


def test_legacy_litellm_service_removed(services):
    assert "litellm" not in services, (
        "Monolithic 'litellm' service should be removed after UI/gateway split"
    )


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------


def test_ui_builds_from_ui_dockerfile(services):
    build = services["ui"].get("build", {})
    assert build.get("dockerfile") == "ui/Dockerfile", (
        f"ui service should build from ui/Dockerfile, got {build.get('dockerfile')}"
    )


def test_gateway_builds_from_gateway_dockerfile(services):
    build = services["gateway"].get("build", {})
    assert build.get("dockerfile") == "gateway/Dockerfile", (
        f"gateway service should build from gateway/Dockerfile, got {build.get('dockerfile')}"
    )


# ---------------------------------------------------------------------------
# Port exposure
# ---------------------------------------------------------------------------


def test_ui_exposes_port_3000(services):
    ports = services["ui"].get("ports", [])
    assert any("3000" in str(p) for p in ports), (
        f"ui service should expose port 3000, got ports={ports}"
    )


def test_gateway_exposes_port_4000(services):
    ports = services["gateway"].get("ports", [])
    assert any("4000" in str(p) for p in ports), (
        f"gateway service should expose port 4000, got ports={ports}"
    )


# ---------------------------------------------------------------------------
# Gateway environment variables
# ---------------------------------------------------------------------------


def test_gateway_has_database_url(services):
    env = services["gateway"].get("environment", {})
    assert "DATABASE_URL" in env, "gateway service must have DATABASE_URL environment variable"


def test_gateway_has_store_model_in_db(services):
    env = services["gateway"].get("environment", {})
    assert "STORE_MODEL_IN_DB" in env, (
        "gateway service must have STORE_MODEL_IN_DB environment variable"
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


def test_gateway_healthcheck_configured(services):
    hc = services["gateway"].get("healthcheck", {})
    assert hc.get("test"), "gateway service must have a healthcheck configured"


def test_ui_healthcheck_configured(services):
    hc = services["ui"].get("healthcheck", {})
    assert hc.get("test"), "ui service must have a healthcheck configured"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def test_gateway_depends_on_db(services):
    depends_on = services["gateway"].get("depends_on", [])
    assert "db" in depends_on, "gateway service must depend on the 'db' service"


# ---------------------------------------------------------------------------
# Supporting services
# ---------------------------------------------------------------------------


def test_db_service_exists(services):
    assert "db" in services, "'db' service missing from docker-compose.yml"


def test_db_healthcheck_configured(services):
    hc = services["db"].get("healthcheck", {})
    assert hc.get("test"), "db service must have a healthcheck configured"


def test_prometheus_service_exists(services):
    assert "prometheus" in services, "'prometheus' service missing from docker-compose.yml"


# ---------------------------------------------------------------------------
# Named volumes
# ---------------------------------------------------------------------------


def test_postgres_data_volume_declared(compose):
    volumes = compose.get("volumes", {})
    assert "postgres_data" in volumes, "postgres_data volume must be declared at top level"


# ---------------------------------------------------------------------------
# prometheus.yml scrape target
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PROMETHEUS_PATH.exists(),
    reason="prometheus.yml not present in this checkout",
)
def test_prometheus_scrapes_gateway_not_litellm():
    with PROMETHEUS_PATH.open() as f:
        prom = yaml.safe_load(f)

    targets = []
    for job in prom.get("scrape_configs", []):
        for sc in job.get("static_configs", []):
            targets.extend(sc.get("targets", []))

    assert any("gateway" in t for t in targets), (
        f"prometheus.yml should target 'gateway:4000', got targets={targets}"
    )
    assert not any(t == "litellm:4000" for t in targets), (
        f"prometheus.yml still references old 'litellm:4000' target; update to 'gateway:4000'"
    )
