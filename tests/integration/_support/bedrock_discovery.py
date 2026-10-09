"""Owned rig and assertions for Bedrock wildcard model discovery through a scripted AWS control plane."""

from __future__ import annotations

import os
import re
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import psutil
import yaml
from integration._support.aws_control_plane import (
    FOUNDATION_MODELS,
    INFERENCE_PROFILES,
    Catalog,
    ControlPlane,
    ControlPlaneRequest,
    verified_scope,
)
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, held, object_value, string_value
from integration._support.process import OwnedProxy, owned_proxy_process
from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from pydantic import JsonValue

HOSTED_REGIONS: Final = ("us-east-1", "us-west-2", "eu-west-1", "us-gov-west-1", "cn-north-1")
UNHOSTED_REGION: Final = "eu-central-1"
CONTROL_MODEL: Final = "integration-discovery-control"
CONVERSE_MODEL: Final = "us.openai.gpt-5.6-sol"
WORKERS: Final = 2
RELOAD_SECONDS: Final = 5
LISTING_TIMEOUT_SECONDS: Final = 10.0
PROFILE_PAGE_CAP: Final = 20
LISTING_PATHS: Final = ("/v1/models", "/models")
INFO_PATHS: Final = ("/model/info", "/v1/model/info")
PROFILES_QUERY: Final[Mapping[str, str]] = {"maxResults": "1000", "typeEquals": "SYSTEM_DEFINED"}
FOUNDATION_QUERY: Final[Mapping[str, str]] = {"byInferenceType": "ON_DEMAND"}
CLOSE: Final[Mapping[str, str]] = {"connection": "close"}
LISTING_FAILURE_LOG: Final = "Error getting valid models"
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
STARTUP_COMPLETE: Final = "Application startup complete."


def control_plane_host(region: str) -> str:
    return f"bedrock.{region}.{get_aws_dns_suffix(region)}"


def credential() -> tuple[str, str]:
    return f"AKIA{uuid.uuid4().hex[:16].upper()}", f"secret-{uuid.uuid4().hex}"


def stem() -> str:
    return f"disc{uuid.uuid4().hex[:10]}"


def catalog_for(marker: str, *, page_size: int | None = None) -> Catalog:
    return Catalog(
        active_profiles=(f"us.{marker}.sonnet-v1:0", f"eu.{marker}.haiku-v1:0"),
        inactive_profiles=(f"us.{marker}.retired-v1:0",),
        on_demand_models=(f"{marker}.nova-micro-v1:0",),
        page_size=page_size,
    )


def discovery_config(parent: Gateway, directory: Path, *, check_provider_endpoint: bool) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        {
            "model_name": CONTROL_MODEL,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": f"{parent.upstream_url}/v1",
            },
        }
    ]
    configuration["litellm_settings"] = {
        **configuration.get("litellm_settings", {}),
        "check_provider_endpoint": check_provider_endpoint,
    }
    path: Final = directory / f"discovery-{'on' if check_provider_endpoint else 'off'}.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


def discovery_environment(plane: ControlPlane, directory: Path) -> Mapping[str, str]:
    empty: Final = directory / "empty-aws-config"
    empty.write_text("")
    return {
        **plane.environment(),
        "AWS_CONFIG_FILE": str(empty),
        "AWS_SHARED_CREDENTIALS_FILE": str(empty),
        "AWS_EC2_METADATA_DISABLED": "true",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(RELOAD_SECONDS),
    }


def aws_environment_names() -> tuple[str, ...]:
    return tuple(name for name in os.environ if name.startswith("AWS_"))


@contextmanager
def discovery_proxy(
    parent: Gateway, directory: Path, plane: ControlPlane, *, check_provider_endpoint: bool, workers: int
) -> Iterator[OwnedProxy]:
    with owned_proxy_process(
        parent,
        directory,
        discovery_environment(plane, directory),
        config=discovery_config(parent, directory, check_provider_endpoint=check_provider_endpoint),
        remove_environment=aws_environment_names(),
        workers=workers,
    ) as owned:
        wait_for_every_worker(owned.log, workers)
        yield owned


def deployment_ids(gateway: Gateway) -> frozenset[str]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    return frozenset(string_value(object_value(object_value(entry)["model_info"])["id"]) for entry in entries)


def remove_deployment(gateway: Gateway, identity: str) -> None:
    gateway.post("/model/delete", {"id": identity})
    eventually(lambda: deployment_ids(gateway), lambda ids: identity not in ids, seconds=RELOAD_SECONDS * 4)


def deployment(
    scenario: Scenario,
    *,
    model_name: str = "bedrock/*",
    model: str = "bedrock/*",
    model_info: Mapping[str, JsonValue] | None = None,
    litellm_params: Mapping[str, JsonValue] | None = None,
) -> str:
    """A wildcard deployment posted straight to /model/new, since Scenario.model fixes a non-wildcard name."""
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {"model": model, **(litellm_params or {})},
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(remove_deployment, scenario.gateway, identity)
    settled_on_every_worker(scenario.gateway, identity)
    return identity


def settled_on_every_worker(gateway: Gateway, identity: str) -> None:
    """Each worker picks a new deployment up on its own reload tick, so a listing straight after /model/new can land on
    a worker without it; the by-id lookup never lists, so holding on it leaves every worker's discovery cache cold."""

    def present() -> bool:
        lookup: Final = gateway.request("GET", "/model/info", params={"litellm_model_id": identity}, headers=CLOSE)
        return lookup.status_code == 200

    held(present, lambda found: found, holding=RELOAD_SECONDS * 2, seconds=RELOAD_SECONDS * 6)


def sigv4_deployment(
    scenario: Scenario,
    key: str,
    secret: str,
    region: str,
    *,
    model_name: str = "bedrock/*",
    model: str = "bedrock/*",
    model_info: Mapping[str, JsonValue] | None = None,
    litellm_params: Mapping[str, JsonValue] | None = None,
) -> str:
    return deployment(
        scenario,
        model_name=model_name,
        model=model,
        model_info=model_info,
        litellm_params={
            "aws_access_key_id": key,
            "aws_secret_access_key": secret,
            "aws_region_name": region,
            **(litellm_params or {}),
        },
    )


def listed_ids(
    gateway: Gateway,
    path: str = "/v1/models",
    *,
    key: str | None = None,
    params: Mapping[str, str] | None = None,
) -> frozenset[str]:
    response: Final = gateway.request("GET", path, key=key, params=params, headers=CLOSE)
    assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text}"
    data: Final = JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(data, list), response.text
    return frozenset(string_value(object_value(entry)["id"]) for entry in data)


def from_stem(marker: str, ids: frozenset[str]) -> frozenset[str]:
    return frozenset(name for name in ids if marker in name)


def discovered(
    gateway: Gateway,
    catalog: Catalog,
    marker: str,
    path: str = "/v1/models",
    *,
    key: str | None = None,
    params: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Poll until the catalog's invocable ids are listed; returns every listed id carrying the stem."""
    listed: Final = eventually(
        lambda: listed_ids(gateway, path, key=key, params=params),
        lambda ids: catalog.invocable_ids() <= ids,
        seconds=RELOAD_SECONDS * 4,
    )
    return from_stem(marker, listed)


def mine(plane: ControlPlane, credential_id: str) -> tuple[ControlPlaneRequest, ...]:
    return tuple(request for request in plane.drain() if request.credential == credential_id)


def listings(requests: tuple[ControlPlaneRequest, ...]) -> Mapping[str, int]:
    return Counter(request.path for request in requests)


def assert_listing_shape(requests: tuple[ControlPlaneRequest, ...], *, pages: int = 1) -> int:
    """Every listing is one foundation-models GET plus `pages` inference-profiles GETs; returns the listing count."""
    assert requests, "no listing reached the control plane"
    counts: Final = listings(requests)
    assert set(counts) == {FOUNDATION_MODELS, INFERENCE_PROFILES}, counts
    assert 1 <= counts[FOUNDATION_MODELS] <= WORKERS, counts
    assert counts[INFERENCE_PROFILES] == counts[FOUNDATION_MODELS] * pages, counts
    for request in requests:
        assert request.method == "GET" and request.body == b"", request
        if request.path == FOUNDATION_MODELS:
            assert request.query == FOUNDATION_QUERY, request
        else:
            assert {name: value for name, value in request.query.items() if name != "nextToken"} == PROFILES_QUERY
    return counts[FOUNDATION_MODELS]


def assert_sigv4(requests: tuple[ControlPlaneRequest, ...], *, key: str, secret: str, region: str) -> None:
    host: Final = control_plane_host(region)
    for request in requests:
        assert request.host == host, request
        assert request.headers["host"] == host, request
        scope: Final = verified_scope(request, secret)
        assert scope == f"{request.headers['x-amz-date'][:8]}/{region}/bedrock/aws4_request", request
        assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={key}/"), request
        assert "x-amz-security-token" not in request.headers, request


def worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(pid) for pid in STARTED_WORKER.findall(log.read_text()))


def live_worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(pid for pid in worker_pids(log) if psutil.pid_exists(pid))


def wait_for_every_worker(log: Path, workers: int) -> None:
    """Readiness answers from the first worker up; uvicorn replaces a child that dies at boot, so the rest can lag."""

    def every_worker_serving() -> bool:
        return len(live_worker_pids(log)) >= workers and log.read_text().count(STARTUP_COMPLETE) >= workers

    eventually(every_worker_serving, bool, seconds=150)


def wait_for_replacement_worker(log: Path, original: tuple[int, ...]) -> None:
    def replacement_is_serving(pids: tuple[int, ...]) -> bool:
        return len(pids) > len(original) and log.read_text().count(STARTUP_COMPLETE) > len(original)

    eventually(lambda: worker_pids(log), replacement_is_serving, seconds=150)


def open_connections_to(pid: int, url: str) -> int:
    port: Final = urlsplit(url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )
