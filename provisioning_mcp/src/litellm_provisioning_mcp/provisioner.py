"""Orchestrates ephemeral LiteLLM deployments via the ``helm/litellm`` chart.

A single ``provision`` call: mints a master-key Secret, optionally stands up a
throwaway Postgres and/or Redis, then ``helm upgrade --install``s the chart
pinned to the requested revision's images. Teardown removes both the helm
release and the auxiliary objects.

This module is free of any ``mcp`` dependency so it can be exercised directly
in unit tests with fake command runners.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import yaml

from . import resources
from .config import Settings
from .helm import HelmRunner
from .kubectl import KubectlRunner
from .naming import derive_image_repos, sanitize_label, sanitize_release_name
from .resources import DatabaseConnection, RedisConnection

_DATASTORE_WAIT_SECONDS = 120


class ProvisionError(RuntimeError):
    """Raised for caller-correctable problems (bad input, missing dependency)."""


@dataclass(frozen=True)
class ProvisionRequest:
    repo_url: str
    revision: str
    release_name: str | None = None
    enable_redis: bool = False
    enable_postgres: bool = True
    enable_ui: bool = False
    service_account: str | None = None
    image_registry: str | None = None
    external_database: dict | None = None
    extra_values: dict = field(default_factory=dict)


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _external_database(spec: dict) -> DatabaseConnection:
    missing = [k for k in ("host", "dbname", "secret_name") if not spec.get(k)]
    if missing:
        raise ProvisionError(
            f"external_database is missing required keys: {', '.join(missing)}"
        )
    return DatabaseConnection(
        host=str(spec["host"]),
        port=int(spec.get("port", 5432)),
        dbname=str(spec["dbname"]),
        secret_name=str(spec["secret_name"]),
        username_key=str(spec.get("username_key", "username")),
        password_key=str(spec.get("password_key", "password")),
    )


def _summarize_pods(items: list[dict]) -> list[dict]:
    summary = []
    for pod in items:
        statuses = pod.get("status", {}).get("containerStatuses", []) or []
        summary.append(
            {
                "name": pod.get("metadata", {}).get("name", ""),
                "phase": pod.get("status", {}).get("phase", "Unknown"),
                "ready": bool(statuses) and all(c.get("ready") for c in statuses),
                "restarts": sum(c.get("restartCount", 0) for c in statuses),
            }
        )
    return summary


class Provisioner:
    def __init__(
        self, settings: Settings, *, helm: HelmRunner, kubectl: KubectlRunner
    ) -> None:
        self._settings = settings
        self._helm = helm
        self._kubectl = kubectl

    def _build_values(
        self,
        *,
        release: str,
        image_repos: dict[str, str],
        revision: str,
        master_key_secret: str,
        database: DatabaseConnection,
        redis_conn: RedisConnection | None,
        request: ProvisionRequest,
    ) -> dict:
        values: dict = {
            "fullnameOverride": release,
            "masterKey": {"secretName": master_key_secret, "secretKey": "master-key"},
            "database": {
                "writer": {
                    "host": database.host,
                    "port": database.port,
                    "dbname": database.dbname,
                    "passwordSecret": {
                        "name": database.secret_name,
                        "usernameKey": database.username_key,
                        "passwordKey": database.password_key,
                    },
                }
            },
            "gateway": {
                "image": {"repository": image_repos["gateway"], "tag": revision}
            },
            "backend": {
                "image": {"repository": image_repos["backend"], "tag": revision}
            },
            "ui": {
                "enabled": request.enable_ui,
                "image": {"repository": image_repos["ui"], "tag": revision},
            },
            "migrationJob": {
                "image": {"repository": image_repos["migrations"], "tag": revision}
            },
        }
        if redis_conn is not None:
            values["redis"] = {"host": redis_conn.host, "port": redis_conn.port}
        if request.service_account:
            values["serviceAccount"] = {
                "create": False,
                "name": request.service_account,
            }
        if request.extra_values:
            values = _deep_merge(values, request.extra_values)
        return values

    async def provision(self, request: ProvisionRequest) -> dict:
        if not request.repo_url.strip():
            raise ProvisionError("repo_url is required")
        if not request.revision.strip():
            raise ProvisionError("revision is required")

        release = sanitize_release_name(
            request.release_name
            or f"{self._settings.release_prefix}-{request.revision}"
        )
        image_repos = derive_image_repos(
            repo_url=request.repo_url,
            registry_override=request.image_registry,
            default_registry=self._settings.default_image_registry,
        )

        mk_manifest, mk_secret = resources.master_key_secret(release)
        await self._kubectl.apply(mk_manifest)

        if request.enable_postgres:
            pg_manifest, database = resources.postgres(release)
            await self._kubectl.apply(pg_manifest)
            await self._kubectl.wait_available(
                deployment=f"{release}-postgres", timeout=_DATASTORE_WAIT_SECONDS
            )
        elif request.external_database:
            database = _external_database(request.external_database)
        else:
            raise ProvisionError(
                "a database is required: set enable_postgres=true or provide external_database"
            )

        redis_conn: RedisConnection | None = None
        if request.enable_redis:
            redis_manifest, redis_conn = resources.redis(release)
            await self._kubectl.apply(redis_manifest)
            await self._kubectl.wait_available(
                deployment=f"{release}-redis", timeout=_DATASTORE_WAIT_SECONDS
            )

        values = self._build_values(
            release=release,
            image_repos=image_repos,
            revision=sanitize_label(request.revision),
            master_key_secret=mk_secret,
            database=database,
            redis_conn=redis_conn,
            request=request,
        )
        await self._helm.upgrade_install(
            release=release,
            chart_path=self._settings.chart_path,
            values_yaml=yaml.safe_dump(values, sort_keys=False),
        )

        pods = await self._kubectl.get_pods(
            selector=f"app.kubernetes.io/instance={release}"
        )
        ns = self._settings.namespace
        endpoints = {"gateway": f"http://{release}-gateway.{ns}.svc.cluster.local:4000"}
        if request.enable_ui:
            endpoints["ui"] = f"http://{release}-ui.{ns}.svc.cluster.local:3000"
        endpoints["backend"] = f"http://{release}-backend.{ns}.svc.cluster.local:4001"

        return {
            "success": True,
            "release": release,
            "namespace": ns,
            "revision": request.revision,
            "images": image_repos,
            "postgres_enabled": request.enable_postgres,
            "redis_enabled": request.enable_redis,
            "ui_enabled": request.enable_ui,
            "endpoints": endpoints,
            "pods": _summarize_pods(pods),
        }

    async def delete(self, release_name: str) -> dict:
        release = sanitize_release_name(release_name)
        await self._helm.uninstall(release=release)
        await self._kubectl.delete_by_label(
            selector=resources.selector_label(release),
            kinds=["deployment", "service", "secret"],
        )
        return {
            "success": True,
            "release": release,
            "namespace": self._settings.namespace,
        }

    async def status(self, release_name: str) -> dict:
        release = sanitize_release_name(release_name)
        helm_status = await self._helm.status(release=release)
        pods = await self._kubectl.get_pods(
            selector=f"app.kubernetes.io/instance={release}"
        )
        info = helm_status.get("info", {})
        return {
            "release": release,
            "namespace": self._settings.namespace,
            "status": info.get("status"),
            "last_deployed": info.get("last_deployed"),
            "pods": _summarize_pods(pods),
        }

    async def list_deployments(self) -> dict:
        releases = await self._helm.list_releases()
        return {
            "namespace": self._settings.namespace,
            "deployments": [
                {
                    "release": r.get("name"),
                    "status": r.get("status"),
                    "chart": r.get("chart"),
                    "updated": r.get("updated"),
                }
                for r in releases
            ],
        }
