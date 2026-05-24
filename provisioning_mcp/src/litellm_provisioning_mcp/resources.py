"""Builders for the auxiliary Kubernetes objects a provisioned release needs.

These are deliberately ephemeral (``emptyDir`` storage, no persistence): they
exist only to back short-lived end-to-end test deployments. Every object
carries the managed-by / release labels so it can be garbage-collected by
selector when the release is torn down.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import yaml

MANAGED_BY = "litellm-provisioning-mcp"
RELEASE_LABEL = "litellm.ai/release"

# Pinned, widely-mirrored images for the throwaway datastores.
POSTGRES_IMAGE = "postgres:16-alpine"
REDIS_IMAGE = "redis:7-alpine"


@dataclass(frozen=True)
class DatabaseConnection:
    host: str
    port: int
    dbname: str
    secret_name: str
    username_key: str
    password_key: str


@dataclass(frozen=True)
class RedisConnection:
    host: str
    port: int


def common_labels(release: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/managed-by": MANAGED_BY,
        # Match the chart's instance label so a single selector covers both the
        # chart pods and these auxiliary datastores when reporting status.
        "app.kubernetes.io/instance": release,
        RELEASE_LABEL: release,
    }


def selector_label(release: str) -> str:
    return f"{RELEASE_LABEL}={release}"


def _dump(*objects: dict) -> str:
    return yaml.safe_dump_all(objects, default_flow_style=False, sort_keys=False)


def master_key_secret(release: str) -> tuple[str, str]:
    """Return ``(manifest, secret_name)`` for a freshly generated master key."""
    name = f"{release}-master-key"
    master_key = "sk-" + secrets.token_urlsafe(32)
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "labels": common_labels(release)},
        "type": "Opaque",
        "stringData": {"master-key": master_key},
    }
    return _dump(manifest), name


def postgres(release: str) -> tuple[str, DatabaseConnection]:
    name = f"{release}-postgres"
    labels = {**common_labels(release), "app.kubernetes.io/component": "postgres"}
    password = secrets.token_urlsafe(24)

    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "labels": labels},
        "type": "Opaque",
        "stringData": {
            "username": "litellm",
            "password": password,
            "dbname": "litellm",
        },
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    RELEASE_LABEL: release,
                    "app.kubernetes.io/component": "postgres",
                }
            },
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": "postgres",
                            "image": POSTGRES_IMAGE,
                            "ports": [{"containerPort": 5432}],
                            "env": [
                                {
                                    "name": "POSTGRES_USER",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": name,
                                            "key": "username",
                                        }
                                    },
                                },
                                {
                                    "name": "POSTGRES_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": name,
                                            "key": "password",
                                        }
                                    },
                                },
                                {"name": "POSTGRES_DB", "value": "litellm"},
                                {
                                    "name": "PGDATA",
                                    "value": "/var/lib/postgresql/data/pgdata",
                                },
                            ],
                            "readinessProbe": {
                                "exec": {
                                    "command": [
                                        "pg_isready",
                                        "-U",
                                        "litellm",
                                        "-d",
                                        "litellm",
                                    ]
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "500m", "memory": "512Mi"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "data",
                                    "mountPath": "/var/lib/postgresql/data",
                                }
                            ],
                        }
                    ],
                    "volumes": [{"name": "data", "emptyDir": {}}],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "selector": {
                RELEASE_LABEL: release,
                "app.kubernetes.io/component": "postgres",
            },
            "ports": [{"port": 5432, "targetPort": 5432}],
        },
    }
    conn = DatabaseConnection(
        host=name,
        port=5432,
        dbname="litellm",
        secret_name=name,
        username_key="username",
        password_key="password",
    )
    return _dump(secret, deployment, service), conn


def redis(release: str) -> tuple[str, RedisConnection]:
    name = f"{release}-redis"
    labels = {**common_labels(release), "app.kubernetes.io/component": "redis"}

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    RELEASE_LABEL: release,
                    "app.kubernetes.io/component": "redis",
                }
            },
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": "redis",
                            "image": REDIS_IMAGE,
                            "ports": [{"containerPort": 6379}],
                            "readinessProbe": {
                                "exec": {"command": ["redis-cli", "ping"]},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                            "resources": {
                                "requests": {"cpu": "50m", "memory": "64Mi"},
                                "limits": {"cpu": "250m", "memory": "256Mi"},
                            },
                        }
                    ],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "labels": labels},
        "spec": {
            "selector": {
                RELEASE_LABEL: release,
                "app.kubernetes.io/component": "redis",
            },
            "ports": [{"port": 6379, "targetPort": 6379}],
        },
    }
    return _dump(deployment, service), RedisConnection(host=name, port=6379)
