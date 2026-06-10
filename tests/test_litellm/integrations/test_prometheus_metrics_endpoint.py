"""
Regression for #30079.

After 1.88.0 the Prometheus ``/metrics`` endpoint was mounted via
``app.mount("/metrics", make_asgi_app())``. Starlette mounts emit a 307
``Location: /metrics/`` when the request path equals the mount prefix
without a trailing slash, and Prometheus / Grafana Alloy scrapers do not
follow redirects -- so ``GET /metrics`` returns an empty body while the
``up`` metric stays ``1``, hiding the regression.

These tests assert that BOTH ``/metrics`` and ``/metrics/`` return 200
directly with the Prometheus text-format content type, no 307 in between.
"""

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_metrics_endpoint_no_trailing_slash_no_redirect(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

    fresh_app = FastAPI()
    monkeypatch.setattr("litellm.proxy.proxy_server.app", fresh_app)

    from litellm.integrations.prometheus import PrometheusLogger

    PrometheusLogger._mount_metrics_endpoint()

    client = TestClient(fresh_app, follow_redirects=False)

    response = client.get("/metrics")

    assert (
        response.status_code == 200
    ), f"bare /metrics must not 307 (got {response.status_code}); see #30079"
    assert "text/plain" in response.headers.get("content-type", "")


def test_metrics_endpoint_with_trailing_slash_works(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

    fresh_app = FastAPI()
    monkeypatch.setattr("litellm.proxy.proxy_server.app", fresh_app)

    from litellm.integrations.prometheus import PrometheusLogger

    PrometheusLogger._mount_metrics_endpoint()

    client = TestClient(fresh_app, follow_redirects=False)

    response = client.get("/metrics/")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
