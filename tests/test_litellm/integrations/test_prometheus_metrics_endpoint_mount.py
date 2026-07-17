"""Regression tests for PrometheusLogger._mount_metrics_endpoint route wiring.

https://github.com/BerriAI/litellm/issues/33676: scraping the canonical
``/metrics`` path returned a 307 redirect to ``/metrics/`` because the endpoint
was only registered as a Starlette Mount. The mount matches ``/metrics/`` and
relies on ``redirect_slashes`` to bounce ``/metrics`` there, doubling scrape
traffic and logs. Both paths must now return 200 directly.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.integrations.prometheus import PrometheusLogger


def _client() -> TestClient:
    app = FastAPI()
    PrometheusLogger._mount_metrics_endpoint(app)
    return TestClient(app, follow_redirects=False)


def test_metrics_without_trailing_slash_returns_200_not_redirect():
    response = _client().get("/metrics")

    assert response.status_code == 200, response.text
    assert not response.is_redirect


def test_metrics_with_trailing_slash_returns_200():
    response = _client().get("/metrics/")

    assert response.status_code == 200, response.text


def test_metrics_serves_prometheus_exposition_payload():
    response = _client().get("/metrics")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain")
