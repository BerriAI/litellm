from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest
from fastapi.testclient import TestClient

from litellm import Router
from litellm.proxy import proxy_server
from tests.test_litellm_rust.support.parity import assert_parity
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

MODEL_GROUP: Final = "ocr"
ROUTER_RETRIES: Final = 1


@dataclass(frozen=True, slots=True)
class Row:
    upstream: ResponseSpec
    document: Mapping[str, object] = MappingProxyType(dict(OCR_DOCUMENT))
    deployment: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ProxyOutcome:
    status: int
    error_type: object
    error_code: object
    upstream_requests: int
    cooled_down: int
    timeout_header: bool
    markdown: object


ROWS: Final = {
    "success": Row(ResponseSpec(body=OCR_RESPONSE)),
    "unauthorized": Row(ResponseSpec(body={"message": "rejected"}, status=401)),
    "not-found": Row(ResponseSpec(body={"message": "no such model"}, status=404)),
    "rate-limited": Row(ResponseSpec(body={"message": "slow down"}, status=429, headers={"retry-after": "0"})),
    "provider-failure": Row(ResponseSpec(body={"message": "provider unavailable"}, status=500)),
    "read-timeout": Row(ResponseSpec(body=OCR_RESPONSE, delay=1.5), deployment=MappingProxyType({"timeout": 0.5})),
    "not-json": Row(ResponseSpec(body=None, raw=b"<html>bad gateway</html>")),
    "missing-document-url": Row(
        ResponseSpec(body=OCR_RESPONSE), document=MappingProxyType({"type": "document_url", "document_url": ""})
    ),
}


def observe(server: RecordingServer, row: Row, monkeypatch: pytest.MonkeyPatch, backend: bool) -> ProxyOutcome:
    monkeypatch.setenv("LITELLM_RUST", "1" if backend else "0")
    server.default_response = row.upstream
    before: Final = len(server.requests)
    router: Final = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": OCR_MODEL,
                    "api_key": "test-key",
                    "api_base": server.base_url,
                    **row.deployment,
                },
            }
        ],
        num_retries=ROUTER_RETRIES,
        retry_after=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    with TestClient(proxy_server.app, raise_server_exceptions=False) as client:
        response: Final = client.post("/v1/ocr", json={"model": MODEL_GROUP, "document": dict(row.document)})
    body: Final = response.json()
    error: Final = body.get("error") if isinstance(body, dict) else None
    pages: Final = body.get("pages") if isinstance(body, dict) else None
    if response.status_code == 200:
        assert response.headers.get("x-litellm-rust") == ("true" if backend else None)
    return ProxyOutcome(
        status=response.status_code,
        error_type=None if not isinstance(error, dict) else error.get("type"),
        error_code=None if not isinstance(error, dict) else error.get("code"),
        upstream_requests=len(server.requests) - before,
        cooled_down=len(router.cooldown_cache.get_active_cooldowns(router.get_model_ids(), parent_otel_span=None)),
        timeout_header="x-litellm-timeout" in response.headers,
        markdown=pages[0].get("markdown") if isinstance(pages, list) and pages else None,
    )


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
def test_router_and_proxy_parity(recording_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, row: Row) -> None:
    recording_server.expected_requests = None
    python: Final = observe(recording_server, row, monkeypatch, backend=False)
    rust: Final = observe(recording_server, row, monkeypatch, backend=True)
    assert_parity(python, rust)
