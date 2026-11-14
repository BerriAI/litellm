import base64
import json
from collections.abc import Iterator, Mapping
from typing import Final

import httpx
import pytest
import responses
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request, Response
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


pytestmark: Final = pytest.mark.usefixtures("drained_logging_worker")


@pytest.fixture
def isolate_passthrough_endpoint_router_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    monkeypatch.setattr(
        passthrough_endpoint_router,
        "deployment_key_to_vertex_credentials",
        passthrough_endpoint_router.deployment_key_to_vertex_credentials.copy(),
    )


@pytest.mark.parametrize("reusable_credentials", [True, False])
@pytest.mark.usefixtures("isolate_passthrough_endpoint_router_state")
def test_initialize_deployment_for_pass_through_success(reusable_credentials):
    """
    Test successful initialization of a Vertex AI pass-through deployment
    """
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.types.utils import CredentialItem

    vertex_project = "test-project"
    vertex_location = "us-central1"
    vertex_credentials = json.dumps({"type": "service_account", "project_id": "test"})

    if not reusable_credentials:
        litellm_params = LiteLLM_Params(
            model="vertex_ai/test-model",
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            use_in_pass_through=True,
        )
    else:
        # add credentials to the credential accessor
        CredentialAccessor.upsert_credentials(
            [
                CredentialItem(
                    credential_name="vertex_credentials",
                    credential_values={
                        "vertex_project": vertex_project,
                        "vertex_location": vertex_location,
                        "vertex_credentials": vertex_credentials,
                    },
                    credential_info={},
                )
            ]
        )
        litellm_params = LiteLLM_Params(
            model="vertex_ai/test-model",
            litellm_credential_name="vertex_credentials",
            use_in_pass_through=True,
        )
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=litellm_params,
    )

    # Test the initialization
    router.add_deployment(deployment)

    # Verify the credentials were properly set
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    vertex_creds = passthrough_endpoint_router.get_vertex_credentials(project_id="test-project", location="us-central1")
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps({"type": "service_account", "project_id": "test"})


def test_initialize_deployment_for_pass_through_missing_params():
    """
    Test initialization fails when required Vertex AI parameters are missing
    """
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=LiteLLM_Params(
            model="vertex_ai/test-model",
            # Missing required parameters
            use_in_pass_through=True,
        ),
    )

    # Test that initialization raises ValueError
    with pytest.raises(
        ValueError,
        match="vertex_project, and vertex_location must be set in litellm_params for pass-through endpoints",
    ):
        router.add_deployment(deployment)


def test_initialize_deployment_when_pass_through_disabled():
    """
    Test that initialization simply exits when use_in_pass_through is False
    """
    router = Router(model_list=[])
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=LiteLLM_Params(
            model="vertex_ai/test-model",
        ),
    )

    # This should exit without error, even with missing vertex parameters
    router.add_deployment(deployment)

    # If we reach this point, the test passes as the method exited without raising any errors
    assert True


@pytest.mark.usefixtures("isolate_passthrough_endpoint_router_state")
def test_add_vertex_pass_through_deployment():
    """
    Test adding a Vertex AI deployment with pass-through configuration
    """
    router = Router(model_list=[])

    # Create a deployment with Vertex AI pass-through settings
    deployment = Deployment(
        model_name="vertex-test",
        litellm_params=LiteLLM_Params(
            model="vertex_ai/test-model",
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=json.dumps({"type": "service_account", "project_id": "test"}),
            use_in_pass_through=True,
        ),
    )

    # Add deployment to router
    router.add_deployment(deployment)

    # Get the vertex credentials from the router
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    # current state of pass-through vertex router

    vertex_creds = passthrough_endpoint_router.get_vertex_credentials(project_id="test-project", location="us-central1")

    # Verify the credentials were properly set
    assert vertex_creds.vertex_project == "test-project"
    assert vertex_creds.vertex_location == "us-central1"
    assert vertex_creds.vertex_credentials == json.dumps({"type": "service_account", "project_id": "test"})


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.mark.asyncio
async def test_gemini_pass_through_endpoint_forwards_key_and_body(
    respx_mock: "respx.MockRouter", httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from fastapi import Request, Response

    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        gemini_proxy_route,
    )

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    endpoint: Final = "v1beta/models/gemini-2.5-flash:countTokens"
    upstream: Final = respx_mock.post(
        url__startswith=f"https://generativelanguage.googleapis.com/{endpoint}"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"totalTokens": 9},
        )
    )
    body: Final = json.dumps(
        {"contents": [{"parts": [{"text": "The quick brown fox jumps over the lazy dog."}]}]}
    ).encode()
    scope: Final = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/gemini/{endpoint}",
        "raw_path": f"/gemini/{endpoint}".encode(),
        "root_path": "",
        "query_string": b"key=sk-9876",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 51234),
        "server": ("proxy.local", 4000),
        "state": {},
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    response: Final = await gemini_proxy_route(
        endpoint=f"{endpoint}?key=sk-9876",
        request=Request(scope, receive),
        fastapi_response=Response(),
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"totalTokens": 9}
    assert upstream.call_count == 1
    outbound: Final = upstream.calls[0].request
    assert outbound.url.path == f"/{endpoint}"
    assert outbound.url.params["key"] == "gemini-test-key"
    assert json.loads(outbound.content) == json.loads(body)


GEMINI_UPLOAD_ENDPOINT: Final = "upload/v1beta/files"
JSONL_BATCH_REQUESTS: Final = (
    b'{"key":"r1","request":{"contents":[{"parts":[{"text":"Say hi"}]}]}}\n'
    b'{"key":"r2","request":{"contents":[{"parts":[{"text":"Say bye"}]}]}}\n'
)


async def _call_gemini_upload_route(
    headers: tuple[tuple[bytes, bytes], ...], body: bytes, query_string: bytes = b""
) -> Response:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        gemini_proxy_route,
    )

    scope: Final = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/gemini/{GEMINI_UPLOAD_ENDPOINT}",
        "raw_path": f"/gemini/{GEMINI_UPLOAD_ENDPOINT}".encode(),
        "root_path": "",
        "query_string": query_string,
        "headers": [(b"x-goog-api-key", b"sk-9876"), *headers],
        "client": ("127.0.0.1", 51234),
        "server": ("proxy.local", 4000),
        "state": {},
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    return await gemini_proxy_route(
        endpoint=GEMINI_UPLOAD_ENDPOINT,
        request=Request(scope, receive),
        fastapi_response=Response(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/jsonl", "application/json"])
async def test_gemini_upload_forwards_jsonl_bytes_and_upload_headers(
    respx_mock: "respx.MockRouter", httpx_transport: None, monkeypatch: pytest.MonkeyPatch, content_type: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    uploaded_file: Final = {"file": {"name": "files/abc", "mimeType": content_type, "sizeBytes": "137"}}
    upstream: Final = respx_mock.post(
        url__startswith=f"https://generativelanguage.googleapis.com/{GEMINI_UPLOAD_ENDPOINT}"
    ).mock(return_value=httpx.Response(200, json=uploaded_file))

    response: Final = await _call_gemini_upload_route(
        headers=((b"content-type", content_type.encode()), (b"x-goog-upload-protocol", b"raw")),
        body=JSONL_BATCH_REQUESTS,
    )

    assert response.status_code == 200
    assert json.loads(response.body) == uploaded_file
    outbound: Final = upstream.calls[0].request
    assert outbound.content == JSONL_BATCH_REQUESTS
    assert outbound.headers["content-type"] == content_type
    assert outbound.headers["x-goog-upload-protocol"] == "raw"
    assert "x-goog-api-key" not in outbound.headers
    assert outbound.url.params["key"] == "gemini-test-key"


GEMINI_UPLOAD_SESSION_PATH: Final = f"/{GEMINI_UPLOAD_ENDPOINT}?upload_id=session-1&upload_protocol=resumable"
GOOGLE_UPLOAD_SESSION_PATH: Final = (
    f"/{GEMINI_UPLOAD_ENDPOINT}?key=gemini-test-key&upload_id=session-1&upload_protocol=resumable"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gemini_api_base", "proxy_base_url", "upstream_upload_url", "expected_upload_url"),
    [
        (
            None,
            None,
            f"https://generativelanguage.googleapis.com{GOOGLE_UPLOAD_SESSION_PATH}",
            f"http://proxy.local:4000/gemini{GEMINI_UPLOAD_SESSION_PATH}",
        ),
        (
            None,
            "https://llm.example.com",
            f"https://generativelanguage.googleapis.com{GOOGLE_UPLOAD_SESSION_PATH}",
            f"https://llm.example.com/gemini{GEMINI_UPLOAD_SESSION_PATH}",
        ),
        (
            "https://gemini-gateway.internal/google",
            None,
            f"https://gemini-gateway.internal/google{GOOGLE_UPLOAD_SESSION_PATH}",
            f"http://proxy.local:4000/gemini{GEMINI_UPLOAD_SESSION_PATH}",
        ),
        (
            "https://gemini-gateway.internal/google",
            None,
            f"https://generativelanguage.googleapis.com{GOOGLE_UPLOAD_SESSION_PATH}",
            f"http://proxy.local:4000/gemini{GEMINI_UPLOAD_SESSION_PATH}",
        ),
    ],
    ids=["default-base", "proxy-base-url", "base-with-path", "url-outside-base"],
)
async def test_gemini_resumable_upload_start_returns_an_upload_url_on_the_proxy(
    respx_mock: "respx.MockRouter",
    httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
    gemini_api_base: str | None,
    proxy_base_url: str | None,
    upstream_upload_url: str,
    expected_upload_url: str,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    for name, value in (("GEMINI_API_BASE", gemini_api_base), ("PROXY_BASE_URL", proxy_base_url)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    upstream: Final = respx_mock.post(
        url__startswith=f"{gemini_api_base or 'https://generativelanguage.googleapis.com'}/{GEMINI_UPLOAD_ENDPOINT}"
    ).mock(
        return_value=httpx.Response(
            200,
            headers={
                "content-type": "text/plain; charset=utf-8",
                "x-goog-upload-status": "active",
                "x-goog-upload-url": upstream_upload_url,
                "x-goog-upload-control-url": upstream_upload_url,
            },
        )
    )
    metadata: Final = b'{"file": {"display_name": "batch-requests"}}'

    response: Final = await _call_gemini_upload_route(
        headers=(
            (b"content-type", b"application/json"),
            (b"x-goog-upload-protocol", b"resumable"),
            (b"x-goog-upload-command", b"start"),
            (b"x-goog-upload-header-content-length", b"137"),
            (b"x-goog-upload-header-content-type", b"application/jsonl"),
        ),
        body=metadata,
    )

    assert response.status_code == 200
    assert response.headers["x-goog-upload-url"] == expected_upload_url
    assert response.headers["x-goog-upload-control-url"] == expected_upload_url
    assert response.headers["x-goog-upload-status"] == "active"
    assert not any("gemini-test-key" in value for value in response.headers.values())
    outbound: Final = upstream.calls[0].request
    assert outbound.content == metadata
    assert outbound.headers["x-goog-upload-command"] == "start"
    assert outbound.headers["x-goog-upload-header-content-length"] == "137"
    assert outbound.headers["x-goog-upload-header-content-type"] == "application/jsonl"


@pytest.mark.asyncio
async def test_gemini_resumable_upload_chunk_reaches_the_upload_session(
    respx_mock: "respx.MockRouter", httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    upstream: Final = respx_mock.post(
        url__startswith=f"https://generativelanguage.googleapis.com/{GEMINI_UPLOAD_ENDPOINT}"
    ).mock(
        return_value=httpx.Response(
            200, headers={"x-goog-upload-status": "final"}, json={"file": {"name": "files/abc", "state": "ACTIVE"}}
        )
    )

    response: Final = await _call_gemini_upload_route(
        headers=((b"x-goog-upload-command", b"upload, finalize"), (b"x-goog-upload-offset", b"0")),
        body=JSONL_BATCH_REQUESTS,
        query_string=b"upload_id=session-1&upload_protocol=resumable",
    )

    assert response.status_code == 200
    assert response.headers["x-goog-upload-status"] == "final"
    outbound: Final = upstream.calls[0].request
    assert outbound.content == JSONL_BATCH_REQUESTS
    assert outbound.url.params["upload_id"] == "session-1"
    assert outbound.url.params["key"] == "gemini-test-key"
    assert outbound.headers["x-goog-upload-command"] == "upload, finalize"
    assert outbound.headers["x-goog-upload-offset"] == "0"

@pytest.mark.asyncio
async def test_router_passthrough_keeps_mapped_baggage_tags_outside_upstream_body():
    from fastapi import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.http_parsing_utils import read_request_body
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import get_passthrough_router_request_metadata
    from litellm.proxy.utils import ProxyLogging

    async def receive():
        return {"type": "http.request", "body": b'{"model":"qa-model"}'}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/openai/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"baggage", b"entrypoint=a")],
        },
        receive,
    )
    admitted = await read_request_body(request)
    await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
        request,
        admitted,
        {"request_tag_mappings": [{"key": "entrypoint", "tag_prefix": "usecase:"}]},
        None,
        UserApiKeyCache(),
    )
    metadata = get_passthrough_router_request_metadata(UserAPIKeyAuth(api_key="qa-key"), request)
    assert metadata["tags"] == ["usecase:a"]
    assert await read_request_body(request) == {"model": "qa-model"}


_VERTEX_DEPLOYMENT_PATH: Final = (
    "v1/projects/deployment-project/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict"
)
_VERTEX_CALLER_PATH: Final = (
    "v1/projects/caller-project/locations/global/publishers/anthropic/models/claude-haiku-4-5:rawPredict"
)
_VERTEX_DEPLOYMENT_PROJECT: Final = "deployment-project"
_VERTEX_TOKEN_URI: Final = "https://oauth2.googleapis.com/token"
_VERTEX_MINTED_TOKEN: Final = "minted-vertex-token"
_VERTEX_CALLER_GOOGLE_TOKEN: Final = "caller-google-token"
_VERTEX_REJECTED_BETA: Final = "files-api-2025-04-14"
_VERTEX_SENT_BETAS: Final = f"claude-code-20250219, {_VERTEX_REJECTED_BETA} ,effort-2025-11-24"
_VERTEX_CLAUDE_BODY: Final = {
    "anthropic_version": "vertex-2023-10-16",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}
_VERTEX_CLAUDE_REPLY: Final = {"id": "msg_vertex", "type": "message", "content": []}


@pytest.mark.parametrize(
    ("header_name", "sent", "dropped", "expected"),
    [
        pytest.param(
            "anthropic-beta",
            "claude-code-20250219,files-api-2025-04-14,effort-2025-11-24",
            {"files-api-2025-04-14"},
            "claude-code-20250219,effort-2025-11-24",
            id="drops-one-keeps-order",
        ),
        pytest.param(
            "anthropic-beta",
            " claude-code-20250219 , files-api-2025-04-14 ,effort-2025-11-24 ",
            {"files-api-2025-04-14"},
            "claude-code-20250219,effort-2025-11-24",
            id="whitespace-around-values",
        ),
        pytest.param(
            "Anthropic-Beta",
            "effort-2025-11-24,files-api-2025-04-14",
            {"files-api-2025-04-14"},
            "effort-2025-11-24",
            id="header-name-any-case",
        ),
        pytest.param(
            "anthropic-beta",
            "files-api-2025-04-14, skills-2025-10-02",
            {"files-api-2025-04-14", "skills-2025-10-02"},
            None,
            id="all-dropped-removes-header",
        ),
        pytest.param(
            "anthropic-beta",
            " claude-code-20250219 ,files-api-2025-04-14",
            set(),
            " claude-code-20250219 ,files-api-2025-04-14",
            id="no-setting-byte-identical",
        ),
        pytest.param(
            "anthropic-beta",
            " claude-code-20250219 ,files-api-2025-04-14-preview",
            {"files-api-2025-04-14"},
            " claude-code-20250219 ,files-api-2025-04-14-preview",
            id="exact-match-only",
        ),
    ],
)
def test_vertex_upstream_headers_drop_only_the_configured_anthropic_betas(
    header_name: str, sent: str, dropped: set[str], expected: str | None
) -> None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _without_dropped_anthropic_betas,
    )

    other_headers: Final = {"content-type": "application/json", "Authorization": "Bearer vertex-access-token"}

    upstream: Final = _without_dropped_anthropic_betas(frozenset(dropped), {header_name: sent, **other_headers})

    expected_beta: Final = {} if expected is None else {header_name: expected}
    assert dict(upstream) == {**expected_beta, **other_headers}


def _vertex_service_account_json() -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _VERTEX_DEPLOYMENT_PROJECT,
            "private_key_id": "test",
            "private_key": private_key,
            "client_email": f"test@{_VERTEX_DEPLOYMENT_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "token_uri": _VERTEX_TOKEN_URI,
        }
    )


@pytest.fixture
def vertex_proxy(
    monkeypatch: pytest.MonkeyPatch, httpx_transport: None, isolate_passthrough_endpoint_router_state: None
) -> Iterator[TestClient]:
    from litellm.proxy import proxy_server

    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5",
                "litellm_params": {
                    "model": "vertex_ai/claude-sonnet-4-5",
                    "vertex_project": _VERTEX_DEPLOYMENT_PROJECT,
                    "vertex_location": "global",
                    "vertex_credentials": _vertex_service_account_json(),
                    "use_in_pass_through": True,
                },
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as google_oauth:
        google_oauth.post(
            _VERTEX_TOKEN_URI,
            json={"access_token": _VERTEX_MINTED_TOKEN, "expires_in": 3600, "token_type": "Bearer"},
        )
        yield TestClient(proxy_server.app)


@pytest.mark.parametrize("beta_header", ["anthropic-beta", "x-pass-anthropic-beta"])
@pytest.mark.parametrize("alias", ["/vertex_ai", "/vertex-ai"])
@pytest.mark.parametrize(
    ("model_path", "caller_headers", "upstream_authorization"),
    [
        pytest.param(
            _VERTEX_DEPLOYMENT_PATH,
            {},
            f"Bearer {_VERTEX_MINTED_TOKEN}",
            id="use_in_pass_through-deployment",
        ),
        pytest.param(
            _VERTEX_CALLER_PATH,
            {"Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}"},
            f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}",
            id="caller-credentials",
        ),
    ],
)
def test_vertex_pass_through_drops_the_configured_anthropic_beta_before_calling_vertex(
    vertex_proxy: TestClient,
    respx_mock: "respx.MockRouter",
    monkeypatch: pytest.MonkeyPatch,
    beta_header: str,
    alias: str,
    model_path: str,
    caller_headers: dict[str, str],
    upstream_authorization: str,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(
        proxy_server.general_settings, "vertex_ai_pass_through_drop_anthropic_betas", [_VERTEX_REJECTED_BETA]
    )
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{model_path}").mock(
        return_value=httpx.Response(200, json=_VERTEX_CLAUDE_REPLY)
    )

    response: Final = vertex_proxy.post(
        f"{alias}/{model_path}",
        json=_VERTEX_CLAUDE_BODY,
        headers={"x-litellm-api-key": "sk-caller", beta_header: _VERTEX_SENT_BETAS, **caller_headers},
    )

    assert (response.status_code, response.json()) == (200, _VERTEX_CLAUDE_REPLY)
    outbound: Final = upstream.calls.last.request
    assert outbound.headers["anthropic-beta"] == "claude-code-20250219,effort-2025-11-24"
    assert outbound.headers["authorization"] == upstream_authorization
    assert json.loads(outbound.content) == _VERTEX_CLAUDE_BODY



@pytest.mark.parametrize(
    "configured",
    [[f" {_VERTEX_REJECTED_BETA} "], [f"{_VERTEX_REJECTED_BETA},afk-mode-2026-01-31"]],
    ids=["spaces-around-the-entry", "comma-separated-entry"],
)
def test_vertex_pass_through_matches_a_configured_beta_written_the_way_the_header_is(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch, configured: list[str]
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "vertex_ai_pass_through_drop_anthropic_betas", configured)
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_CALLER_PATH}").mock(
        return_value=httpx.Response(200, json=_VERTEX_CLAUDE_REPLY)
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_CALLER_PATH}",
        json=_VERTEX_CLAUDE_BODY,
        headers={
            "x-litellm-api-key": "sk-caller",
            "Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}",
            "anthropic-beta": _VERTEX_SENT_BETAS,
        },
    )

    assert (response.status_code, response.json()) == (200, _VERTEX_CLAUDE_REPLY)
    assert upstream.calls.last.request.headers["anthropic-beta"] == "claude-code-20250219,effort-2025-11-24"

def test_vertex_pass_through_refuses_a_malformed_drop_setting_without_calling_vertex(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(
        proxy_server.general_settings, "vertex_ai_pass_through_drop_anthropic_betas", _VERTEX_REJECTED_BETA
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_CALLER_PATH}",
        json=_VERTEX_CLAUDE_BODY,
        headers={
            "x-litellm-api-key": "sk-caller",
            "Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}",
            "anthropic-beta": _VERTEX_SENT_BETAS,
        },
    )

    assert response.status_code == 500
    assert "vertex_ai_pass_through_drop_anthropic_betas must be a list of strings" in response.text
    assert respx_mock.calls.call_count == 0


def test_vertex_pass_through_forwards_a_request_without_betas_despite_a_malformed_drop_setting(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(
        proxy_server.general_settings, "vertex_ai_pass_through_drop_anthropic_betas", _VERTEX_REJECTED_BETA
    )
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_CALLER_PATH}").mock(
        return_value=httpx.Response(200, json=_VERTEX_CLAUDE_REPLY)
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_CALLER_PATH}",
        json=_VERTEX_CLAUDE_BODY,
        headers={"x-litellm-api-key": "sk-caller", "Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}"},
    )

    assert (response.status_code, response.json()) == (200, _VERTEX_CLAUDE_REPLY)
    assert "anthropic-beta" not in upstream.calls.last.request.headers


_VERTEX_GEMINI_MODEL_PATH: Final = "v1/projects/deployment-project/locations/global/publishers/google/models/gemini-2.5-flash"
_VERTEX_LABEL_FIELDS: Final = ("user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata")
_VERTEX_IDENTITY_LABELS: Final = {"user_api_key_alias": "search-bot", "user_api_key_team_alias": "search-team"}
_VERTEX_SPEND_LOGS_LABELS: Final = {"cost_center": "cc-42"}
_VERTEX_FORGED_LABELS: Final = {"user_api_key_team_alias": "other-team", "env": "prod", "app": "chat"}
_VERTEX_GEMINI_REPLY: Final = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}]}


@pytest.fixture
def search_team_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy._types import UserAPIKeyAuth

    async def authenticate(request: object, api_key: str) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key=api_key,
            key_alias="search-bot",
            team_id="team-search",
            team_alias="search-team",
            metadata={"spend_logs_metadata": _VERTEX_SPEND_LOGS_LABELS},
        )

    monkeypatch.setattr(proxy_server, "user_custom_auth", authenticate)


def _gemini_upstream(respx_mock: "respx.MockRouter", action: str) -> respx.Route:
    reply: Final = (
        httpx.Response(200, json=_VERTEX_GEMINI_REPLY)
        if action == "generateContent"
        else httpx.Response(
            200, text=f"data: {json.dumps(_VERTEX_GEMINI_REPLY)}\n\n", headers={"content-type": "text/event-stream"}
        )
    )
    return respx_mock.post(url__startswith=f"https://aiplatform.googleapis.com/{_VERTEX_GEMINI_MODEL_PATH}:").mock(
        return_value=reply
    )


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize("action", ["generateContent", "streamGenerateContent"])
@pytest.mark.parametrize(
    ("caller_labels", "expected_labels"),
    [
        pytest.param(
            _VERTEX_FORGED_LABELS,
            {**_VERTEX_IDENTITY_LABELS, "env": "prod", "app": "chat", **_VERTEX_SPEND_LOGS_LABELS},
            id="forged-team-replaced-caller-labels-kept",
        ),
        pytest.param(None, {**_VERTEX_IDENTITY_LABELS, **_VERTEX_SPEND_LOGS_LABELS}, id="identity-added"),
    ],
)
def test_vertex_gemini_pass_through_bills_the_callers_own_identity(
    vertex_proxy: TestClient,
    respx_mock: "respx.MockRouter",
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    caller_labels: Mapping[str, str] | None,
    expected_labels: Mapping[str, str],
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = _gemini_upstream(respx_mock, action)
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        **({} if caller_labels is None else {"labels": caller_labels}),
    }

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_GEMINI_MODEL_PATH}:{action}", json=body, headers={"x-litellm-api-key": "sk-caller"}
    )

    assert response.status_code == 200
    sent: Final = json.loads(upstream.calls.last.request.content)
    assert list(sent["labels"].items()) == list(expected_labels.items())
    assert {key: value for key, value in sent.items() if key != "labels"} == {
        key: value for key, value in body.items() if key != "labels"
    }


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize("action", ["generateContent", "streamGenerateContent"])
def test_vertex_gemini_pass_through_forwards_the_body_untouched_with_the_setting_off(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", action: str
) -> None:
    upstream: Final = _gemini_upstream(respx_mock, action)
    body: Final = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "labels": _VERTEX_FORGED_LABELS}

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_GEMINI_MODEL_PATH}:{action}", json=body, headers={"x-litellm-api-key": "sk-caller"}
    )

    assert response.status_code == 200
    assert list(json.loads(upstream.calls.last.request.content).items()) == list(body.items())


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize(
    "endpoint",
    [
        f"{_VERTEX_GEMINI_MODEL_PATH}:predict",
        f"{_VERTEX_GEMINI_MODEL_PATH}:predictLongRunning",
        "v1/projects/deployment-project/locations/global/batchPredictionJobs",
    ],
    ids=["predict", "predict-long-running", "batch-prediction-job"],
)
def test_vertex_pass_through_adds_identity_labels_to_predict_and_batch_jobs(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{endpoint}").mock(
        return_value=httpx.Response(200, json={})
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{endpoint}", json={"instances": [{"content": "hi"}]}, headers={"x-litellm-api-key": "sk-caller"}
    )

    assert response.status_code == 200
    assert json.loads(upstream.calls.last.request.content) == {
        "instances": [{"content": "hi"}],
        "labels": {**_VERTEX_IDENTITY_LABELS, **_VERTEX_SPEND_LOGS_LABELS},
    }


@pytest.mark.usefixtures("search_team_caller")
def test_vertex_pass_through_replaces_forged_labels_on_any_endpoint(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_GEMINI_MODEL_PATH}:countTokens").mock(
        return_value=httpx.Response(200, json={"totalTokens": 1})
    )
    body: Final = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "labels": _VERTEX_FORGED_LABELS}

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_GEMINI_MODEL_PATH}:countTokens", json=body, headers={"x-litellm-api-key": "sk-caller"}
    )

    assert response.status_code == 200
    assert list(json.loads(upstream.calls.last.request.content)["labels"].items()) == list(
        {**_VERTEX_IDENTITY_LABELS, "env": "prod", "app": "chat", **_VERTEX_SPEND_LOGS_LABELS}.items()
    )


@pytest.mark.usefixtures("search_team_caller")
def test_vertex_pass_through_adds_no_labels_where_the_caller_sent_none_outside_billed_methods(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_GEMINI_MODEL_PATH}:countTokens").mock(
        return_value=httpx.Response(200, json={"totalTokens": 1})
    )
    body: Final = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_GEMINI_MODEL_PATH}:countTokens", json=body, headers={"x-litellm-api-key": "sk-caller"}
    )

    assert response.status_code == 200
    assert json.loads(upstream.calls.last.request.content) == body
    assert "x-vertex-ai-labels" not in upstream.calls.last.request.headers


def _labels_header(labels: Mapping[str, str]) -> str:
    return base64.b64encode(json.dumps(dict(labels)).encode()).decode()


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize(
    ("model_path", "caller_headers"),
    [
        pytest.param(
            _VERTEX_DEPLOYMENT_PATH,
            {"x-pass-x-vertex-ai-labels": _labels_header(_VERTEX_FORGED_LABELS)},
            id="x-pass-header-on-deployment-credentials",
        ),
        pytest.param(
            _VERTEX_CALLER_PATH,
            {
                "Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}",
                "X-Vertex-AI-Labels": _labels_header(_VERTEX_FORGED_LABELS),
            },
            id="forwarded-header-on-caller-credentials",
        ),
        pytest.param(
            "v1/projects/publishers/locations/global/publishers/anthropic/models/claude-haiku-4-5:rawPredict",
            {
                "Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}",
                "X-Vertex-AI-Labels": _labels_header(_VERTEX_FORGED_LABELS),
            },
            id="project-id-named-publishers",
        ),
    ],
)
def test_vertex_claude_pass_through_replaces_a_forged_labels_header(
    vertex_proxy: TestClient,
    respx_mock: "respx.MockRouter",
    monkeypatch: pytest.MonkeyPatch,
    model_path: str,
    caller_headers: Mapping[str, str],
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{model_path}").mock(
        return_value=httpx.Response(200, json=_VERTEX_CLAUDE_REPLY)
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{model_path}", json=_VERTEX_CLAUDE_BODY, headers={"x-litellm-api-key": "sk-caller", **caller_headers}
    )

    assert (response.status_code, response.json()) == (200, _VERTEX_CLAUDE_REPLY)
    outbound: Final = upstream.calls.last.request
    sent_headers: Final = outbound.headers.get_list("x-vertex-ai-labels")
    assert len(sent_headers) == 1
    assert list(json.loads(base64.b64decode(sent_headers[0])).items()) == list(
        {**_VERTEX_IDENTITY_LABELS, "env": "prod", "app": "chat", **_VERTEX_SPEND_LOGS_LABELS}.items()
    )
    assert json.loads(outbound.content) == _VERTEX_CLAUDE_BODY


@pytest.mark.usefixtures("search_team_caller")
def test_vertex_claude_pass_through_moves_body_labels_into_the_labels_header(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_DEPLOYMENT_PATH}").mock(
        return_value=httpx.Response(200, json=_VERTEX_CLAUDE_REPLY)
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_DEPLOYMENT_PATH}",
        json={**_VERTEX_CLAUDE_BODY, "labels": _VERTEX_FORGED_LABELS},
        headers={"x-litellm-api-key": "sk-caller"},
    )

    assert (response.status_code, response.json()) == (200, _VERTEX_CLAUDE_REPLY)
    outbound: Final = upstream.calls.last.request
    sent_headers: Final = outbound.headers.get_list("x-vertex-ai-labels")
    assert len(sent_headers) == 1
    assert list(json.loads(base64.b64decode(sent_headers[0])).items()) == list(
        {**_VERTEX_IDENTITY_LABELS, "env": "prod", "app": "chat", **_VERTEX_SPEND_LOGS_LABELS}.items()
    )
    assert json.loads(outbound.content) == _VERTEX_CLAUDE_BODY


@pytest.mark.usefixtures("search_team_caller")
def test_vertex_claude_pass_through_forwards_the_callers_labels_header_with_the_setting_off(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter"
) -> None:
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_CALLER_PATH}").mock(
        return_value=httpx.Response(200, json=_VERTEX_CLAUDE_REPLY)
    )
    forged_header: Final = _labels_header(_VERTEX_FORGED_LABELS)

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_CALLER_PATH}",
        json=_VERTEX_CLAUDE_BODY,
        headers={
            "x-litellm-api-key": "sk-caller",
            "Authorization": f"Bearer {_VERTEX_CALLER_GOOGLE_TOKEN}",
            "X-Vertex-AI-Labels": forged_header,
        },
    )

    assert response.status_code == 200
    assert upstream.calls.last.request.headers.get_list("x-vertex-ai-labels") == [forged_header]


@pytest.mark.usefixtures("search_team_caller")
def test_vertex_mistral_pass_through_moves_body_labels_into_the_labels_header(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    endpoint: Final = (
        "v1/projects/deployment-project/locations/global/publishers/mistralai/models/mistral-medium-3:rawPredict"
    )
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{endpoint}").mock(
        return_value=httpx.Response(200, json={})
    )
    body: Final = {"model": "mistral-medium-3", "messages": [{"role": "user", "content": "hi"}]}

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{endpoint}",
        json={**body, "labels": _VERTEX_FORGED_LABELS},
        headers={"x-litellm-api-key": "sk-caller"},
    )

    assert response.status_code == 200
    outbound: Final = upstream.calls.last.request
    assert json.loads(base64.b64decode(outbound.headers["x-vertex-ai-labels"])) == {
        **_VERTEX_IDENTITY_LABELS,
        "env": "prod",
        "app": "chat",
        **_VERTEX_SPEND_LOGS_LABELS,
    }
    assert json.loads(outbound.content) == body


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize(
    "endpoint",
    [
        "v1/projects/deployment-project/locations/global/endpoints/123:rawPredict",
        "v1/projects/deployment-project/locations/global/publishers/ai21/models/jamba-1.5-mini:streamRawPredict",
    ],
    ids=["custom-endpoint", "publisher-without-a-labels-header"],
)
def test_vertex_raw_predict_pass_through_leaves_other_publishers_as_the_caller_sent_them(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{endpoint}").mock(
        return_value=httpx.Response(200, json={})
    )
    body: Final = {"instances": [{"content": "hi"}], "labels": _VERTEX_FORGED_LABELS}

    response: Final = vertex_proxy.post(f"/vertex_ai/{endpoint}", json=body, headers={"x-litellm-api-key": "sk-caller"})

    assert response.status_code == 200
    outbound: Final = upstream.calls.last.request
    assert json.loads(outbound.content) == body
    assert "x-vertex-ai-labels" not in outbound.headers


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize("http_method", ["PATCH", "PUT"])
def test_vertex_pass_through_leaves_labels_on_non_post_requests_as_the_caller_sent_them(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch, http_method: str
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    endpoint: Final = "v1/projects/deployment-project/locations/global/endpoints/123"
    upstream: Final = respx_mock.route(method=http_method, url=f"https://aiplatform.googleapis.com/{endpoint}").mock(
        return_value=httpx.Response(200, json={})
    )
    body: Final = {"labels": _VERTEX_FORGED_LABELS}

    response: Final = vertex_proxy.request(
        http_method, f"/vertex_ai/{endpoint}", json=body, headers={"x-litellm-api-key": "sk-caller"}
    )

    assert response.status_code == 200
    assert json.loads(upstream.calls.last.request.content) == body


@pytest.mark.usefixtures("search_team_caller")
@pytest.mark.parametrize("action", ["generateContent", "countTokens"])
def test_vertex_gemini_pass_through_drops_a_callers_labels_header_where_it_writes_body_labels(
    vertex_proxy: TestClient, respx_mock: "respx.MockRouter", monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    monkeypatch.setattr(litellm, "vertex_request_label_fields", list(_VERTEX_LABEL_FIELDS))
    upstream: Final = respx_mock.post(f"https://aiplatform.googleapis.com/{_VERTEX_GEMINI_MODEL_PATH}:{action}").mock(
        return_value=httpx.Response(200, json=_VERTEX_GEMINI_REPLY)
    )

    response: Final = vertex_proxy.post(
        f"/vertex_ai/{_VERTEX_GEMINI_MODEL_PATH}:{action}",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "labels": {"env": "prod"}},
        headers={"x-litellm-api-key": "sk-caller", "x-pass-x-vertex-ai-labels": _labels_header(_VERTEX_FORGED_LABELS)},
    )

    assert response.status_code == 200
    outbound: Final = upstream.calls.last.request
    assert "x-vertex-ai-labels" not in outbound.headers
    assert json.loads(outbound.content)["labels"] == {
        **_VERTEX_IDENTITY_LABELS,
        "env": "prod",
        **_VERTEX_SPEND_LOGS_LABELS,
    }
