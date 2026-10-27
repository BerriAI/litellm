import asyncio
import importlib
import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import responses
import respx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


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
    router._initialize_deployment_for_pass_through(
        deployment=deployment,
        custom_llm_provider="vertex_ai",
    )

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
        router._initialize_deployment_for_pass_through(
            deployment=deployment,
            custom_llm_provider="vertex_ai",
        )


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
    router._initialize_deployment_for_pass_through(
        deployment=deployment,
        custom_llm_provider="vertex_ai",
    )

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


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    yield
    loop.close()
    asyncio.set_event_loop(None)


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
