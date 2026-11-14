import asyncio
import base64
import importlib
import json
from collections.abc import Iterator, Mapping
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
_VERTEX_CLAUDE_BODY: Final = {
    "anthropic_version": "vertex-2023-10-16",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}
_VERTEX_CLAUDE_REPLY: Final = {"id": "msg_vertex", "type": "message", "content": []}


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
