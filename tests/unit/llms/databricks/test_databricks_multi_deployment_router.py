import base64
from typing import Final
from uuid import uuid4

import pytest
import responses
import respx

import litellm
from litellm import Router
from litellm.types.decisions import NoulAnswer

_WS_A: Final = "https://workspace-a.cloud.databricks.com"
_WS_B: Final = "https://workspace-b.cloud.databricks.com"
_BASE_A: Final = f"{_WS_A}/serving-endpoints"
_BASE_B: Final = f"{_WS_B}/serving-endpoints"

_TOKEN_URL_A: Final = f"{_WS_A}/oidc/v1/token"
_TOKEN_URL_B: Final = f"{_WS_B}/oidc/v1/token"

_CHAT_URL_A: Final = f"{_BASE_A}/chat/completions"
_CHAT_URL_B: Final = f"{_BASE_B}/chat/completions"

_EMBED_URL_A: Final = f"{_BASE_A}/embeddings"
_EMBED_URL_B: Final = f"{_BASE_B}/embeddings"

_AGENT_URL_A: Final = f"{_BASE_A}/my-agent/invocations"
_AGENT_URL_B: Final = f"{_BASE_B}/my-agent/invocations"

_DECISIONS_URL_A: Final = f"{_WS_A}/api/2.0/ai-functions/ai-decide"
_DECISIONS_URL_B: Final = f"{_WS_B}/api/2.0/ai-functions/ai-decide"

_CHAT_RESPONSE_A: Final = {
    "id": "chat-a",
    "object": "chat.completion",
    "created": 1234567890,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "from workspace a"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}
_CHAT_RESPONSE_B: Final = {
    "id": "chat-b",
    "object": "chat.completion",
    "created": 1234567890,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "from workspace b"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}

_EMBED_RESPONSE_A: Final = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
    "model": "bge-large-en",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}
_EMBED_RESPONSE_B: Final = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.3, 0.4], "index": 0}],
    "model": "bge-large-en",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}

_AGENT_RESPONSE_A: Final = {
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "agent from a"}]}]
}
_AGENT_RESPONSE_B: Final = {
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "agent from b"}]}]
}

_DECISIONS_RESPONSE: Final = {"response": {"answers": {"is_defect": {"type": "noul", "noul": 0.95}}}}


def _decode_basic_auth(auth_header: str) -> tuple[str, str]:
    assert auth_header.startswith("Basic ")
    encoded: Final = auth_header.split(" ", 1)[1]
    decoded: Final = base64.b64decode(encoded).decode("utf-8")
    client_id, client_secret = decoded.split(":", 1)
    return client_id, client_secret


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_API_BASE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_chat_completions(respx_mock: respx.MockRouter) -> None:
    token_resp_a: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-chat-a", "expires_in": 3600})
    token_resp_b: Final = responses.post(_TOKEN_URL_B, json={"access_token": "token-chat-b", "expires_in": 3600})

    route_a: Final = respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)
    route_b: Final = respx_mock.post(_CHAT_URL_B).respond(json=_CHAT_RESPONSE_B)

    router: Final = Router(
        model_list=[
            {
                "model_name": "workspace-a/chat",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_A,
                    "client_id": "client-a-id",
                    "client_secret": "client-a-secret",
                },
            },
            {
                "model_name": "workspace-b/chat",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_B,
                    "client_id": "client-b-id",
                    "client_secret": "client-b-secret",
                },
            },
        ]
    )

    resp_a: Final = await router.acompletion(
        model="workspace-a/chat",
        messages=[{"role": "user", "content": "hi a"}],
    )
    assert resp_a.choices[0].message.content == "from workspace a"
    assert token_resp_a.call_count == 1
    assert _decode_basic_auth(token_resp_a.calls[0].request.headers["Authorization"]) == (
        "client-a-id",
        "client-a-secret",
    )
    assert route_a.calls.last.request.headers["Authorization"] == "Bearer token-chat-a"
    assert b"client-a-secret" not in route_a.calls.last.request.content

    resp_b: Final = await router.acompletion(
        model="workspace-b/chat",
        messages=[{"role": "user", "content": "hi b"}],
    )
    assert resp_b.choices[0].message.content == "from workspace b"
    assert token_resp_b.call_count == 1
    assert _decode_basic_auth(token_resp_b.calls[0].request.headers["Authorization"]) == (
        "client-b-id",
        "client-b-secret",
    )
    assert route_b.calls.last.request.headers["Authorization"] == "Bearer token-chat-b"


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_embeddings(respx_mock: respx.MockRouter) -> None:
    token_resp_a: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-emb-a", "expires_in": 3600})
    token_resp_b: Final = responses.post(_TOKEN_URL_B, json={"access_token": "token-emb-b", "expires_in": 3600})

    route_a: Final = respx_mock.post(_EMBED_URL_A).respond(json=_EMBED_RESPONSE_A)
    route_b: Final = respx_mock.post(_EMBED_URL_B).respond(json=_EMBED_RESPONSE_B)

    router: Final = Router(
        model_list=[
            {
                "model_name": "workspace-a/embed",
                "litellm_params": {
                    "model": "databricks/bge-large-en",
                    "api_base": _BASE_A,
                    "client_id": "client-emb-a",
                    "client_secret": "secret-emb-a",
                },
            },
            {
                "model_name": "workspace-b/embed",
                "litellm_params": {
                    "model": "databricks/bge-large-en",
                    "api_base": _BASE_B,
                    "client_id": "client-emb-b",
                    "client_secret": "secret-emb-b",
                },
            },
        ]
    )

    emb_a: Final = await router.aembedding(model="workspace-a/embed", input=["text a"])
    assert emb_a.data[0]["embedding"] == [0.1, 0.2]
    assert token_resp_a.call_count == 1
    assert _decode_basic_auth(token_resp_a.calls[0].request.headers["Authorization"]) == (
        "client-emb-a",
        "secret-emb-a",
    )
    assert route_a.calls.last.request.headers["Authorization"] == "Bearer token-emb-a"
    assert b"secret-emb-a" not in route_a.calls.last.request.content

    emb_b: Final = await router.aembedding(model="workspace-b/embed", input=["text b"])
    assert emb_b.data[0]["embedding"] == [0.3, 0.4]
    assert token_resp_b.call_count == 1
    assert _decode_basic_auth(token_resp_b.calls[0].request.headers["Authorization"]) == (
        "client-emb-b",
        "secret-emb-b",
    )
    assert route_b.calls.last.request.headers["Authorization"] == "Bearer token-emb-b"


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_agents_cached_tokens(respx_mock: respx.MockRouter) -> None:
    token_resp_a: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-agent-a", "expires_in": 3600})
    token_resp_b: Final = responses.post(_TOKEN_URL_B, json={"access_token": "token-agent-b", "expires_in": 3600})

    route_a: Final = respx_mock.post(_AGENT_URL_A).respond(json=_AGENT_RESPONSE_A)
    route_b: Final = respx_mock.post(_AGENT_URL_B).respond(json=_AGENT_RESPONSE_B)

    cid_a: Final = f"agent-client-a-{uuid4().hex}"
    cid_b: Final = f"agent-client-b-{uuid4().hex}"

    router: Final = Router(
        model_list=[
            {
                "model_name": "workspace-a/agent",
                "litellm_params": {
                    "model": "databricks_agent/my-agent",
                    "api_base": _BASE_A,
                    "client_id": cid_a,
                    "client_secret": "secret-agent-a",
                },
            },
            {
                "model_name": "workspace-b/agent",
                "litellm_params": {
                    "model": "databricks_agent/my-agent",
                    "api_base": _BASE_B,
                    "client_id": cid_b,
                    "client_secret": "secret-agent-b",
                },
            },
        ]
    )

    resp_a1: Final = await router.acompletion(
        model="workspace-a/agent", messages=[{"role": "user", "content": "ping 1"}]
    )
    assert resp_a1.choices[0].message.content == "agent from a"
    assert token_resp_a.call_count == 1

    resp_a2: Final = await router.acompletion(
        model="workspace-a/agent", messages=[{"role": "user", "content": "ping 2"}]
    )
    assert resp_a2.choices[0].message.content == "agent from a"
    assert token_resp_a.call_count == 1

    resp_b1: Final = await router.acompletion(
        model="workspace-b/agent", messages=[{"role": "user", "content": "ping 3"}]
    )
    assert resp_b1.choices[0].message.content == "agent from b"
    assert token_resp_b.call_count == 1

    resp_b2: Final = await router.acompletion(
        model="workspace-b/agent", messages=[{"role": "user", "content": "ping 4"}]
    )
    assert resp_b2.choices[0].message.content == "agent from b"
    assert token_resp_b.call_count == 1

    assert route_a.calls[0].request.headers["Authorization"] == "Bearer token-agent-a"
    assert route_a.calls[1].request.headers["Authorization"] == "Bearer token-agent-a"
    assert route_b.calls[0].request.headers["Authorization"] == "Bearer token-agent-b"
    assert route_b.calls[1].request.headers["Authorization"] == "Bearer token-agent-b"


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_decisions_apis(respx_mock: respx.MockRouter) -> None:
    token_resp_a: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-dec-a", "expires_in": 3600})
    token_resp_b: Final = responses.post(_TOKEN_URL_B, json={"access_token": "token-dec-b", "expires_in": 3600})

    route_a: Final = respx_mock.post(_DECISIONS_URL_A).respond(json=_DECISIONS_RESPONSE)
    route_b: Final = respx_mock.post(_DECISIONS_URL_B).respond(json=_DECISIONS_RESPONSE)

    cid_a: Final = f"dec-client-a-{uuid4().hex}"
    cid_b: Final = f"dec-client-b-{uuid4().hex}"

    router: Final = Router(
        model_list=[
            {
                "model_name": "workspace-a/decider",
                "litellm_params": {
                    "model": "databricks/ai_decide",
                    "api_base": _BASE_A,
                    "client_id": cid_a,
                    "client_secret": "secret-dec-a",
                },
            },
            {
                "model_name": "workspace-b/decider",
                "litellm_params": {
                    "model": "databricks/ai_decide",
                    "api_base": _BASE_B,
                    "client_id": cid_b,
                    "client_secret": "secret-dec-b",
                },
            },
        ]
    )

    res_a: Final = await router.asystemone(
        model="workspace-a/decider",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "defect?"}},
    )
    assert res_a.answers["is_defect"] == NoulAnswer(type="noul", noul=0.95)
    assert token_resp_a.call_count == 1
    assert route_a.calls.last.request.headers["Authorization"] == "Bearer token-dec-a"

    res_b: Final = await router.adecisions(
        model="workspace-b/decider",
        input="review",
        questions=[{"type": "predicate", "name": "is_defect", "instructions": "defect?"}],
    )
    assert res_b.answers is not None
    assert route_b.calls.last.request.headers["Authorization"] == "Bearer token-dec-b"
    assert token_resp_b.call_count == 1


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_static_api_key_precedence(respx_mock: respx.MockRouter) -> None:
    token_resp_a: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-a", "expires_in": 3600})
    token_resp_b: Final = responses.post(_TOKEN_URL_B, json={"access_token": "token-b", "expires_in": 3600})

    route_a: Final = respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)
    route_b: Final = respx_mock.post(_CHAT_URL_B).respond(json=_CHAT_RESPONSE_B)

    router: Final = Router(
        model_list=[
            {
                "model_name": "deployment-oauth",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_A,
                    "client_id": "oauth-client-id",
                    "client_secret": "oauth-client-secret",
                },
            },
            {
                "model_name": "deployment-static-key",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_B,
                    "api_key": "static-pat-key-b",
                    "client_id": "ignored-client-id",
                    "client_secret": "ignored-client-secret",
                },
            },
        ]
    )

    await router.acompletion(model="deployment-oauth", messages=[{"role": "user", "content": "hello a"}])
    assert token_resp_a.call_count == 1
    assert route_a.calls.last.request.headers["Authorization"] == "Bearer token-a"

    await router.acompletion(model="deployment-static-key", messages=[{"role": "user", "content": "hello b"}])
    assert token_resp_b.call_count == 0
    assert route_b.calls.last.request.headers["Authorization"] == "Bearer static-pat-key-b"


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_credential_rotation(respx_mock: respx.MockRouter) -> None:
    token_resp: Final = responses.post(
        _TOKEN_URL_A,
        json={"access_token": "token-rotated", "expires_in": 3600},
    )
    respx_mock.post(_AGENT_URL_A).respond(json=_AGENT_RESPONSE_A)

    cid: Final = f"rot-client-{uuid4().hex}"

    router: Final = Router(
        model_list=[
            {
                "model_name": "agent-rot",
                "litellm_params": {
                    "model": "databricks_agent/my-agent",
                    "api_base": _BASE_A,
                    "client_id": cid,
                    "client_secret": "secret-v1",
                },
            }
        ]
    )
    await router.acompletion(model="agent-rot", messages=[{"role": "user", "content": "v1"}])
    assert token_resp.call_count == 1
    assert _decode_basic_auth(token_resp.calls[0].request.headers["Authorization"]) == (cid, "secret-v1")

    await router.acompletion(model="agent-rot", messages=[{"role": "user", "content": "v1 again"}])
    assert token_resp.call_count == 1

    router_rotated: Final = Router(
        model_list=[
            {
                "model_name": "agent-rot",
                "litellm_params": {
                    "model": "databricks_agent/my-agent",
                    "api_base": _BASE_A,
                    "client_id": cid,
                    "client_secret": "secret-v2-rotated",
                },
            }
        ]
    )
    await router_rotated.acompletion(model="agent-rot", messages=[{"role": "user", "content": "v2"}])
    assert token_resp.call_count == 2
    assert _decode_basic_auth(token_resp.calls[1].request.headers["Authorization"]) == (cid, "secret-v2-rotated")


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_incomplete_credentials_isolated_failure(
    respx_mock: respx.MockRouter,
) -> None:
    token_resp_a: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-a", "expires_in": 3600})
    respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)

    router: Final = Router(
        model_list=[
            {
                "model_name": "valid-deployment",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_A,
                    "client_id": "valid-id",
                    "client_secret": "valid-secret",
                },
            },
            {
                "model_name": "broken-deployment",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_B,
                    "client_id": "lone-id-without-secret",
                },
            },
        ]
    )

    resp: Final = await router.acompletion(model="valid-deployment", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "from workspace a"
    assert token_resp_a.call_count == 1

    with pytest.raises(litellm.BadRequestError, match="Incomplete Databricks OAuth credentials"):
        await router.acompletion(model="broken-deployment", messages=[{"role": "user", "content": "hi"}])

    resp2: Final = await router.acompletion(model="valid-deployment", messages=[{"role": "user", "content": "hi"}])
    assert resp2.choices[0].message.content == "from workspace a"


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_os_environ_resolution(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("ENV_CLIENT_ID_A", "dyn-client-id-a")
    monkeypatch.setenv("ENV_CLIENT_SECRET_A", "dyn-secret-a")

    token_resp: Final = responses.post(_TOKEN_URL_A, json={"access_token": "dyn-token-a", "expires_in": 3600})
    route_a: Final = respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)

    router: Final = Router(
        model_list=[
            {
                "model_name": "dyn-model",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_A,
                    "client_id": "os.environ/ENV_CLIENT_ID_A",
                    "client_secret": "os.environ/ENV_CLIENT_SECRET_A",
                },
            }
        ]
    )

    resp: Final = await router.acompletion(model="dyn-model", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "from workspace a"
    assert token_resp.call_count == 1
    assert _decode_basic_auth(token_resp.calls[0].request.headers["Authorization"]) == (
        "dyn-client-id-a",
        "dyn-secret-a",
    )
    assert route_a.calls.last.request.headers["Authorization"] == "Bearer dyn-token-a"


@responses.activate
@pytest.mark.asyncio
async def test_multi_deployment_router_deployment_oauth_overrides_ambient_global_api_key(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_KEY", "ambient-global-pat-leak")

    token_resp: Final = responses.post(_TOKEN_URL_A, json={"access_token": "token-dep-oauth", "expires_in": 3600})
    route_a: Final = respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)

    router: Final = Router(
        model_list=[
            {
                "model_name": "oauth-model",
                "litellm_params": {
                    "model": "databricks/dbrx-instruct",
                    "api_base": _BASE_A,
                    "client_id": "dep-client-id",
                    "client_secret": "dep-client-secret",
                },
            }
        ]
    )

    resp: Final = await router.acompletion(model="oauth-model", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "from workspace a"
    assert token_resp.call_count == 1
    assert route_a.calls.last.request.headers["Authorization"] == "Bearer token-dep-oauth"


def _token_endpoint_accepting(accepted: tuple[str, str]) -> None:
    def answer(request: object) -> tuple[int, dict[str, str], str]:
        sent: Final = _decode_basic_auth(getattr(request, "headers")["Authorization"])
        if sent == accepted:
            return 200, {}, '{"access_token": "token-a", "expires_in": 3600}'
        return 401, {}, '{"error": "invalid_client"}'

    responses.add_callback(responses.POST, _TOKEN_URL_A, callback=answer)


def _oauth_router(litellm_params: dict[str, str]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "oauth-model",
                "litellm_params": {"model": "databricks/dbrx-instruct", "api_base": _BASE_A, **litellm_params},
            }
        ]
    )


@responses.activate
@pytest.mark.asyncio
@pytest.mark.parametrize("configured_in", ["deployment", "environment"])
async def test_caller_oauth_pair_rejection_leaves_the_shared_deployment_routable(
    configured_in: str, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    if configured_in == "environment":
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "configured-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "configured-secret")
    _token_endpoint_accepting(("configured-id", "configured-secret"))
    respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)
    router: Final = _oauth_router(
        {"client_id": "configured-id", "client_secret": "configured-secret"} if configured_in == "deployment" else {}
    )

    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(
            model="oauth-model",
            messages=[{"role": "user", "content": "hi"}],
            client_id="caller-id",
            client_secret="caller-secret",
        )

    response: Final = await router.acompletion(model="oauth-model", messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "from workspace a"


@responses.activate
@pytest.mark.asyncio
async def test_configured_oauth_pair_rejection_cools_down_the_deployment(respx_mock: respx.MockRouter) -> None:
    _token_endpoint_accepting(("configured-id", "rotated-secret"))
    respx_mock.post(_CHAT_URL_A).respond(json=_CHAT_RESPONSE_A)
    router: Final = _oauth_router({"client_id": "configured-id", "client_secret": "revoked-secret"})

    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(model="oauth-model", messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(litellm.RouterRateLimitError) as excinfo:
        await router.acompletion(model="oauth-model", messages=[{"role": "user", "content": "hi"}])

    assert excinfo.value.all_deployments_in_cooldown is True
