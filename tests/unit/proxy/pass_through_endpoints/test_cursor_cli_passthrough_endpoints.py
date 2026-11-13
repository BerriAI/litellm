import asyncio
import gzip
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, nullcontext
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.types import Message
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.enterprise.enterprise_hooks.openai_moderation import ENTERPRISE_OpenAI_Moderation
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_JWTAuth,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TagTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_UserTable,
    LitellmUserRoles,
    RoleBasedPermissions,
    ScopeMapping,
    UserAPIKeyAuth,
)
from litellm.proxy.agent_endpoints import agent_registry
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.common_utils.http_parsing_utils import safe_set_request_parsed_body
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    project_cache_key,
    tag_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.pass_through_endpoints.cursor_cli_passthrough_endpoints import cursor_cli_passthrough
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.utils import hash_token
from litellm.types.agents import AgentResponse
from tests._master_key import MASTER_KEY

_RUN: Final = "agent.v1.AgentService/RunSSE"
_APPEND: Final = "aiserver.v1.BidiService/BidiAppend"
_BASE: Final = "https://api2.cursor.sh"
_TOKEN: Final = "test-cursor-bearer"
_HEADERS: Final = {"authorization": f"Bearer {_TOKEN}", "x-litellm-api-key": MASTER_KEY}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> Iterator[TestClient]:
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(proxy_server.proxy_logging_obj.internal_usage_cache, "dual_cache", DualCache())
    yield TestClient(proxy_server.app)


@pytest.mark.parametrize("endpoint,content_type", [(_RUN, "application/connect+proto"), (_APPEND, "application/proto")])
@pytest.mark.parametrize("compressed", [False, True])
def test_native_bytes_and_credentials_reach_only_cursor(
    client: TestClient, respx_mock: respx.MockRouter, endpoint: str, content_type: str, compressed: bool
) -> None:
    payload: Final = b"\x00\xff\x80\x08\x01\x12\x03abc"
    body: Final = gzip.compress(payload, mtime=0) if compressed else payload
    result: Final = b"\x02\x00\x00\x00\x02{}\xff"
    upstream: Final = respx_mock.post(f"{_BASE}/{endpoint}").respond(
        content=result, headers={"content-type": "application/connect+json"}
    )
    response: Final = client.post(
        f"/cursor-cli/{endpoint}",
        content=body,
        headers={
            **_HEADERS,
            "content-type": content_type,
            **({"content-encoding": "gzip"} if compressed else {}),
            "cookie": "test-cookie",
            "x-copy-key": MASTER_KEY,
            "x-pass-authorization": "Bearer test-injected-token",
            "x-pass-x-litellm-api-key": "test-injected-key",
            "connection": "x-hop-secret",
            "x-hop-secret": "test-hop-value",
            "connect-protocol-version": "1",
        },
    )
    assert response.status_code == 200
    assert response.content == result
    assert response.headers["content-type"] == "application/connect+json"
    sent: Final = upstream.calls.last.request
    decoded: Final = gzip.decompress(sent.content) if sent.headers.get("content-encoding") == "gzip" else sent.content
    assert decoded == payload
    assert sent.headers["authorization"] == f"Bearer {_TOKEN}"
    assert sent.headers["connect-protocol-version"] == "1"
    assert not {"cookie", "x-copy-key", "x-litellm-api-key", "x-hop-secret"}.intersection(sent.headers)
    assert not any(name.startswith("x-pass-") for name in sent.headers)


def test_upstream_compression_keeps_native_frames_and_response_headers_consistent(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    frames: Final = b"\x00\x00\x00\x00\x02{}\x02\x00\x00\x00\x02{}"
    respx_mock.post(f"{_BASE}/{_RUN}").respond(
        content=gzip.compress(frames, mtime=0),
        headers={"content-type": "application/connect+proto", "content-encoding": "gzip"},
    )
    response: Final = client.post(f"/cursor-cli/{_RUN}", headers=_HEADERS, content=b"\xff")
    assert response.status_code == 200
    assert response.content == frames
    assert "content-encoding" not in response.headers


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": f"Bearer {_TOKEN}"},
        {"x-litellm-api-key": MASTER_KEY},
        {**_HEADERS, "x-litellm-api-key": "sk-invalid"},
        {**_HEADERS, "authorization": f"Bearer {MASTER_KEY}"},
        {**_HEADERS, "authorization": "Basic test"},
    ],
)
def test_separate_valid_gateway_and_cursor_credentials_are_required(
    client: TestClient, respx_mock: respx.MockRouter, headers: dict[str, str]
) -> None:
    response: Final = client.post(f"/cursor-cli/{_RUN}", headers=headers, content=b"\xff")
    assert response.status_code in (400, 401, 403)
    assert not respx_mock.calls


def test_gateway_bearer_cannot_replace_the_separate_gateway_header(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    response: Final = client.post(
        f"/cursor-cli/{_RUN}", headers={"authorization": f"Bearer {MASTER_KEY}"}, content=b"\xff"
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "A separate LiteLLM gateway credential is required"
    assert not respx_mock.calls


@pytest.mark.parametrize("credential", [MASTER_KEY, "sk-cursor-normalized-key"])
@pytest.mark.parametrize("template", ["Bearer  {}", "bEaReR {}", "Bearer {}  "])
def test_normalized_gateway_bearers_never_reach_cursor(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    credential: str,
    template: str,
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(credential), value=UserAPIKeyAuth(token=hash_token(credential))
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}",
        headers={"x-litellm-api-key": credential, "authorization": template.format(credential)},
        content=b"\xff",
    )
    assert response.status_code == 401, response.text
    assert not upstream.called


@pytest.mark.parametrize("credential", [MASTER_KEY, "sk-cursor-normalized-key"])
def test_normalized_copied_gateway_secret_is_removed_without_changing_cursor_credentials(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, credential: str
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(credential), value=UserAPIKeyAuth(token=hash_token(credential))
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    bearer: Final = f"bEaReR  {_TOKEN}  "
    response: Final = client.post(
        f"/cursor-cli/{_RUN}",
        headers={"x-litellm-api-key": credential, "authorization": bearer, "x-copy-key": f"bEaReR  {credential}  "},
        content=b"\xff",
    )
    assert response.status_code == 200, response.text
    assert upstream.calls.last.request.headers["authorization"] == bearer
    assert "x-copy-key" not in upstream.calls.last.request.headers


@pytest.mark.parametrize("source", ["key", "team", "project", "header", "master-header"])
@pytest.mark.parametrize("tpm_limit", [0, 100, None])
def test_effective_tag_token_caps_cannot_be_bypassed_by_opaque_requests(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    source: str,
    tpm_limit: int | None,
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    key: Final = MASTER_KEY if source == "master-header" else "sk-cursor-tagged-key"
    tag: Final = "cursor-budget-tag"
    proxy_server.user_api_key_cache.set_cache(
        key=tag_cache_key(tag),
        value=LiteLLM_TagTable(
            tag_name=tag,
            litellm_budget_table=LiteLLM_BudgetTable(budget_id="tag-budget", tpm_limit=tpm_limit, rpm_limit=100),
        ),
    )
    proxy_server.user_api_key_cache.set_cache(
        key="team_id:tag-team",
        value=LiteLLM_TeamTableCachedObj(team_id="tag-team", metadata={"tags": [tag]}),
    )
    proxy_server.user_api_key_cache.set_cache(
        key=project_cache_key("tag-project"),
        value=LiteLLM_ProjectTableCachedObj(project_id="tag-project", metadata={"tags": [tag]}),
    )
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key),
        value=UserAPIKeyAuth(
            token=hash_token(key),
            metadata={"tags": [tag]} if source == "key" else {},
            team_id="tag-team" if source == "team" else None,
            project_id="tag-project" if source == "project" else None,
        ),
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}",
        headers={
            **_HEADERS,
            "x-litellm-api-key": key,
            **({"x-litellm-tags": tag} if source in {"header", "master-header"} else {}),
        },
        content=b"\xff",
    )
    assert response.status_code == (200 if tpm_limit is None else 403), response.text
    assert upstream.called is (tpm_limit is None)


@pytest.mark.parametrize("lookup", ["absent", "error", "no-database"])
def test_opaque_tag_lookup_distinguishes_absent_tags_from_unverifiable_limits(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, lookup: str
) -> None:
    database: Final = MagicMock()
    database.db.litellm_tagtable.find_many = AsyncMock(
        return_value=[], side_effect=ConnectionError("tag database unavailable") if lookup == "error" else None
    )
    monkeypatch.setattr(proxy_server, "prisma_client", None if lookup == "no-database" else database)
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}", headers={**_HEADERS, "x-litellm-tags": "unregistered-tag"}, content=b"\xff"
    )
    assert response.status_code == (403 if lookup == "error" else 200), response.text
    assert upstream.called is (lookup != "error")


@pytest.mark.parametrize("restriction", ["role", "scope"])
def test_signed_jwt_with_model_permissions_cannot_enter_an_opaque_relay(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, restriction: str
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks_url: Final = "https://identity.invalid/cursor/jwks"
    keys: Final = DualCache()
    keys.set_cache(
        key=f"litellm_jwt_auth_keys_{jwks_url}",
        value=[json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))],
    )
    handler: Final = JWTHandler()
    handler.update_environment(
        prisma_client=None,
        user_api_key_cache=keys,
        litellm_jwtauth=LiteLLM_JWTAuth(
            team_ids_jwt_field="groups",
            team_id_jwt_field="team",
            end_user_id_jwt_field=None,
            team_allowed_routes=["/cursor-cli/*"],
            enforce_rbac=restriction == "role",
            enforce_scope_based_access=restriction == "scope",
            scope_mappings=[ScopeMapping(scope="inference", models=["scoped-model"])]
            if restriction == "scope"
            else None,
        ),
    )
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", jwks_url)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.setattr(proxy_server, "jwt_handler", handler)
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "prisma_client", "test-database")
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {
            "enable_jwt_auth": True,
            "role_permissions": [
                RoleBasedPermissions(role=LitellmUserRoles.TEAM, models=["scoped-model"], routes=["/cursor-cli/*"])
            ],
        }
        if restriction == "role"
        else {"enable_jwt_auth": True},
    )
    proxy_server.user_api_key_cache.set_cache(
        key="team_id:cursor-jwt-team", value=LiteLLM_TeamTableCachedObj(team_id="cursor-jwt-team", models=[])
    )
    token: Final = jwt.encode(
        {
            "sub": "cursor-jwt-user",
            "groups": ["cursor-jwt-team"],
            "team": "cursor-jwt-team",
            "scope": "inference",
        },
        private_key,
        algorithm="RS256",
    )
    response: Final = client.post(
        f"/cursor-cli/{_RUN}", headers={**_HEADERS, "x-litellm-api-key": token}, content=b"\xff"
    )
    assert response.status_code == 403
    assert "virtual key or master key" in response.text
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "endpoint", ["aiserver.v1.AiService/GetUsableModels", "https://untrusted.test/run", "../admin"]
)
def test_only_agent_session_rpcs_are_exposed(client: TestClient, respx_mock: respx.MockRouter, endpoint: str) -> None:
    response: Final = client.post(f"/cursor-cli/{endpoint}", headers=_HEADERS, content=b"\xff")
    assert response.status_code in (404, 422)
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "controls",
    [
        {"models": ["allowed-model"]},
        {"team_models": ["allowed-model"]},
        {"tpm_limit": 100},
        {"team_tpm_limit": 100},
        {"metadata": {"guardrails": ["must-inspect"]}},
        {"metadata": {"model_tpm_limit": {"scoped-model": 100}}},
        {"project_metadata": {"model_itpm_limit": {"scoped-model": 100}}},
        {"model_max_budget": {"scoped-model": {"tpm_limit": 100}}},
        {"metadata": {"allowed_tools": ["read_file"]}},
        {"tpm_limit": 0},
        {"allowed_routes": ["/v1/chat/completions"]},
    ],
)
def test_unenforceable_or_disallowed_key_policies_fail_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, controls: dict[str, object]
) -> None:
    key: Final = "sk-cursor-restricted-test"
    monkeypatch.setattr(proxy_server, "prisma_client", "test-database")
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key), value=UserAPIKeyAuth.model_validate({"token": hash_token(key), **controls})
    )
    response: Final = client.post(
        f"/cursor-cli/{_APPEND}", headers={**_HEADERS, "x-litellm-api-key": key}, content=b"\xff"
    )
    assert response.status_code == 403
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "budget_field", ["max_budget", "team_max_budget", "user_max_budget", "organization_max_budget", "end_user_max_budget"]
)
@pytest.mark.parametrize("limit,spend", [(0, 0), (10, 0), (10, 11)])
def test_spend_capped_keys_cannot_use_an_unmetered_cursor_relay(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    budget_field: str,
    limit: int,
    spend: int,
) -> None:
    key: Final = "sk-cursor-spend-capped"
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key),
        value=UserAPIKeyAuth.model_validate({"token": hash_token(key), budget_field: limit, "spend": spend}),
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}", headers={**_HEADERS, "x-litellm-api-key": key}, content=b"\xff"
    )
    assert response.status_code in (403, 422), response.text
    assert not upstream.called


@pytest.fixture
def member_gateway(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> tuple[str, MagicMock]:
    database: Final = MagicMock()
    database.db.litellm_budgettable.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    cache: Final = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    key: Final = "sk-cursor-default-member"
    cache.set_cache(
        key=hash_token(key),
        value=UserAPIKeyAuth(token=hash_token(key), user_id="default-member", team_id="default-team"),
    )
    cache.set_cache(
        key="default-member",
        value=LiteLLM_UserTable(user_id="default-member", user_role=LitellmUserRoles.INTERNAL_USER),
    )
    cache.set_cache(
        key="team_id:default-team",
        value=LiteLLM_TeamTableCachedObj(
            team_id="default-team", metadata={"team_member_budget_id": "default-member-budget"}
        ),
    )
    cache.set_cache(
        key=team_membership_reservation_cache_key(user_id="default-member", team_id="default-team"),
        value=LiteLLM_TeamMembership(user_id="default-member", team_id="default-team"),
    )
    return key, database


@pytest.mark.parametrize("linked_budget", [False, True])
@pytest.mark.parametrize("default_limit", [None, 0, 10])
@pytest.mark.parametrize("model_limited", [False, True])
def test_effective_default_member_limits_control_opaque_admission(
    client: TestClient,
    member_gateway: tuple[str, MagicMock],
    respx_mock: respx.MockRouter,
    linked_budget: bool,
    default_limit: int | None,
    model_limited: bool,
) -> None:
    key, _ = member_gateway
    proxy_server.user_api_key_cache.set_cache(
        key=team_membership_reservation_cache_key(user_id="default-member", team_id="default-team"),
        value=LiteLLM_TeamMembership(
            user_id="default-member",
            team_id="default-team",
            budget_id="linked-member-budget" if linked_budget else None,
            litellm_budget_table=LiteLLM_BudgetTable(budget_id="linked-member-budget") if linked_budget else None,
        ),
    )
    proxy_server.user_api_key_cache.set_cache(
        key="team_member_default_budget:default-member-budget",
        value=LiteLLM_BudgetTable(
            budget_id="default-member-budget",
            max_budget=default_limit,
            model_max_budget={"restricted-model": {"max_budget": 10}} if model_limited else {},
        ),
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}", headers={**_HEADERS, "x-litellm-api-key": key}, content=b"\xff"
    )
    limited: Final = model_limited or (default_limit is not None and default_limit > 0)
    assert response.status_code == (403 if limited else 200), response.text
    assert upstream.called is not limited


@pytest.mark.parametrize("lookup", ["missing", "error"])
def test_unverifiable_default_member_budgets_block_opaque_admission(
    client: TestClient, member_gateway: tuple[str, MagicMock], respx_mock: respx.MockRouter, lookup: str
) -> None:
    key, database = member_gateway
    database.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=None, side_effect=ConnectionError("budget lookup unavailable") if lookup == "error" else None
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}", headers={**_HEADERS, "x-litellm-api-key": key}, content=b"\xff"
    )
    assert response.status_code in (403, 503), response.text
    assert not upstream.called


@pytest.mark.parametrize("caller", ["team-only", "master"])
def test_default_member_limits_do_not_apply_without_a_member_identity(
    client: TestClient, member_gateway: tuple[str, MagicMock], respx_mock: respx.MockRouter, caller: str
) -> None:
    key, _ = member_gateway
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key), value=UserAPIKeyAuth(token=hash_token(key), team_id="default-team")
    )
    proxy_server.user_api_key_cache.set_cache(
        key="team_member_default_budget:default-member-budget",
        value=LiteLLM_BudgetTable(budget_id="default-member-budget", max_budget=10),
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_RUN}",
        headers={**_HEADERS, "x-litellm-api-key": MASTER_KEY if caller == "master" else key},
        content=b"\xff",
    )
    assert response.status_code == 200, response.text
    assert upstream.called


@pytest.mark.parametrize("agent_id,status,forwarded", [(None, 200, True), ("scoped-agent", 403, False)])
def test_agent_bound_keys_cannot_bypass_opaque_model_scopes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    agent_id: str | None,
    status: int,
    forwarded: bool,
) -> None:
    key: Final = "sk-cursor-agent-policy-test"
    agent: Final = AgentResponse(
        agent_id="scoped-agent", agent_name="scoped-agent", agent_card_params={}, access_group_ids=("no-model-grants",)
    )
    registry: Final = agent_registry.AgentRegistry()
    registry.register_agent(agent)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)
    database: Final = MagicMock()
    database.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=agent)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key), value=UserAPIKeyAuth.model_validate({"token": hash_token(key), "agent_id": agent_id})
    )
    upstream: Final = respx_mock.post(f"{_BASE}/{_APPEND}").respond(content=b"\x08\x01")
    response: Final = client.post(
        f"/cursor-cli/{_APPEND}", headers={**_HEADERS, "x-litellm-api-key": key}, content=b"\xff"
    )
    assert response.status_code == status, response.text
    assert upstream.called is forwarded


@pytest.mark.parametrize("event", ["pre_call", "post_call"])
@pytest.mark.parametrize("default_on", [False, True])
def test_configured_content_guardrails_cannot_be_silently_bypassed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, event: str, default_on: bool
) -> None:
    guardrail: Final = CustomGuardrail(guardrail_name="must-inspect", default_on=default_on, event_hook=event)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    response: Final = client.post(f"/cursor-cli/{_RUN}", headers=_HEADERS, content=b"\xff")
    assert response.status_code == 403
    assert not respx_mock.calls


def test_legacy_moderation_cannot_be_silently_bypassed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [ENTERPRISE_OpenAI_Moderation()])
    respx_mock.post(f"{_BASE}/{_RUN}").respond(content=b"\x08\x01")
    response: Final = client.post(f"/cursor-cli/{_RUN}", headers=_HEADERS, content=b"\xff")
    assert response.status_code == 403, response.text
    assert not respx_mock.calls


def test_attached_model_policies_cannot_be_bypassed_without_a_parsed_model(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    policies: Final = get_policy_registry()
    attachments: Final = get_attachment_registry()
    policies.load_policies({"must-inspect": {"guardrails": {"add": ["not-default"]}}})
    attachments.load_attachments([{"policy": "must-inspect", "models": ["scoped-model"]}])
    try:
        response: Final = client.post(f"/cursor-cli/{_RUN}", headers=_HEADERS, content=b"\xff")
        assert response.status_code == 403
        assert not respx_mock.calls
    finally:
        attachments.clear()
        policies.clear()


@pytest.mark.parametrize("status", [307, 401, 429])
def test_provider_status_body_and_redirects_are_not_rewritten(
    client: TestClient, respx_mock: respx.MockRouter, status: int
) -> None:
    body: Final = b'{"code":"resource_exhausted","message":"quota exhausted"}'
    respx_mock.post(f"{_BASE}/{_APPEND}").respond(
        status, content=body, headers={"location": "https://untrusted.test/receive", "retry-after": "10"}
    )
    response: Final = client.post(f"/cursor-cli/{_APPEND}", headers=_HEADERS, content=b"\xff", follow_redirects=False)
    assert response.status_code == status
    assert response.content == body
    assert response.headers["retry-after"] == "10"
    assert len(respx_mock.calls) == 1


class StagedStream(httpx.AsyncByteStream):
    def __init__(self, failure: Exception | None = None, close_gate: asyncio.Event | None = None) -> None:
        self.failure: Final = failure
        self.close_gate: Final = close_gate
        self.closing: Final = asyncio.Event()
        self.release: Final = asyncio.Event()
        self.waiting: Final = asyncio.Event()
        self.closed: Final = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"\x00\x00\x00"
        self.waiting.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        yield b"\x00\x02{}\x02\x00\x00\x00\x02{}"

    async def aclose(self) -> None:
        self.closing.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed.set()


async def _native_stream(owner: AsyncExitStack) -> StreamingResponse:
    async def receive() -> Message:
        return {"type": "http.request", "body": b"\x00", "more_body": False}

    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": f"/cursor-cli/{_RUN}",
            "headers": [(name.encode(), value.encode()) for name, value in _HEADERS.items()],
            "query_string": b"",
        },
        receive,
    )
    safe_set_request_parsed_body(request, {})
    identity: Final = UserAPIKeyAuth(api_key=MASTER_KEY).model_copy(update={"via_virtual_key": True})
    response: Final = await cursor_cli_passthrough("agent.v1.AgentService/RunSSE", request, identity, owner)
    assert isinstance(response, StreamingResponse)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/event-stream", "application/connect+json", "application/connect+proto"])
@pytest.mark.parametrize("status", [200, 429])
async def test_stream_is_incremental_and_closes_on_task_cancellation(
    client: TestClient, respx_mock: respx.MockRouter, content_type: str, status: int
) -> None:
    stream: Final = StagedStream()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(
        return_value=httpx.Response(status, stream=stream, headers={"content-type": content_type})
    )
    async with AsyncExitStack() as owner:
        response: Final = await _native_stream(owner)
        assert response.status_code == status
        assert await anext(response.body_iterator) == b"\x00\x00\x00"
        pending: Final = asyncio.create_task(anext(response.body_iterator))
        await stream.waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert stream.closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,status,forwarded", [(1, 429, False), (2, 200, True)])
async def test_parallel_request_limit_counts_overlapping_agent_rpcs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    limit: int,
    status: int,
    forwarded: bool,
) -> None:
    key: Final = "sk-cursor-concurrency-test"
    limiter: Final = proxy_server.proxy_logging_obj.max_parallel_request_limiter
    monkeypatch.setattr(litellm, "callbacks", [limiter])
    monkeypatch.setattr(litellm, "_async_success_callback", [limiter])
    monkeypatch.setattr(litellm, "_async_failure_callback", [limiter])
    monkeypatch.setattr(proxy_server, "prisma_client", "test-database")
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key),
        value=UserAPIKeyAuth.model_validate({"token": hash_token(key), "max_parallel_requests": limit}),
    )
    stream: Final = StagedStream()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(200, stream=stream))
    upstream: Final = respx_mock.post(f"{_BASE}/{_APPEND}").respond(content=b"\x08\x01")
    headers: Final = {**_HEADERS, "x-litellm-api-key": key}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as caller:
        running: Final = asyncio.create_task(caller.post(f"/cursor-cli/{_RUN}", headers=headers, content=b"\xff"))
        waiting: Final = asyncio.create_task(stream.waiting.wait())
        try:
            completed, _ = await asyncio.wait((running, waiting), return_when=asyncio.FIRST_COMPLETED)
            assert waiting in completed
            response: Final = await caller.post(f"/cursor-cli/{_APPEND}", headers=headers, content=b"\xff")
        finally:
            stream.release.set()
            waiting.cancel()
            run_response: Final = await running
        await GLOBAL_LOGGING_WORKER.flush()
    assert run_response.status_code == 200
    assert response.status_code == status, response.text
    assert upstream.called is forwarded


@pytest.mark.asyncio
async def test_complete_fragmented_stream_preserves_terminal_frames(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    stream: Final = StagedStream()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(200, stream=stream))
    async with AsyncExitStack() as owner:
        response: Final = await _native_stream(owner)
        first: Final = await anext(response.body_iterator)
        stream.release.set()
        last: Final = await anext(response.body_iterator)
        assert isinstance(first, bytes) and isinstance(last, bytes)
        assert first + last == b"\x00\x00\x00\x00\x02{}\x02\x00\x00\x00\x02{}"
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
    assert stream.closed.is_set()


@pytest.mark.asyncio
async def test_closing_response_between_chunks_closes_upstream_immediately(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    stream: Final = StagedStream()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(200, stream=stream))
    async with AsyncExitStack() as owner:
        response: Final = await _native_stream(owner)
        assert await anext(response.body_iterator) == b"\x00\x00\x00"
        await response.body_iterator.aclose()
        assert stream.closed.is_set()


@pytest.mark.asyncio
async def test_upstream_closes_even_when_response_iteration_never_starts(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    stream: Final = StagedStream()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(200, stream=stream))
    async with AsyncExitStack() as owner:
        await _native_stream(owner)
    assert stream.closed.is_set()


class UsageRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
        self.failures: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
        self.post_failures: Final[asyncio.Queue[Exception]] = asyncio.Queue()

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        await self.post_failures.put(original_exception)

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self.events.put(kwargs)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self.failures.put(kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["generator-close", "task-cancel"])
async def test_client_disconnect_logs_after_suspending_upstream_cleanup(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, cancel: bool
) -> None:
    recorder: Final = UsageRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    close_gate: Final = asyncio.Event()
    stream: Final = StagedStream(close_gate=close_gate)
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(200, stream=stream))
    async with AsyncExitStack() as owner:
        response: Final = await _native_stream(owner)
        assert await anext(response.body_iterator) == b"\x00\x00\x00"
        closing: Final = asyncio.create_task(
            anext(response.body_iterator) if cancel else response.body_iterator.aclose()
        )
        try:
            if cancel:
                await stream.waiting.wait()
                closing.cancel()
            await stream.closing.wait()
            assert not closing.done()
            close_gate.set()
            expected: Final = pytest.raises(asyncio.CancelledError) if cancel else nullcontext()
            with expected:
                await closing
        finally:
            close_gate.set()
            await asyncio.gather(closing, return_exceptions=True)
        await GLOBAL_LOGGING_WORKER.flush()
    assert stream.closed.is_set()
    assert recorder.events.qsize() == 1
    assert recorder.failures.empty()
    assert recorder.post_failures.empty()


@pytest.mark.asyncio
async def test_cancelled_session_releases_parallel_request_slot(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    key: Final = "sk-cursor-cancelled-session"
    limiter: Final = proxy_server.proxy_logging_obj.max_parallel_request_limiter
    recorder: Final = UsageRecorder()
    monkeypatch.setattr(litellm, "callbacks", [limiter, recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [limiter, recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [limiter, recorder])
    monkeypatch.setattr(proxy_server, "prisma_client", "test-database")
    proxy_server.user_api_key_cache.set_cache(
        key=hash_token(key),
        value=UserAPIKeyAuth.model_validate({"token": hash_token(key), "max_parallel_requests": 1}),
    )
    stream: Final = StagedStream()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(200, stream=stream))
    respx_mock.post(f"{_BASE}/{_APPEND}").respond(content=b"\x08\x01")
    headers: Final = {**_HEADERS, "x-litellm-api-key": key}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as caller:
        running: Final = asyncio.create_task(caller.post(f"/cursor-cli/{_RUN}", headers=headers, content=b"\xff"))
        waiting: Final = asyncio.create_task(stream.waiting.wait())
        try:
            completed, _ = await asyncio.wait((running, waiting), return_when=asyncio.FIRST_COMPLETED)
            assert waiting in completed
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            await GLOBAL_LOGGING_WORKER.flush()
            assert stream.closed.is_set()
            assert recorder.events.qsize() == 1
            assert recorder.failures.empty()
            response: Final = await caller.post(f"/cursor-cli/{_APPEND}", headers=headers, content=b"\xff")
            assert response.status_code == 200, response.text
        finally:
            stream.release.set()
            waiting.cancel()
            await asyncio.gather(running, waiting, return_exceptions=True)
        await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 429])
async def test_upstream_read_failure_reports_once_without_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, status: int
) -> None:
    recorder: Final = UsageRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    stream: Final = StagedStream(failure=httpx.ReadError("upstream dropped mid-stream"))
    stream.release.set()
    respx_mock.post(f"{_BASE}/{_RUN}").mock(return_value=httpx.Response(status, stream=stream))
    async with AsyncExitStack() as owner:
        response: Final = await _native_stream(owner)
        assert response.status_code == status
        assert await anext(response.body_iterator) == b"\x00\x00\x00"
        with pytest.raises(httpx.ReadError, match="upstream dropped mid-stream"):
            await anext(response.body_iterator)
        await GLOBAL_LOGGING_WORKER.flush()
    assert stream.closed.is_set()
    assert recorder.events.empty()
    assert recorder.failures.qsize() == 1
    assert recorder.post_failures.qsize() == 1
    assert recorder.failures.get_nowait()["standard_logging_object"]["status"] == "failure"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429])
async def test_http_errors_produce_only_failure_callbacks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, status: int
) -> None:
    recorder: Final = UsageRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_failure_callback", [recorder])
    respx_mock.post(f"{_BASE}/{_APPEND}").respond(
        status, content=b"\xff", headers={"content-type": "application/proto"}
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as caller:
        response: Final = await caller.post(f"/cursor-cli/{_APPEND}", headers=_HEADERS, content=b"\xff")
        assert response.status_code == status
        await GLOBAL_LOGGING_WORKER.flush()
        failure: Final = await asyncio.wait_for(recorder.failures.get(), timeout=5)
    assert failure["standard_logging_object"]["status"] == "failure"
    assert recorder.events.empty()
    assert recorder.failures.empty()


@pytest.mark.asyncio
async def test_synchronous_failure_callbacks_receive_raw_http_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    failures: Final[asyncio.Queue[dict[str, object]]] = asyncio.Queue()
    loop: Final = asyncio.get_running_loop()

    def record_failure(
        kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        loop.call_soon_threadsafe(failures.put_nowait, kwargs)

    monkeypatch.setattr(litellm, "failure_callback", [record_failure])
    respx_mock.post(f"{_BASE}/{_APPEND}").respond(429, content=b"\xff", headers={"content-type": "application/proto"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as caller:
        response: Final = await caller.post(f"/cursor-cli/{_APPEND}", headers=_HEADERS, content=b"\xff")
        assert response.status_code == 429
        failure: Final = await asyncio.wait_for(failures.get(), timeout=5)
    assert failure["standard_logging_object"]["status"] == "failure"
    assert failures.empty()


@pytest.mark.asyncio
async def test_transport_logs_never_contain_subscription_or_gateway_secrets(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    recorder: Final = UsageRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    respx_mock.post(f"{_BASE}/{_APPEND}").respond(content=b"\x08\x01", headers={"content-type": "application/proto"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as caller:
        response: Final = await caller.post(
            f"/cursor-cli/{_APPEND}",
            headers={**_HEADERS, "x-copy-key": MASTER_KEY, "cookie": "test-cookie"},
            content=b"\xff",
        )
        assert response.status_code == 200
        logged: Final = await asyncio.wait_for(recorder.events.get(), timeout=5)
    serialized: Final = json.dumps(logged, default=str)
    assert not any(secret in serialized for secret in (_TOKEN, MASTER_KEY, "test-cookie"))
