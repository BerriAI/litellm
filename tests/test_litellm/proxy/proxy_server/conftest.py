"""Shared fixtures for tests/test_litellm/proxy/proxy_server/.

All fixtures and helpers used by PR1/PR2/PR3 test files live here. Do NOT
add fixtures inside individual test files. If a fixture is missing, add it
here and update the Notion plan.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# Repo root, anchored to this file (not CWD) so the path is correct no
# matter where pytest is invoked from. With the project installed via
# uv this is defensive — `litellm` already resolves through site-packages
# — but it lets the harness work in editable-source layouts too.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


# ---------------------------------------------------------------------------
# normalize() — used by every dict-equality assertion to scrub volatile fields
# ---------------------------------------------------------------------------

VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "key",
        "token",
        "id",
        "request_id",
        "expires",
        "expires_at",
        "litellm_call_id",
        "key_alias",
        "created",
    }
)


def normalize(data: Any, volatile: frozenset[str] = VOLATILE_KEYS) -> Any:
    """Replace volatile field values with "<VOLATILE>" so dict equality works.

    Recursive over dicts and lists. Pass an explicit ``volatile`` set to
    extend or override the default.
    """
    if isinstance(data, dict):
        return {
            k: ("<VOLATILE>" if k in volatile else normalize(v, volatile))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [normalize(v, volatile) for v in data]
    return data


# ---------------------------------------------------------------------------
# app + client — session-scoped so app import + TestClient setup amortize
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    """Return the proxy_server FastAPI app with lifespan effectively disabled.

    TestClient used WITHOUT the ``with`` context manager skips the lifespan,
    so the startup event (DB connect, Router init, OTEL setup) never fires.
    Module import still runs once; module-level globals are harmless.
    """
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    from litellm.proxy.proxy_server import app as _app

    return _app


@pytest.fixture(scope="session")
def client(app):
    """TestClient wrapping the session app.

    NOT entered as a context manager — lifespan does not fire. Tests that
    require a real lifespan should use a function-scoped TestClient with
    a ``with`` block locally and accept the per-test cost.
    """
    from fastapi.testclient import TestClient

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# mock_prisma — function-scoped MagicMock with the common table methods stubbed
# ---------------------------------------------------------------------------

# Tables most-touched by proxy_server.py routes. Add to this list if a
# test discovers a missing table.
_PRISMA_TABLES: List[str] = [
    "litellm_verificationtoken",
    "litellm_teamtable",
    "litellm_usertable",
    "litellm_endusertable",
    "litellm_organizationtable",
    "litellm_organizationmembership",
    "litellm_proxymodeltable",
    "litellm_modeltable",
    "litellm_budgettable",
    "litellm_spendlogs",
    "litellm_invitationlink",
    "litellm_credentialstable",
    "litellm_mcpservertable",
    "litellm_objectpermissiontable",
    "litellm_configtable",
    "litellm_audit_log",
    "litellm_dailyuserspend",
    "litellm_dailyteamspend",
    "litellm_dailytagspend",
    "litellm_managed_object_table",
    "litellm_managed_vector_stores_table",
    "litellm_promptstable",
    "litellm_guardrailstable",
    "litellm_managed_files",
    "litellm_session_token_table",
    "litellm_passthrough_endpoint_table",
    "litellm_cron_job",
    "litellm_passthrough_logs",
    "litellm_health_check_table",
    "litellm_mcpusercredentials",
]


def _make_table_mock() -> MagicMock:
    table = MagicMock()
    table.find_unique = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=[])
    table.find_first = AsyncMock(return_value=None)
    table.create = AsyncMock()
    table.create_many = AsyncMock()
    table.update = AsyncMock()
    table.update_many = AsyncMock()
    table.upsert = AsyncMock()
    table.delete = AsyncMock()
    table.delete_many = AsyncMock()
    table.count = AsyncMock(return_value=0)
    table.group_by = AsyncMock(return_value=[])
    table.aggregate = AsyncMock(return_value={})
    return table


@pytest.fixture
def mock_prisma() -> MagicMock:
    """MagicMock prisma_client with .db.<table> methods stubbed.

    Default returns: find_unique/find_first -> None, find_many/group_by -> [],
    count -> 0. Override in a test with::

        mock_prisma.db.litellm_teamtable.find_unique.return_value = ...
    """
    client_mock = MagicMock()
    client_mock.db = MagicMock()
    client_mock.connect = AsyncMock()
    client_mock.disconnect = AsyncMock()
    client_mock.health_check = AsyncMock(return_value=True)
    for table_name in _PRISMA_TABLES:
        setattr(client_mock.db, table_name, _make_table_mock())
    return client_mock


# ---------------------------------------------------------------------------
# auth_as — context manager that overrides user_api_key_auth dependency
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_as(app) -> Callable[..., contextlib.AbstractContextManager]:
    """Context manager that overrides ``user_api_key_auth`` for a role.

    Usage::

        def test_admin_only(client, auth_as):
            from litellm.proxy._types import LitellmUserRoles
            with auth_as(LitellmUserRoles.PROXY_ADMIN):
                response = client.get("/some/admin/route")
                assert response.status_code == 200

    Outside the ``with`` block the override is removed so other tests see
    the real dependency.
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    @contextlib.contextmanager
    def _auth_as(
        role: Any = None,
        user_id: str = "test-user-id",
        team_id: Optional[str] = None,
        api_key: str = "sk-test-key",
        **kwargs: Any,
    ) -> Iterator[Any]:
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

        if role is None:
            role = LitellmUserRoles.PROXY_ADMIN

        fake_auth = UserAPIKeyAuth(
            api_key=api_key,
            user_id=user_id,
            team_id=team_id,
            user_role=role,
            **kwargs,
        )

        async def _override() -> UserAPIKeyAuth:
            return fake_auth

        previous = app.dependency_overrides.get(user_api_key_auth)
        app.dependency_overrides[user_api_key_auth] = _override
        try:
            yield fake_auth
        finally:
            if previous is None:
                app.dependency_overrides.pop(user_api_key_auth, None)
            else:
                app.dependency_overrides[user_api_key_auth] = previous

    return _auth_as


# ---------------------------------------------------------------------------
# Response builders — used by mock_router for parametrized responses
# ---------------------------------------------------------------------------


def make_acompletion_response(
    model: str = "gpt-4",
    messages: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    content: str = "Hello from mock",
    **kwargs: Any,
) -> Any:
    """Build a deterministic chat-completion response.

    Returns:
        - An async generator when ``stream=True``
        - A tool-call shape when ``tools`` is non-empty
        - A plain text response otherwise
    """
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
        ModelResponse,
        Usage,
    )

    if stream:
        return _stream_chunks(model=model, content=content)

    if tools:
        tool_name = tools[0].get("function", {}).get("name", "fake_tool")
        message = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_test",
                    type="function",
                    function=Function(name=tool_name, arguments="{}"),
                )
            ],
        )
    else:
        message = Message(role="assistant", content=content)

    return ModelResponse(
        id="chatcmpl-test",
        choices=[Choices(finish_reason="stop", index=0, message=message)],
        created=0,
        model=model,
        object="chat.completion",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


async def _stream_chunks(
    model: str = "gpt-4", content: str = "Hi"
) -> AsyncIterator[Any]:
    from litellm.types.utils import (
        Delta,
        ModelResponseStream,
        StreamingChoices,
    )

    for piece in [content, ""]:
        yield ModelResponseStream(
            id="chatcmpl-test",
            choices=[
                StreamingChoices(
                    finish_reason=None if piece else "stop",
                    index=0,
                    delta=Delta(content=piece or None, role="assistant"),
                )
            ],
            created=0,
            model=model,
            object="chat.completion.chunk",
        )


def make_embedding_response(
    model: str = "text-embedding-ada-002",
    input: Any = None,
    dimensions: int = 8,
    **kwargs: Any,
) -> Any:
    from litellm.types.utils import EmbeddingResponse

    if isinstance(input, list):
        n = len(input)
    elif input is None:
        n = 1
    else:
        n = 1
    return EmbeddingResponse(
        model=model,
        data=[
            {"embedding": [0.0] * dimensions, "index": i, "object": "embedding"}
            for i in range(n)
        ],
        object="list",
        usage={"prompt_tokens": n, "total_tokens": n},
    )


def make_image_response(model: str = "dall-e-3", **kwargs: Any) -> Any:
    from litellm.types.utils import ImageResponse

    return ImageResponse(
        created=0,
        data=[{"url": "https://example.invalid/image.png"}],
    )


def make_speech_response(**kwargs: Any) -> bytes:
    """Return a fake audio blob. The route serializes bytes to a streaming response."""
    return b"\x00" * 128


def make_transcription_response(**kwargs: Any) -> Any:
    from litellm.types.utils import TranscriptionResponse

    return TranscriptionResponse(text="hello world")


def make_moderation_response(**kwargs: Any) -> Dict[str, Any]:
    return {
        "id": "modr-test",
        "model": "text-moderation-latest",
        "results": [
            {
                "flagged": False,
                "categories": {},
                "category_scores": {},
            }
        ],
    }


# ---------------------------------------------------------------------------
# mock_router — fake Router with all the *async* call surfaces stubbed
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_router() -> MagicMock:
    """A MagicMock standing in for ``llm_router`` with parametrized responses."""

    async def _acompletion(model: str = "gpt-4", messages=None, **kwargs):
        return make_acompletion_response(model=model, messages=messages, **kwargs)

    async def _aembedding(model: str = "text-embedding-ada-002", input=None, **kwargs):
        return make_embedding_response(model=model, input=input, **kwargs)

    async def _aimage_generation(**kwargs):
        return make_image_response(**kwargs)

    async def _aspeech(**kwargs):
        return make_speech_response(**kwargs)

    async def _atranscription(**kwargs):
        return make_transcription_response(**kwargs)

    async def _amoderation(**kwargs):
        return make_moderation_response(**kwargs)

    router = MagicMock()
    router.acompletion = AsyncMock(side_effect=_acompletion)
    router.aembedding = AsyncMock(side_effect=_aembedding)
    router.aimage_generation = AsyncMock(side_effect=_aimage_generation)
    router.aspeech = AsyncMock(side_effect=_aspeech)
    router.atranscription = AsyncMock(side_effect=_atranscription)
    router.amoderation = AsyncMock(side_effect=_amoderation)
    router.model_list = [
        {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
        {
            "model_name": "claude-sonnet",
            "litellm_params": {"model": "anthropic/claude-3-5-sonnet-latest"},
        },
        {
            "model_name": "bedrock-claude",
            "litellm_params": {"model": "bedrock/anthropic.claude-3-5-sonnet"},
        },
    ]
    router.model_names = ["gpt-4", "claude-sonnet", "bedrock-claude"]
    router.get_model_list = MagicMock(return_value=router.model_list)
    return router


# ---------------------------------------------------------------------------
# mock_callbacks_disabled — autouse: zero out global callbacks per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_callbacks_disabled(monkeypatch) -> None:
    """Wipe ``litellm.callbacks`` and friends so tests don't leak side effects."""
    import litellm

    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "input_callback",
        "service_callback",
    ):
        if hasattr(litellm, attr):
            monkeypatch.setattr(litellm, attr, [], raising=False)


# ---------------------------------------------------------------------------
# Builders for DB-like objects (used by routes that load from DB)
# ---------------------------------------------------------------------------


def make_user(
    user_id: str = "user-test",
    role: Any = None,
    teams: Optional[List[str]] = None,
    max_budget: Optional[float] = None,
    spend: float = 0.0,
    **kwargs: Any,
) -> Any:
    from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles

    if role is None:
        role = LitellmUserRoles.INTERNAL_USER

    return LiteLLM_UserTable(
        user_id=user_id,
        user_role=role,
        teams=teams or [],
        max_budget=max_budget,
        spend=spend,
        **kwargs,
    )


def make_team(
    team_id: str = "team-test",
    team_alias: str = "Test Team",
    max_budget: Optional[float] = None,
    spend: float = 0.0,
    members_with_roles: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Any:
    from litellm.proxy._types import LiteLLM_TeamTable

    return LiteLLM_TeamTable(
        team_id=team_id,
        team_alias=team_alias,
        max_budget=max_budget,
        spend=spend,
        members_with_roles=members_with_roles or [],
        **kwargs,
    )


def make_key(
    token: str = "hashed-test-key",
    key_alias: Optional[str] = None,
    team_id: Optional[str] = None,
    user_id: str = "user-test",
    spend: float = 0.0,
    max_budget: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    from litellm.proxy._types import LiteLLM_VerificationToken

    return LiteLLM_VerificationToken(
        token=token,
        key_alias=key_alias,
        team_id=team_id,
        user_id=user_id,
        spend=spend,
        max_budget=max_budget,
        **kwargs,
    )
