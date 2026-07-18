"""Live e2e: the gateway's authorization and error-shape contract.

A virtual key may only call models in its allow-list and route groups in its
allowed_routes; both denials are a 403 raised before any provider is touched. A
syntactically valid request naming a non-existent model is a 400 with a JSON body,
never forwarded and never a 5xx. Migrated from
litellm-regression-tests/tests/test_access_control.py: the source asserted 401 for
the disallowed-model case against an older proxy, but the current contract
(auth_checks.py) is a 403 key_model_access_denied, and the unknown-route check is
replaced by a stronger route-permission check (an llm-only key rejected from a
management route).
"""

from __future__ import annotations

import json

import pytest

from access_control_client import (
    AccessControlClient,
    ALL_TEAM_MODELS_SENTINEL,
    MODEL_ACCESS_DENIED_MARKER,
    ROUTE_NOT_ALLOWED_MARKER,
)
from e2e_config import unique_marker
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

ALLOWED_MODEL = "gemini-2.5-flash"
DISALLOWED_MODEL = "gpt-5.5"
PROXY_CHAT_MODELS = frozenset({"gpt-5.5", "claude-haiku-4-5", "gemini-2.5-flash"})


def _is_json(body: str) -> bool:
    try:
        json.loads(body)
        return True
    except ValueError:
        return False


class TestAccessControl:
    def test_disallowed_model_is_denied_403(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = resources.key(models=[ALLOWED_MODEL])
        result = client.chat_status(
            key, DISALLOWED_MODEL, f"capital of France? {unique_marker()}"
        )
        assert result.status_code == 403, (
            f"key limited to {ALLOWED_MODEL!r} calling {DISALLOWED_MODEL!r} must be "
            f"denied 403, got {result.status_code}: {result.body[:300]}"
        )
        assert MODEL_ACCESS_DENIED_MARKER in result.body, (
            f"403 body must be a model-access denial, got: {result.body[:300]}"
        )

    def test_llm_only_key_forbidden_from_management_route_403(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = client.llm_only_key()
        resources.defer(lambda: client.delete_key(key))
        result = client.create_model_status(key, f"e2e-forbidden-{unique_marker()}")
        assert result.status_code == 403, (
            f"llm-only key calling a management route must be denied 403, got "
            f"{result.status_code}: {result.body[:300]}"
        )
        assert ROUTE_NOT_ALLOWED_MARKER in result.body, (
            f"403 body must be a route-permission denial, got: {result.body[:300]}"
        )

    def test_unknown_model_returns_400(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = resources.key()
        result = client.chat_status(
            key, f"nonexistent-model-{unique_marker()}", "hi this is a test"
        )
        assert result.status_code == 400, (
            f"unknown model must be rejected 400 before forwarding, got "
            f"{result.status_code}: {result.body[:300]}"
        )
        assert _is_json(result.body), f"400 body must be valid JSON: {result.body[:300]}"


class TestTeamlessAllTeamModels:
    """A key scoped to ``all-team-models`` with no team assigned inherits the
    full proxy model list, exactly as if its models field were left empty. This
    is the intended contract (GH #30737): #29746 tightened it to deny such keys
    and was reverted in #32032. These cases pin both surfaces a client touches,
    listing (GET /v1/models) and calling (/chat/completions), so re-introducing
    the team_id guard on either path fails here."""

    def test_teamless_all_team_models_key_lists_all_proxy_models(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = resources.key(models=[ALL_TEAM_MODELS_SENTINEL])
        listed = set(client.gateway.list_models(key))
        assert PROXY_CHAT_MODELS <= listed, (
            f"teamless {ALL_TEAM_MODELS_SENTINEL!r} key must list every proxy model, "
            f"missing {PROXY_CHAT_MODELS - listed}; got {sorted(listed)}"
        )
        assert ALL_TEAM_MODELS_SENTINEL not in listed, (
            f"the {ALL_TEAM_MODELS_SENTINEL!r} sentinel must expand to real models, "
            f"never surface as a listed model; got {sorted(listed)}"
        )

    def test_teamless_all_team_models_key_can_call_any_model(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = resources.key(models=[ALL_TEAM_MODELS_SENTINEL])
        result = client.chat_status(
            key, ALLOWED_MODEL, f"capital of France? {unique_marker()}"
        )
        assert result.status_code == 200, (
            f"teamless {ALL_TEAM_MODELS_SENTINEL!r} key must be allowed to call "
            f"{ALLOWED_MODEL!r}, got {result.status_code}: {result.body[:300]}"
        )
