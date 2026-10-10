"""Regression tests for #45543.

A pre_call guardrail used to write ``data["litellm_metadata"]`` directly. Every
reader resolves the proxy-internal bucket through
``get_metadata_variable_name_from_kwargs``, which answers ``litellm_metadata`` as
soon as that key merely exists, so creating it on a chat-completions request (whose
bucket is ``metadata``) moved every later read onto a dict holding only the
``user_api_key_*`` keys. Tags and router fields stayed behind in ``metadata``:
tag-based routing found no tags and picked a deployment of another model, and
SpendLogs lost ``model_group`` and ``model_id``. Nothing raised.
"""

from typing import Final

import pytest

from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    _ensure_litellm_metadata,
)


@pytest.fixture
def user_api_key_dict() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test-45543", user_id="user-1", team_id="team-1")


def test_chat_completions_request_keeps_its_metadata_bucket(user_api_key_dict: UserAPIKeyAuth) -> None:
    """The reported shape: proxy state already lives in ``metadata``."""
    data: Final = {
        "model": "all-models",
        "metadata": {"tags": ["model:glm-5"], "model_group": "all-models"},
    }

    _ensure_litellm_metadata(data, user_api_key_dict)

    # The bug was creating this key at all; its presence is what flips the readers.
    assert "litellm_metadata" not in data
    assert get_metadata_variable_name_from_kwargs(data) == "metadata"
    # The fields the routing and spend-logging readers need are still where they look.
    assert data["metadata"]["tags"] == ["model:glm-5"]
    assert data["metadata"]["model_group"] == "all-models"
    # And the guardrail still gets the key metadata it asked for.
    assert data["metadata"]["user_api_key_team_id"] == "team-1"
    assert data["metadata"]["user_api_key_user_id"] == "user-1"


def test_existing_metadata_values_are_not_overwritten(user_api_key_dict: UserAPIKeyAuth) -> None:
    """Merging is ``setdefault``: a value already resolved upstream wins."""
    data: Final = {
        "model": "all-models",
        "metadata": {"user_api_key_user_id": "resolved-upstream"},
    }

    _ensure_litellm_metadata(data, user_api_key_dict)

    assert data["metadata"]["user_api_key_user_id"] == "resolved-upstream"
    assert data["metadata"]["user_api_key_team_id"] == "team-1"


def test_litellm_metadata_route_still_uses_litellm_metadata(user_api_key_dict: UserAPIKeyAuth) -> None:
    """Batch and file routes really do keep proxy state in ``litellm_metadata``.

    The fix must not flip those the other way, so the bucket is resolved rather
    than hardcoded in either direction.
    """
    data: Final = {"model": "m", "litellm_metadata": {"tags": ["keep-me"]}}

    _ensure_litellm_metadata(data, user_api_key_dict)

    assert "metadata" not in data
    assert get_metadata_variable_name_from_kwargs(data) == "litellm_metadata"
    assert data["litellm_metadata"]["tags"] == ["keep-me"]
    assert data["litellm_metadata"]["user_api_key_team_id"] == "team-1"


def test_request_without_any_metadata_gets_the_default_bucket(user_api_key_dict: UserAPIKeyAuth) -> None:
    data: Final = {"model": "m"}

    _ensure_litellm_metadata(data, user_api_key_dict)

    assert "litellm_metadata" not in data
    assert data["metadata"]["user_api_key_team_id"] == "team-1"


def test_no_user_api_key_metadata_creates_no_bucket() -> None:
    """``None`` yields no metadata, so nothing should be written at all."""
    data: Final = {"model": "m"}

    _ensure_litellm_metadata(data, None)  # type: ignore[arg-type]  # reason: the hook is reached with no key in tests and on unauthenticated routes

    assert "litellm_metadata" not in data
    assert "metadata" not in data
