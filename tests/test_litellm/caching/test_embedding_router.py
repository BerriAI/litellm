import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching._embedding_router import (
    build_router_embedding_metadata,
    resolve_embedding_router,
)


def test_resolve_returns_router_when_model_is_a_deployment():
    router = MagicMock()
    assert (
        resolve_embedding_router("sem-embed", router, [{"model_name": "sem-embed"}])
        is router
    )


def test_resolve_returns_none_when_model_not_in_router():
    router = MagicMock()
    assert (
        resolve_embedding_router("sem-embed", router, [{"model_name": "other"}]) is None
    )


def test_resolve_returns_none_when_router_is_none():
    assert (
        resolve_embedding_router("sem-embed", None, [{"model_name": "sem-embed"}])
        is None
    )


def test_resolve_returns_none_when_model_list_is_none():
    router = MagicMock()
    assert resolve_embedding_router("sem-embed", router, None) is None


def test_resolve_skips_entries_missing_model_name():
    router = MagicMock()
    model_list = [
        {"litellm_params": {"model": "bedrock/x"}},
        {"model_name": "sem-embed"},
    ]
    assert resolve_embedding_router("sem-embed", router, model_list) is router
    assert resolve_embedding_router("other", router, [{"litellm_params": {}}]) is None


def test_build_metadata_preserves_request_fields_and_adds_flag():
    md = build_router_embedding_metadata(
        {"user_api_key": "sk-x", "user_api_key_team_id": "team-1", "trace_id": "t-1"}
    )
    assert md == {
        "user_api_key": "sk-x",
        "user_api_key_team_id": "team-1",
        "trace_id": "t-1",
        "semantic-cache-embedding": True,
    }


def test_build_metadata_handles_none_and_does_not_mutate_input():
    original = {"user_api_key": "sk-x"}
    md = build_router_embedding_metadata(original)
    assert md == {"user_api_key": "sk-x", "semantic-cache-embedding": True}
    assert original == {"user_api_key": "sk-x"}
    assert build_router_embedding_metadata(None) == {"semantic-cache-embedding": True}
