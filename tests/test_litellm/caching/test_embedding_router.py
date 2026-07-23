import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

import litellm

from litellm.caching._embedding_router import (
    build_router_embedding_metadata,
    resolve_embedding_router,
)


def test_resolve_routes_exact_name_model_via_real_router():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "sem-embed",
                "litellm_params": {"model": "text-embedding-3-small"},
            }
        ]
    )
    assert resolve_embedding_router("sem-embed", router) is router


def test_resolve_routes_provider_prefixed_wildcard_via_real_router():
    router = litellm.Router(
        model_list=[
            {"model_name": "bedrock/*", "litellm_params": {"model": "bedrock/*"}}
        ]
    )
    # bedrock/amazon.titan-embed-text-v2:0 is NOT an exact model_name; only the
    # bedrock/* pattern serves it. The old exact-name code returned None here.
    assert (
        resolve_embedding_router("bedrock/amazon.titan-embed-text-v2:0", router)
        is router
    )


def test_resolve_routes_visible_model_group_alias_via_real_router():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "real-embed",
                "litellm_params": {"model": "text-embedding-3-small"},
            }
        ],
        model_group_alias={"aliased-embed": "real-embed"},
    )
    # aliased-embed is only reachable through model_group_alias.
    assert resolve_embedding_router("aliased-embed", router) is router


def test_resolve_returns_none_when_real_router_does_not_serve_model():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "other-embed",
                "litellm_params": {"model": "text-embedding-3-small"},
            }
        ]
    )
    assert resolve_embedding_router("sem-embed", router) is None


def test_resolve_returns_none_when_router_is_none():
    assert resolve_embedding_router("sem-embed", None) is None


def test_resolve_returns_none_when_get_model_list_returns_none():
    # get_model_list is annotated Optional[List]; in practice it returns [],
    # but pin the falsy-None path so the `if ...:` gate stays correct.
    router = MagicMock()
    router.get_model_list = MagicMock(return_value=None)
    assert resolve_embedding_router("sem-embed", router) is None


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
