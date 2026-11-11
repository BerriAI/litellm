import pytest

from litellm.router_rust.invoke.walk import WalkMetadata


def _no_base_model(_model_id: str) -> str | None:
    return None


@pytest.mark.parametrize("field", ["metadata", "litellm_metadata"])
def test_hop_kwargs_keep_the_requests_key_labels_under_the_walk_stamps(field: str) -> None:
    """The fallback loggers (prometheus' fallback counters) read the caller's key and team off
    the hop's metadata, so a hop keeps the request's bucket and lays its own stamps over it."""
    request = {
        "model": "primary",
        field: {
            "user_api_key_hash": "hashed-key",
            "user_api_key_alias": "key-alias",
            "user_api_key_team_id": "team-id",
            "user_api_key_team_alias": "team-alias",
            "attempted_fallbacks": "stale",
        },
    }
    walk = WalkMetadata(request, field, _no_base_model)
    walk.hop({"parent": 0, "scope": 1, "from": "primary", "to": "fallback", "depth": 1, "max_fallbacks": 5})

    kwargs = walk.hop_kwargs({"scope": 1, "to": "fallback"})

    assert kwargs["model"] == "fallback"
    metadata = kwargs[field]
    assert isinstance(metadata, dict)
    assert metadata["user_api_key_hash"] == "hashed-key"
    assert metadata["user_api_key_alias"] == "key-alias"
    assert metadata["user_api_key_team_id"] == "team-id"
    assert metadata["user_api_key_team_alias"] == "team-alias"
    assert metadata["model_group"] == "fallback"
    assert metadata["original_model_group"] == "primary"
    assert metadata["attempted_fallbacks"] == 1
    assert request[field]["attempted_fallbacks"] == "stale"
    assert "model_group" not in request[field]


def test_hop_kwargs_without_request_metadata_hold_the_walk_stamps() -> None:
    walk = WalkMetadata({"model": "primary"}, "metadata", _no_base_model)
    walk.hop({"parent": 0, "scope": 1, "from": "primary", "to": "fallback", "depth": 1, "max_fallbacks": 5})

    metadata = walk.hop_kwargs({"scope": 1, "to": "fallback"})["metadata"]

    assert metadata == {"original_model_group": "primary", "model_group": "fallback", "attempted_fallbacks": 1}
