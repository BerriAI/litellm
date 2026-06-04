from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from litellm.proxy.utils import (
    _check_and_merge_model_level_guardrails,
    _merge_guardrails_with_existing,
)


def normalize(value):
    return value


def _router_with_deployment(guardrails, *, by_alias: bool = False):
    """Build a stub router whose `get_deployment(model_id=...)` returns a
    deployment with the given guardrails. ``by_alias=True`` also stubs
    `get_model_list(model_name=...)` to return one matching deployment so the
    pre_call alias fallback (#29652) resolves."""
    deployment = SimpleNamespace(litellm_params={"guardrails": guardrails})
    router = MagicMock()
    router.get_deployment.return_value = deployment
    router.get_model_list.return_value = (
        [{"litellm_params": {"guardrails": guardrails}}] if by_alias else []
    )
    return router


def _router_with_deployments(group_guardrails):
    """Stub a router whose `get_model_list(model_name=...)` returns multiple
    deployments — each entry of `group_guardrails` is the guardrails list for
    one deployment in the group (use None for a deployment with no guardrails).
    Used to pin the UNION-across-deployments fallback (veria-ai Medium on
    #29654)."""
    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"guardrails": g} if g is not None else {}}
        for g in group_guardrails
    ]
    return router


def _router_without_deployment():
    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = []
    return router


def test_check_and_merge_model_level_guardrails_happy_path_merges_lists():
    router = _router_with_deployment(["pii-redact", "toxic-filter"])
    data = {
        "model": "gpt-4o",
        "metadata": {
            "model_info": {"id": "deployment-123"},
            "guardrails": ["user-policy"],
        },
    }
    result = _check_and_merge_model_level_guardrails(data, router)
    snapshot = {
        "model": result["model"],
        "model_info_id": result["metadata"]["model_info"]["id"],
        "guardrails_sorted": sorted(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "model": "gpt-4o",
        "model_info_id": "deployment-123",
        "guardrails_sorted": ["pii-redact", "toxic-filter", "user-policy"],
    }


def test_check_and_merge_model_level_guardrails_returns_data_when_router_none():
    data = {"metadata": {"model_info": {"id": "x"}}, "model": "m", "other": 1}
    result = _check_and_merge_model_level_guardrails(data, None)
    assert result is data
    assert normalize(result) == {
        "metadata": {"model_info": {"id": "x"}},
        "model": "m",
        "other": 1,
    }


def test_check_and_merge_model_level_guardrails_returns_data_when_model_id_missing_and_alias_unknown():
    """When model_id is missing AND the model alias doesn't resolve to a
    deployment (router returns None for both lookups), data is unchanged."""
    router = _router_with_deployment(["pii"])  # by_alias=False by default
    data = {"metadata": {"model_info": {}}, "model": "m", "extra": "v"}
    result = _check_and_merge_model_level_guardrails(data, router)
    snapshot = {
        "is_same_object": result is data,
        "metadata": result["metadata"],
        "model": result["model"],
        "extra": result["extra"],
    }
    assert snapshot == {
        "is_same_object": True,
        "metadata": {"model_info": {}},
        "model": "m",
        "extra": "v",
    }
    router.get_deployment.assert_not_called()
    # Alias fallback was attempted; it just didn't find a deployment.
    router.get_model_list.assert_called_once()


def test_check_and_merge_model_level_guardrails_falls_back_to_model_alias_when_model_id_missing():
    """Pre_call path: model_info.id isn't populated yet because route_request
    hasn't run. The helper must fall back to looking up deployments by the
    model alias (#29652) so DB/UI-assigned guardrails still fire."""
    router = _router_with_deployment(["pii"], by_alias=True)
    data = {"metadata": {"model_info": {}}, "model": "m", "extra": "v"}
    result = _check_and_merge_model_level_guardrails(data, router)
    # Merge happened via the alias fallback.
    assert "pii" in result["metadata"]["guardrails"]
    router.get_model_list.assert_called_once()


def test_check_and_merge_model_level_guardrails_unions_guardrails_across_group_deployments():
    """veria-ai Medium on #29654: when model_id is missing, the router has not
    yet picked the deployment, so taking the FIRST deployment's guardrails
    would silently drop a guardrail defined only on a non-first deployment.
    The fix is to union the guardrails from all deployments in the group."""
    router = _router_with_deployments([["pii"], ["secret-scan"], None])
    data = {"metadata": {"model_info": {}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert sorted(result["metadata"]["guardrails"]) == ["pii", "secret-scan"]


def test_check_and_merge_model_level_guardrails_dedups_guardrails_across_group_deployments():
    """Two deployments with the same guardrail must not produce duplicate
    entries in the merged guardrails list."""
    router = _router_with_deployments([["pii"], ["pii", "secret-scan"]])
    data = {"metadata": {"model_info": {}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert sorted(result["metadata"]["guardrails"]) == ["pii", "secret-scan"]


def test_check_and_merge_model_level_guardrails_group_with_no_guardrails_returns_data():
    """If all deployments in the group have no guardrails (or empty lists),
    the helper returns the data unchanged."""
    router = _router_with_deployments([None, None, []])
    data = {"metadata": {"model_info": {}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert result is data
    assert "guardrails" not in result["metadata"]


def test_check_and_merge_model_level_guardrails_ignores_client_model_info_id_when_distrusted():
    """veria-ai HIGH on #29654: when allow_client_pricing_override is set,
    add_litellm_data_to_request preserves the client-supplied
    metadata.model_info, so a caller could spoof an unknown/unguarded
    model_info.id while requesting a guarded alias and bypass the merge.
    On the pre_call path (trust_client_model_info=False), the helper must
    ignore the spoofed id and fall back to the alias-union path.
    """
    router = MagicMock()
    # Spoofed id resolves to an unguarded deployment.
    spoofed_deployment = SimpleNamespace(litellm_params={"guardrails": []})
    router.get_deployment.return_value = spoofed_deployment
    # The real alias group has a guarded deployment.
    router.get_model_list.return_value = [
        {"litellm_params": {"guardrails": ["alias-secret-scan"]}}
    ]

    data = {
        "model": "guarded-alias",
        "metadata": {"model_info": {"id": "spoofed-unguarded-deployment"}},
    }
    result = _check_and_merge_model_level_guardrails(
        data, router, trust_client_model_info=False
    )
    assert "alias-secret-scan" in result["metadata"]["guardrails"]
    # The model_id lookup must NOT have been used.
    router.get_deployment.assert_not_called()
    router.get_model_list.assert_called_once()


def test_check_and_merge_model_level_guardrails_trusts_client_model_info_id_by_default():
    """Post_call paths (default trust_client_model_info=True) still use the
    model_id route because route_request has populated model_info.id by then.
    """
    router = _router_with_deployment(["post-call-guardrail"])
    data = {
        "model": "any",
        "metadata": {"model_info": {"id": "deployment-123"}},
    }
    result = _check_and_merge_model_level_guardrails(data, router)
    assert "post-call-guardrail" in result["metadata"]["guardrails"]
    router.get_deployment.assert_called_once_with(model_id="deployment-123")


def test_check_and_merge_model_level_guardrails_post_call_accepts_bare_string_guardrail():
    """greptile P1 on #29654: the mypy-narrowing isinstance(list) guard must
    not silently drop bare-string guardrail values that were truthy-accepted
    before. Wrap a scalar into a one-element list so post_call merge keeps
    working with single-guardrail configs."""
    deployment = SimpleNamespace(litellm_params={"guardrails": "scalar-guardrail"})
    router = MagicMock()
    router.get_deployment.return_value = deployment
    data = {"model": "any", "metadata": {"model_info": {"id": "deployment-x"}}}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert "scalar-guardrail" in result["metadata"]["guardrails"]


def test_check_and_merge_model_level_guardrails_alias_union_accepts_bare_string_guardrail():
    """Same scalar-string contract on the pre_call alias-union path."""
    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"guardrails": "scalar-alias-guardrail"}}
    ]
    data = {"model": "alias-m", "metadata": {"model_info": {}}}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert "scalar-alias-guardrail" in result["metadata"]["guardrails"]


def test_check_and_merge_model_level_guardrails_alias_fallback_passes_team_id():
    """veria-ai Medium on #29654: route_request resolves team-scoped public
    model names with metadata.user_api_key_team_id. The pre_call alias
    lookup must pass that team_id to get_model_list, or team-scoped
    deployments are invisible and their guardrails are silently dropped."""
    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = [
        {"litellm_params": {"guardrails": ["team-guardrail"]}}
    ]
    data = {
        "model": "team-scoped-alias",
        "metadata": {
            "model_info": {},
            "user_api_key_team_id": "team-abc",
        },
    }
    result = _check_and_merge_model_level_guardrails(
        data, router, trust_client_model_info=False
    )
    assert "team-guardrail" in result["metadata"]["guardrails"]
    router.get_model_list.assert_called_once_with(
        model_name="team-scoped-alias", team_id="team-abc"
    )


def test_check_and_merge_model_level_guardrails_alias_fallback_reads_team_id_from_litellm_metadata():
    """Backstop: some call sites stash the team id on litellm_metadata
    instead of metadata. The alias fallback should accept either."""
    router = MagicMock()
    router.get_deployment.return_value = None
    router.get_model_list.return_value = []
    data = {
        "model": "alias-m",
        "metadata": {"model_info": {}},
        "litellm_metadata": {"user_api_key_team_id": "team-xyz"},
    }
    _check_and_merge_model_level_guardrails(
        data, router, trust_client_model_info=False
    )
    router.get_model_list.assert_called_once_with(
        model_name="alias-m", team_id="team-xyz"
    )


def test_check_and_merge_model_level_guardrails_returns_data_when_deployment_none():
    router = _router_without_deployment()
    data = {"metadata": {"model_info": {"id": "x"}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert result is data


def test_check_and_merge_model_level_guardrails_returns_data_when_guardrails_none():
    router = _router_with_deployment(None)
    data = {"metadata": {"model_info": {"id": "x"}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert result is data


def test_check_and_merge_model_level_guardrails_handles_missing_metadata():
    """No metadata at all + alias unknown to the router → data unchanged."""
    router = _router_with_deployment(["pii"])  # by_alias=False
    data = {"model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    snapshot = {
        "is_same_object": result is data,
        "model": result["model"],
        "metadata_present": "metadata" in result,
    }
    assert snapshot == {
        "is_same_object": True,
        "model": "m",
        "metadata_present": False,
    }


def test_check_and_merge_model_level_guardrails_raises_when_metadata_is_not_dict():
    router = _router_with_deployment(["pii"])
    data = {"metadata": "not-a-dict", "model": "m"}
    with pytest.raises(AttributeError):
        _check_and_merge_model_level_guardrails(data, router)


def test_merge_guardrails_with_existing_happy_path_combines_lists():
    data = {
        "metadata": {"guardrails": ["a", "b"], "user": "u"},
        "model": "m",
    }
    result = _merge_guardrails_with_existing(data, ["c", "a"])
    snapshot = {
        "guardrails_sorted": sorted(result["metadata"]["guardrails"]),
        "user": result["metadata"]["user"],
        "model": result["model"],
        "is_copy": result is not data,
    }
    assert snapshot == {
        "guardrails_sorted": ["a", "b", "c"],
        "user": "u",
        "model": "m",
        "is_copy": True,
    }


def test_merge_guardrails_with_existing_wraps_scalar_existing_guardrail():
    data = {"metadata": {"guardrails": "single-policy"}}
    result = _merge_guardrails_with_existing(data, ["model-policy"])
    snapshot = {
        "guardrails_sorted": sorted(result["metadata"]["guardrails"]),
        "is_list": isinstance(result["metadata"]["guardrails"], list),
        "count": len(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "guardrails_sorted": ["model-policy", "single-policy"],
        "is_list": True,
        "count": 2,
    }


def test_merge_guardrails_with_existing_wraps_scalar_model_guardrail():
    data = {"metadata": {}}
    result = _merge_guardrails_with_existing(data, "model-policy")
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "is_list": isinstance(result["metadata"]["guardrails"], list),
        "count": len(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "guardrails": ["model-policy"],
        "is_list": True,
        "count": 1,
    }


def test_merge_guardrails_with_existing_empty_existing_empty_model_yields_empty():
    data = {"metadata": {"guardrails": None}}
    result = _merge_guardrails_with_existing(data, None)
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "is_list": isinstance(result["metadata"]["guardrails"], list),
        "count": len(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "guardrails": [],
        "is_list": True,
        "count": 0,
    }


def test_merge_guardrails_with_existing_creates_metadata_when_missing():
    data = {"model": "m"}
    result = _merge_guardrails_with_existing(data, ["g1"])
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "model_preserved": result["model"],
        "original_data_unchanged": "metadata" not in data,
    }
    assert snapshot == {
        "guardrails": ["g1"],
        "model_preserved": "m",
        "original_data_unchanged": True,
    }


def test_merge_guardrails_with_existing_raises_on_unhashable_guardrail():
    data = {"metadata": {"guardrails": [{"unhashable": True}]}}
    with pytest.raises(TypeError):
        _merge_guardrails_with_existing(data, ["g1"])
