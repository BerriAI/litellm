import copy as copy_module
from unittest.mock import patch

from litellm.router_utils.pattern_match_deployments import PatternMatchRouter


def _openai_wildcard_router() -> PatternMatchRouter:
    router = PatternMatchRouter()
    router.add_pattern(
        "openai/*",
        {"model_name": "openai/*", "litellm_params": {"model": "openai/*"}, "model_info": {}},
    )
    return router


def test_is_match_reports_membership_without_deepcopy():
    """
    Regression for #33636: is_match must report pattern membership without the
    per-deployment copy.deepcopy that route performs. route deep-copied the full
    deployment (including model_info) for every matched deployment, and it was
    called once per model on the GET /v1/models hot path.
    """
    router = _openai_wildcard_router()

    with patch.object(copy_module, "deepcopy", wraps=copy_module.deepcopy) as spy_deepcopy:
        assert router.is_match("openai/gpt-4o-mini") is True
        assert router.is_match("anthropic/claude-3") is False

    assert spy_deepcopy.call_count == 0


def test_is_match_handles_none_request():
    assert _openai_wildcard_router().is_match(None) is False


def test_is_match_matches_route_truthiness():
    router = _openai_wildcard_router()
    for request in ("openai/gpt-4o-mini", "anthropic/claude-3", "openai/"):
        assert router.is_match(request) is (router.route(request) is not None)


def test_route_still_copies_matched_deployments():
    router = _openai_wildcard_router()

    with patch.object(copy_module, "deepcopy", wraps=copy_module.deepcopy) as spy_deepcopy:
        deployments = router.route("openai/gpt-4o-mini")

    assert deployments is not None
    assert deployments[0]["litellm_params"]["model"] == "openai/gpt-4o-mini"
    assert spy_deepcopy.call_count >= 1
