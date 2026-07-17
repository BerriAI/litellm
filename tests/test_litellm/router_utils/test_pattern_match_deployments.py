import re

from litellm.router_utils.pattern_match_deployments import PatternMatchRouter


def test_return_pattern_matched_deployments_does_not_deepcopy_source():
    # Regression for issue #33636: expanding a wildcard route used to
    # copy.deepcopy the whole deployment per matched deployment, which pegged
    # the proxy CPU for large deployments listed via GET /v1/models. Only
    # litellm_params["model"] is rewritten, so the expansion must shallow-copy:
    # the source deployment stays untouched and every nested object is shared
    # by identity (a reintroduced deepcopy would break the identity assert).
    shared_metadata = {"tier": "gold"}
    deployment = {
        "model_name": "openai/*",
        "litellm_params": {"model": "openai/*", "api_key": "sk-test", "metadata": shared_metadata},
        "model_info": {"id": "abc"},
    }

    router = PatternMatchRouter()
    router.add_pattern("openai/*", deployment)

    matched = router.route("openai/gpt-4")

    assert matched is not None
    (expanded,) = matched
    assert expanded["litellm_params"]["model"] == "openai/gpt-4"

    # source is left untouched
    assert deployment["litellm_params"]["model"] == "openai/*"

    # shallow copy: new top-level and litellm_params dicts, shared nested objects
    assert expanded is not deployment
    assert expanded["litellm_params"] is not deployment["litellm_params"]
    assert expanded["litellm_params"]["metadata"] is shared_metadata
    assert expanded["model_info"] is deployment["model_info"]
