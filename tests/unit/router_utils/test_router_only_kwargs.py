from litellm.router_utils.common_utils import without_router_only_kwargs


def test_without_router_only_kwargs_removes_model_alias_map():
    kwargs = {
        "model": "gpt-4o",
        "model_alias_map": {"alias": "gpt-4o"},
        "silent_model": "shadow-model",
        "include_fallback_errors": True,
        "_mid_stream_continuation": True,
    }

    assert without_router_only_kwargs(kwargs) == {"model": "gpt-4o"}
    assert kwargs["model_alias_map"] == {"alias": "gpt-4o"}
