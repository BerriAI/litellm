from litellm.proxy.auth.v2.data_plane import can_call_model


def test_model_in_allowed_list_is_permitted():
    assert can_call_model(["gpt-4o", "claude-3"], "gpt-4o") is True


def test_model_not_in_allowed_list_is_denied():
    assert can_call_model(["gpt-4o"], "claude-3") is False


def test_empty_list_is_unrestricted_matching_v1():
    # In litellm an empty models list means "no restriction", not "no access".
    assert can_call_model([], "anything") is True
    assert can_call_model(None, "anything") is True


def test_wildcards_are_unrestricted():
    assert can_call_model(["*"], "anything") is True
    assert can_call_model(["all-proxy-models"], "anything") is True
    assert can_call_model(["all-team-models"], "anything") is True


def test_specific_list_denies_unlisted_model():
    assert can_call_model(["gpt-4o", "gpt-4o-mini"], "o1") is False


def test_wildcard_mixed_with_specific_still_unrestricted():
    assert can_call_model(["gpt-4o", "all-proxy-models"], "o1") is True
