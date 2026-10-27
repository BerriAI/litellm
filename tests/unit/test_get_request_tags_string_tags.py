from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


def test_get_request_tags_accepts_string_tags():
    """Regression for #47935: a string in metadata["tags"] crashed the standard
    logging payload with "'str' object has no attribute 'copy'"."""
    tags = StandardLoggingPayloadSetup.get_request_tags({"metadata": {"tags": "production"}}, {})
    assert tags == ["production"]


def test_get_request_tags_list_is_preserved():
    tags = StandardLoggingPayloadSetup.get_request_tags({"metadata": {"tags": ["a", "b"]}}, {})
    assert tags == ["a", "b"]


def test_get_request_tags_falls_back_to_litellm_metadata():
    tags = StandardLoggingPayloadSetup.get_request_tags({"litellm_metadata": {"tags": "x"}}, {})
    assert tags == ["x"]


def test_get_request_tags_empty_when_missing():
    assert StandardLoggingPayloadSetup.get_request_tags({}, {}) == []
