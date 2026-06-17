import litellm


def test_suppress_debug_info_has_bool_type_annotation() -> None:
    assert litellm.__annotations__.get("suppress_debug_info") == "bool"
