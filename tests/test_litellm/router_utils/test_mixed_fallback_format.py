from litellm.router_utils.fallback_event_handlers import (
    _check_non_standard_fallback_format,
)


def test_mixed_direct_fallback_targets_are_non_standard_format() -> None:
    fallbacks = [
        "backup-a",
        {"model": "backup-b", "temperature": 0},
    ]

    assert _check_non_standard_fallback_format(fallbacks) is True


def test_model_group_mapping_remains_standard_format() -> None:
    fallbacks = [{"primary": ["backup-a", {"model": "backup-b"}]}]

    assert _check_non_standard_fallback_format(fallbacks) is False


def test_model_key_mapping_with_list_targets_remains_standard_format() -> None:
    fallbacks = [{"model": ["backup-a", "backup-b"]}]

    assert _check_non_standard_fallback_format(fallbacks) is False


def test_multi_key_list_mapping_remains_standard_format() -> None:
    fallbacks = [{"model": ["qwen-backup"], "region": ["us-east-1"]}]

    assert _check_non_standard_fallback_format(fallbacks) is False


def test_list_valued_direct_model_with_request_overrides_is_non_standard_format() -> None:
    fallbacks = [{"model": ["backup-a", "backup-b"], "temperature": 0}]

    assert _check_non_standard_fallback_format(fallbacks) is True


def test_mixed_string_and_list_valued_model_target_is_non_standard_format() -> None:
    fallbacks = ["backup-a", {"model": ["backup-b", "backup-c"]}]

    assert _check_non_standard_fallback_format(fallbacks) is True
