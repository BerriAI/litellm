import pytest

from litellm.litellm_core_utils.hook_filter_utils import parse_hook_filters, should_run_hook_for_filters
from litellm.types.hook_filters import HookFilterConfig


class TestShouldRunHookForFilters:
    def test_none_filter_always_runs(self):
        assert (
            should_run_hook_for_filters(
                None, model="gpt-4o", key_alias="prod-key", model_tags=(), request_tags=()
            )
            is True
        )

    def test_empty_filter_always_runs(self):
        assert (
            should_run_hook_for_filters(
                HookFilterConfig(), model="gpt-4o", key_alias="prod-key", model_tags=(), request_tags=()
            )
            is True
        )

    def test_models_dimension_matches_glob(self):
        hook_filter = HookFilterConfig(models=["gpt-4o*"])
        assert (
            should_run_hook_for_filters(
                hook_filter, model="gpt-4o-mini", key_alias=None, model_tags=(), request_tags=()
            )
            is True
        )
        assert (
            should_run_hook_for_filters(
                hook_filter, model="claude-3-opus", key_alias=None, model_tags=(), request_tags=()
            )
            is False
        )

    def test_models_dimension_no_model_never_matches(self):
        hook_filter = HookFilterConfig(models=["gpt-4o*"])
        assert (
            should_run_hook_for_filters(hook_filter, model=None, key_alias=None, model_tags=(), request_tags=())
            is False
        )

    def test_key_aliases_dimension_or_within(self):
        hook_filter = HookFilterConfig(key_aliases=["prod-*", "staging-*"])
        assert (
            should_run_hook_for_filters(
                hook_filter, model=None, key_alias="staging-key-1", model_tags=(), request_tags=()
            )
            is True
        )
        assert (
            should_run_hook_for_filters(
                hook_filter, model=None, key_alias="dev-key-1", model_tags=(), request_tags=()
            )
            is False
        )

    def test_request_tags_matches_if_any_request_tag_matches_any_pattern(self):
        hook_filter = HookFilterConfig(request_tags=["scan-me"])
        assert (
            should_run_hook_for_filters(
                hook_filter,
                model=None,
                key_alias=None,
                model_tags=(),
                request_tags=("unrelated", "scan-me"),
            )
            is True
        )
        assert (
            should_run_hook_for_filters(
                hook_filter, model=None, key_alias=None, model_tags=(), request_tags=("unrelated",)
            )
            is False
        )

    def test_explicit_empty_list_dimension_matches_nothing_unlike_unset(self):
        """An explicit `[]` is not the same as leaving the field unset: it has
        no patterns for any value to satisfy, so it matches nothing, while an
        unset (None) dimension always matches."""
        hook_filter = HookFilterConfig(models=[])
        assert (
            should_run_hook_for_filters(
                hook_filter, model="gpt-4o", key_alias=None, model_tags=(), request_tags=()
            )
            is False
        )

    def test_model_tags_matches_if_any_deployment_tag_matches_any_pattern(self):
        hook_filter = HookFilterConfig(model_tags=["needs-pii-scan"])
        assert (
            should_run_hook_for_filters(
                hook_filter,
                model=None,
                key_alias=None,
                model_tags=("needs-pii-scan", "other-tag"),
                request_tags=(),
            )
            is True
        )
        assert (
            should_run_hook_for_filters(
                hook_filter, model=None, key_alias=None, model_tags=("other-tag",), request_tags=()
            )
            is False
        )

    def test_multiple_dimensions_are_and_ed_together(self):
        hook_filter = HookFilterConfig(models=["gpt-4o*"], key_aliases=["prod-*"])
        # model matches but key_alias doesn't -> overall False
        assert (
            should_run_hook_for_filters(
                hook_filter, model="gpt-4o", key_alias="dev-key", model_tags=(), request_tags=()
            )
            is False
        )
        # both match -> True
        assert (
            should_run_hook_for_filters(
                hook_filter, model="gpt-4o", key_alias="prod-key", model_tags=(), request_tags=()
            )
            is True
        )
        # neither match -> False
        assert (
            should_run_hook_for_filters(
                hook_filter, model="claude-3", key_alias="dev-key", model_tags=(), request_tags=()
            )
            is False
        )


class TestParseHookFilters:
    def test_valid_hook_names_parse_successfully(self):
        parsed = parse_hook_filters(
            "lakera_guard",
            {"async_pre_call_hook": {"models": ["gpt-4o*"]}},
        )
        assert parsed["async_pre_call_hook"].models == ("gpt-4o*",)

    def test_unknown_hook_name_raises(self):
        with pytest.raises(ValueError, match="unknown hook name"):
            parse_hook_filters("lakera_guard", {"not_a_real_hook": {"models": ["gpt-4o*"]}})

    def test_model_tags_on_non_deployment_hook_raises(self):
        with pytest.raises(ValueError, match="model_tags"):
            parse_hook_filters(
                "lakera_guard",
                {"async_pre_call_hook": {"model_tags": ["needs-pii-scan"]}},
            )

    def test_model_tags_on_deployment_scoped_hook_is_allowed(self):
        parsed = parse_hook_filters(
            "lakera_guard",
            {"async_pre_call_deployment_hook": {"model_tags": ["needs-pii-scan"]}},
        )
        assert parsed["async_pre_call_deployment_hook"].model_tags == ("needs-pii-scan",)
