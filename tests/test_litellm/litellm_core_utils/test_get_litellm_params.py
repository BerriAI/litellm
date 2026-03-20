"""
Tests for get_litellm_params and related helpers.

Ensures backward compatibility after sparse kwargs extraction optimization.
"""


from litellm.litellm_core_utils.get_litellm_params import (
    _OPTIONAL_KWARGS_KEYS,
    _get_base_model_from_litellm_call_metadata,
    get_litellm_params,
)


class TestGetBaseModelFromLitellmCallMetadata:
    def test_none_metadata_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata(None) is None

    def test_empty_metadata_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({}) is None

    def test_missing_model_info_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"foo": "bar"}) is None

    def test_model_info_none_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"model_info": None}) is None

    def test_model_info_empty_dict_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"model_info": {}}) is None

    def test_returns_base_model(self):
        result = _get_base_model_from_litellm_call_metadata(
            {"model_info": {"base_model": "gpt-4"}}
        )
        assert result == "gpt-4"


class TestGetLitellmParamsKwargsExtraction:
    """Verify that optional kwargs are correctly extracted via sparse extraction."""

    def test_no_kwargs_omits_optional_keys(self):
        """When no kwargs passed, optional keys should not be in result."""
        result = get_litellm_params(api_key="test-key")
        for key in _OPTIONAL_KWARGS_KEYS:
            assert key not in result

    def test_present_kwargs_are_extracted(self):
        result = get_litellm_params(
            aws_region_name="us-east-1",
            timeout=30,
            rpm=100,
        )
        assert result["aws_region_name"] == "us-east-1"
        assert result["timeout"] == 30
        assert result["rpm"] == 100

    def test_subset_of_kwargs_only_includes_provided(self):
        """Only provided kwargs appear, others remain absent."""
        result = get_litellm_params(azure_ad_token="token123")
        assert result["azure_ad_token"] == "token123"
        assert "aws_region_name" not in result
        assert "timeout" not in result

    def test_unknown_kwargs_are_ignored(self):
        result = get_litellm_params(some_random_kwarg="value")
        assert "some_random_kwarg" not in result

    def test_all_optional_kwargs_extractable(self):
        """Every key in _OPTIONAL_KWARGS_KEYS can be extracted."""
        kwargs = {key: f"val_{key}" for key in _OPTIONAL_KWARGS_KEYS}
        result = get_litellm_params(**kwargs)
        for key in _OPTIONAL_KWARGS_KEYS:
            assert result[key] == f"val_{key}"


class TestGetLitellmParamsBaseModel:
    """Verify base_model resolution precedence."""

    def test_explicit_base_model_takes_precedence(self):
        result = get_litellm_params(
            base_model="explicit",
            metadata={"model_info": {"base_model": "from-metadata"}},
        )
        assert result["base_model"] == "explicit"

    def test_falls_back_to_metadata(self):
        result = get_litellm_params(
            metadata={"model_info": {"base_model": "from-metadata"}}
        )
        assert result["base_model"] == "from-metadata"

    def test_none_when_no_source(self):
        result = get_litellm_params()
        assert result["base_model"] is None


class TestGetLitellmParamsExplicitFields:
    """Verify explicit parameters are always present in the result."""

    def test_explicit_params_always_present(self):
        result = get_litellm_params()
        # Spot-check a few explicit keys that should always be in the dict
        expected_keys = [
            "acompletion",
            "api_key",
            "force_timeout",
            "verbose",
            "custom_llm_provider",
            "api_base",
            "metadata",
            "model_info",
            "max_retries",
            "ssl_verify",
            "api_version",
        ]
        for key in expected_keys:
            assert key in result

    def test_no_log_from_kwargs(self):
        """no-log can come via **kwargs as well as the explicit param."""
        result = get_litellm_params(**{"no-log": True})
        assert result["no-log"] is True

    def test_no_log_from_explicit_param(self):
        result = get_litellm_params(no_log=True)
        assert result["no-log"] is True


class TestGetLitellmParamsLitellmMetadataMerge:
    """Verify litellm_metadata is merged into metadata for callback visibility."""

    def test_merge_when_no_metadata(self):
        """When only litellm_metadata is provided, it becomes metadata."""
        lm_meta = {"user_api_key_hash": "hashed-abc", "team_id": "t-1"}
        result = get_litellm_params(litellm_metadata=lm_meta)

        assert result["metadata"] == lm_meta
        # Must be a copy, not the same object
        assert result["metadata"] is not lm_meta

    def test_merge_preserves_existing_metadata_keys(self):
        """Existing metadata keys are not overwritten by litellm_metadata."""
        metadata = {"user_api_key": "sk-real", "trace_id": "t-1"}
        lm_meta = {
            "user_api_key_hash": "hashed-abc",
            "user_api_key": "should-not-overwrite",
        }
        result = get_litellm_params(metadata=metadata, litellm_metadata=lm_meta)

        assert result["metadata"]["user_api_key"] == "sk-real"
        assert result["metadata"]["trace_id"] == "t-1"
        assert result["metadata"]["user_api_key_hash"] == "hashed-abc"

    def test_merge_does_not_mutate_caller_metadata(self):
        """The caller's original metadata dict must not be mutated."""
        original_metadata = {"user_api_key": "sk-real"}
        lm_meta = {"user_api_key_hash": "hashed-abc"}
        result = get_litellm_params(
            metadata=original_metadata, litellm_metadata=lm_meta
        )

        assert "user_api_key_hash" not in original_metadata
        assert result["metadata"] is not original_metadata

    def test_merge_does_not_mutate_caller_litellm_metadata(self):
        """The caller's original litellm_metadata dict must not be mutated."""
        lm_meta = {"user_api_key_hash": "hashed-abc"}
        original_keys = set(lm_meta.keys())
        get_litellm_params(litellm_metadata=lm_meta)

        assert set(lm_meta.keys()) == original_keys

    def test_no_merge_when_litellm_metadata_is_none(self):
        """When litellm_metadata is None, metadata is returned as-is."""
        metadata = {"user_api_key": "sk-real"}
        result = get_litellm_params(metadata=metadata, litellm_metadata=None)

        assert result["metadata"] == metadata
