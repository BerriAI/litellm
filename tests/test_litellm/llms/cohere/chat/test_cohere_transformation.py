import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.llms.cohere.common_utils import maybe_drop_unsupported_num_generations


class TestCohereTransform:
    def setup_method(self):
        self.config = CohereChatConfig()
        self.model = "command-r-plus-latest"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_cohere_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}

    # -----------------------------------------------------------------
    # Regression tests for issue: passing n=<value> to a Cohere chat
    # model crashed with "unknown field: parameter 'num_generations' is
    # not a valid field", because Cohere's Chat API has no equivalent of
    # OpenAI's `n` at all -- not even for n=1, which is the common case
    # (many callers pass n=1 explicitly even though it's a no-op).
    # -----------------------------------------------------------------

    def test_n_equal_1_is_a_noop_and_does_not_raise(self):
        """n=1 is exactly what Cohere chat always returns -- must not error or add a field."""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "n": 1},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7}
        assert "num_generations" not in result

    def test_n_greater_than_1_raises_without_drop_params(self):
        """n>1 cannot be honoured by Cohere chat; must raise, not silently send a bad field."""
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={"n": 3},
                optional_params={},
                model=self.model,
                drop_params=False,
            )

    def test_n_greater_than_1_dropped_silently_with_drop_params_true(self):
        """With drop_params=True, n>1 is dropped rather than raised or sent to Cohere."""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "n": 3},
            optional_params={},
            model=self.model,
            drop_params=True,
        )

        assert result == {"temperature": 0.7}
        assert "num_generations" not in result

    def test_n_greater_than_1_dropped_silently_with_litellm_drop_params_true(self, monkeypatch):
        """The global litellm.drop_params=True flag has the same effect as the per-call flag."""
        monkeypatch.setattr(litellm, "drop_params", True)

        result = self.config.map_openai_params(
            non_default_params={"n": 3},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert "num_generations" not in result


class TestCohereV2Transform:
    def setup_method(self):
        self.config = CohereV2ChatConfig()
        self.model = "command-r"

    def test_v2_supports_max_completion_tokens(self):
        """max_completion_tokens must be advertised so get_optional_params does not reject it"""
        assert "max_completion_tokens" in self.config.get_supported_openai_params(self.model)

    def test_v2_max_tokens_only_still_maps(self):
        """max_tokens alone maps to cohere max_tokens when max_completion_tokens is absent"""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 200}

    def test_v2_map_max_completion_tokens_overrides_max_tokens(self):
        """max_completion_tokens maps to cohere max_tokens and overrides max_tokens, matching v1"""
        result = self.config.map_openai_params(
            non_default_params={
                "temperature": 0.7,
                "max_tokens": 200,
                "max_completion_tokens": 256,
            },
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_v2_max_completion_tokens_precedence_is_order_independent(self):
        """max_completion_tokens wins over max_tokens regardless of dict ordering"""
        max_tokens_first = self.config.map_openai_params(
            non_default_params={"max_tokens": 200, "max_completion_tokens": 256},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        max_completion_first = self.config.map_openai_params(
            non_default_params={"max_completion_tokens": 256, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert max_tokens_first == {"max_tokens": 256}
        assert max_completion_first == {"max_tokens": 256}

    def test_v2_default_route_accepts_max_completion_tokens(self):
        """The default cohere_chat route resolves to v2; max_completion_tokens must not raise"""
        optional_params = litellm.get_optional_params(
            model=self.model,
            custom_llm_provider="cohere_chat",
            max_completion_tokens=256,
        )

        assert optional_params["max_tokens"] == 256

    # -----------------------------------------------------------------
    # Same n=<value> regression coverage as TestCohereTransform, on the v2
    # config -- this is the config that actually handles the default
    # cohere_chat route (see test_v2_default_route_accepts_max_completion_tokens
    # above), so this is the path the originally reported bug hit.
    # -----------------------------------------------------------------

    def test_v2_n_equal_1_is_a_noop_and_does_not_raise(self):
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "n": 1},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7}
        assert "num_generations" not in result

    def test_v2_n_greater_than_1_raises_without_drop_params(self):
        with pytest.raises(UnsupportedParamsError):
            self.config.map_openai_params(
                non_default_params={"n": 3},
                optional_params={},
                model=self.model,
                drop_params=False,
            )

    def test_v2_n_greater_than_1_dropped_silently_with_drop_params_true(self):
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "n": 3},
            optional_params={},
            model=self.model,
            drop_params=True,
        )

        assert result == {"temperature": 0.7}
        assert "num_generations" not in result

    def test_v2_default_route_n_equal_1_end_to_end_does_not_raise(self):
        """End-to-end reproduction of the reported bug via the public entrypoint."""
        optional_params = litellm.get_optional_params(
            model=self.model,
            custom_llm_provider="cohere_chat",
            n=1,
        )

        assert "num_generations" not in optional_params

    def test_v2_default_route_n_greater_than_1_end_to_end_raises(self):
        with pytest.raises(UnsupportedParamsError):
            litellm.get_optional_params(
                model=self.model,
                custom_llm_provider="cohere_chat",
                n=3,
            )

    def test_v2_n_greater_than_1_dropped_silently_with_litellm_drop_params_true(self, monkeypatch):
        """v2 equivalent of the v1 global-flag test above -- same shared helper, same guarantee."""
        monkeypatch.setattr(litellm, "drop_params", True)

        result = self.config.map_openai_params(
            non_default_params={"n": 3},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert "num_generations" not in result


class TestMaybeDropUnsupportedNumGenerations:
    """Direct unit tests against the shared helper itself, independent of either
    config class, so every branch is exercised explicitly regardless of which
    (if any) higher-level integration path happens to reach it."""

    def test_value_none_is_a_noop(self):
        # No exception, no return value to check -- just must not raise.
        maybe_drop_unsupported_num_generations(value=None, drop_params=False)

    def test_value_one_is_a_noop(self):
        maybe_drop_unsupported_num_generations(value=1, drop_params=False)

    def test_value_greater_than_one_raises_by_default(self):
        with pytest.raises(UnsupportedParamsError) as exc_info:
            maybe_drop_unsupported_num_generations(value=5, drop_params=False)
        assert "n=5" in str(exc_info.value)

    def test_value_greater_than_one_with_drop_params_true_does_not_raise(self):
        # Must not raise; the only observable effect is a warning log.
        maybe_drop_unsupported_num_generations(value=5, drop_params=True)

    def test_value_greater_than_one_with_global_drop_params_true_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", True)
        maybe_drop_unsupported_num_generations(value=5, drop_params=False)

    def test_warning_logged_when_dropped(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            litellm.verbose_logger,
            "warning",
            lambda msg, *args: warnings.append(msg % args),
        )

        maybe_drop_unsupported_num_generations(value=5, drop_params=True)

        assert len(warnings) == 1
        assert "5" in warnings[0]
