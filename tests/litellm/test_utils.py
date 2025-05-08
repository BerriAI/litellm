import pytest

import litellm
from litellm import UnsupportedParamsError
from litellm.utils import (
    _get_optional_params_defaults,
    _get_optional_params_non_default_params,
    get_optional_params,
)


class TestGetOptionalParamsDefaults:
    def test_returns_dictionary(self):
        """Test that the function returns a dictionary."""
        result = _get_optional_params_defaults()
        assert isinstance(result, dict)

    def test_return_value_is_not_mutated(self):
        """Test that subsequent calls return independent copies of the default dictionary."""
        first_call = _get_optional_params_defaults()
        second_call = _get_optional_params_defaults()

        # Verify they're equal but not the same object
        assert first_call == second_call
        assert first_call is not second_call

        # Modify the first result and verify the second isn't affected
        first_call["temperature"] = 0.7
        assert second_call["temperature"] is None

    @pytest.mark.parametrize(
        "param_name, expected_value",
        [
            ("additional_drop_params", None),
            ("allowed_openai_params", None),
            ("api_version", None),
            ("audio", None),
            ("custom_llm_provider", ""),
            ("drop_params", None),
            ("extra_headers", None),
            ("frequency_penalty", None),
            ("function_call", None),
            ("functions", None),
            ("logit_bias", None),
            ("logprobs", None),
            ("max_completion_tokens", None),
            ("max_retries", None),
            ("max_tokens", None),
            ("messages", None),
            ("modalities", None),
            ("model", None),
            ("n", None),
            ("parallel_tool_calls", None),
            ("prediction", None),
            ("presence_penalty", None),
            ("reasoning_effort", None),
            ("response_format", None),
            ("seed", None),
            ("stop", None),
            ("stream", False),
            ("stream_options", None),
            ("temperature", None),
            ("thinking", None),
            ("tool_choice", None),
            ("tools", None),
            ("top_logprobs", None),
            ("top_p", None),
            ("user", None),
        ],
    )
    def test_individual_defaults(self, param_name, expected_value):
        """Test that each parameter has the expected default value."""
        defaults = _get_optional_params_defaults()
        assert param_name in defaults
        assert defaults[param_name] == expected_value

    def test_completeness(self):
        """Test that the function returns all expected parameters with no extras or missing items."""
        expected_params = {
            "additional_drop_params",
            "allowed_openai_params",
            "api_version",
            "audio",
            "custom_llm_provider",
            "drop_params",
            "extra_headers",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "max_retries",
            "max_tokens",
            "messages",
            "modalities",
            "model",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "reasoning_effort",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
        }

        actual_params = set(_get_optional_params_defaults().keys())

        # Check for extra parameters
        extra_params = actual_params - expected_params
        assert not extra_params, f"Unexpected parameters found: {extra_params}"

        # Check for missing parameters
        missing_params = expected_params - actual_params
        assert not missing_params, f"Expected parameters missing: {missing_params}"

    def test_custom_llm_provider_is_empty_string(self):
        """Specifically test that custom_llm_provider has empty string as default (not None)."""
        defaults = _get_optional_params_defaults()
        assert defaults["custom_llm_provider"] == ""
        assert defaults["custom_llm_provider"] is not None

    def test_stream_is_false(self):
        """Specifically test that stream has False as default (not None)."""
        defaults = _get_optional_params_defaults()
        assert not defaults["stream"]

    def test_all_others_are_none(self):
        """Test that all parameters except custom_llm_provider have None as default.

        This test may change in the future or no longer be required, but is included for now.
        """
        defaults = _get_optional_params_defaults()
        for key, value in defaults.items():
            if key in ["custom_llm_provider", "stream"]:
                continue
            assert value is None, f"Expected {key} to be None, but got {value}"


class TestGetOptionalParamsNonDefaultParams:
    @pytest.mark.parametrize(
        "passed_params, default_params, additional_drop_params, expected",
        [
            # no non-defaults, should return empty
            (
                {"model": "gpt-4", "api_version": "v1"},
                _get_optional_params_defaults(),
                None,
                {},
            ),
            # one non-default parameter not excluded
            (
                {
                    "temperature": 0.9,
                    "additional_drop_params": None,
                    "allowed_openai_params": "test",
                    "api_version": "v1",
                    "custom_llm_provider": "llamafile",
                    "drop_params": ["foo"],
                    "messages": ["bar"],
                    "model": "gpt-4",
                },
                _get_optional_params_defaults(),
                None,
                {"temperature": 0.9},
            ),
            # specifically exclude (drop) a parameter that is not default
            (
                {
                    "temperature": 0.9,
                    "additional_drop_params": None,
                    "allowed_openai_params": "test",
                    "api_version": "v1",
                    "custom_llm_provider": "llamafile",
                    "drop_params": ["foo"],
                    "messages": ["bar"],
                    "model": "gpt-4",
                },
                _get_optional_params_defaults(),
                ["temperature"],
                {},
            ),
            # non-default param dropped, not default param left alone
            (
                {"temperature": 0.9, "top_p": 0.95},
                _get_optional_params_defaults(),
                ["top_p"],
                {"temperature": 0.9},
            ),
        ],
    )
    def test_get_optional_params_non_default_params(
        self, passed_params, default_params, additional_drop_params, expected
    ):
        result = _get_optional_params_non_default_params(
            passed_params,
            default_params,
            additional_drop_params=additional_drop_params,
        )
        assert result == expected


class TestGetOptionalParms:
    def test_raises_on_unsupported_function_calling(self):
        original_flag = litellm.add_function_to_prompt

        try:
            litellm.add_function_to_prompt = False

            with pytest.raises(
                UnsupportedParamsError,
                match=r"^litellm.UnsupportedParamsError: Function calling is not supported by bad_provider.",
            ):
                get_optional_params(
                    model="qwerty",
                    custom_llm_provider="bad_provider",
                    functions="not_supported",
                )
        finally:
            litellm.add_function_to_prompt = original_flag

    def test_ollama_sets_json_and_removes_tool_choice(self):
        original_flag = litellm.add_function_to_prompt

        try:
            optional_params = get_optional_params(
                model="qwerty",
                custom_llm_provider="ollama",
                functions="my_function",
                tool_choice="auto",
            )

            assert optional_params["format"] == "json"
            assert litellm.add_function_to_prompt
            assert optional_params["functions_unsupported_model"] == "my_function"
        finally:
            litellm.add_function_to_prompt = original_flag

    @pytest.mark.parametrize(
        "tools, functions, function_call, expected_value",
        [
            ("foo", None, None, "foo"),
            (None, None, "baz", "baz"),
            ("foo", "bar", None, "foo"),
            ("foo", None, "baz", "foo"),
            (None, "bar", "baz", "bar"),
            ("foo", "bar", "baz", "foo"),
        ],
    )
    def test_supplying_tools_funcs_calls(
        self, tools, functions, function_call, expected_value
    ):
        original_flag = litellm.add_function_to_prompt
        try:
            optional_params = get_optional_params(
                model="qwerty",
                custom_llm_provider="ollama",
                tools=tools,
                functions=functions,
                function_call=function_call,
            )

            assert optional_params["format"] == "json"
            assert litellm.add_function_to_prompt
            assert optional_params["functions_unsupported_model"] == expected_value
        finally:
            litellm.add_function_to_prompt = original_flag
