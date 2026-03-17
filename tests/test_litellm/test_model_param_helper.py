from litellm.litellm_core_utils.model_param_helper import ModelParamHelper


def test_cached_relevant_logging_args_matches_dynamic():
    """Verify the cached frozenset matches the dynamically computed set."""
    cached = ModelParamHelper._relevant_logging_args
    dynamic = ModelParamHelper._get_relevant_args_to_use_for_logging()
    assert cached == dynamic
    assert isinstance(cached, frozenset)


def test_get_standard_logging_model_parameters_filters():
    """Verify model parameters are filtered to only supported keys."""
    params = {"temperature": 0.7, "messages": [{"role": "user"}], "max_tokens": 100}
    result = ModelParamHelper.get_standard_logging_model_parameters(params)
    assert "temperature" in result
    assert "max_tokens" in result
    assert "messages" not in result  # excluded prompt content


def test_get_standard_logging_model_parameters_excludes_prompt_content():
    """Verify all prompt content keys are excluded."""
    params = {
        "messages": [{"role": "user", "content": "hi"}],
        "prompt": "hello",
        "input": "test",
        "temperature": 0.5,
    }
    result = ModelParamHelper.get_standard_logging_model_parameters(params)
    assert "messages" not in result
    assert "prompt" not in result
    assert "input" not in result
    assert result == {"temperature": 0.5}
