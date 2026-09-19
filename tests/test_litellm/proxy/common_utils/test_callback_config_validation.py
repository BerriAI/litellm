from litellm.proxy.common_utils.callback_config_validation import (
    callback_config_error,
)


def test_callback_config_error_rejects_invalid_langfuse_environment():
    for callback in ["langfuse", "langfuse_otel"]:
        error = callback_config_error(callback, {"langfuse_environment": "Production"})
        assert error is not None and "langfuse_environment" in error

    assert callback_config_error("langfuse", {"langfuse_environment": "team-a-prod"}) is None
    assert callback_config_error("langfuse", {"langfuse_public_key": "pk"}) is None


def test_callback_config_error_rejects_an_unknown_langfuse_span_scope():
    for bad in ["everything", "LLM_ONLY", "llm-only", ""]:
        error = callback_config_error("langfuse_otel", {"langfuse_span_scope": bad})
        assert error is not None and "langfuse_span_scope" in error and "llm_only" in error

    assert callback_config_error("langfuse_otel", {"langfuse_span_scope": "llm_only"}) is None
    assert callback_config_error("langfuse_otel", {"langfuse_span_scope": "full"}) is None


def test_a_span_scope_on_a_callback_that_does_not_read_it_is_rejected():
    """Only langfuse_otel filters on the scope. Accepting it on the classic Langfuse
    callback or on an unrelated backend would store a setting that never takes
    effect, with the full tree still exported."""
    for callback_name in ["langfuse", "datadog", "otel", None]:
        error = callback_config_error(callback_name, {"langfuse_span_scope": "llm_only"})
        assert error is not None and "langfuse_span_scope" in error and "langfuse_otel" in error

    assert callback_config_error("langfuse", {"langfuse_environment": "team-a-prod"}) is None


def test_a_bad_span_scope_is_reported_even_when_the_environment_is_fine():
    error = callback_config_error(
        "langfuse_otel", {"langfuse_environment": "team-a-prod", "langfuse_span_scope": "everything"}
    )
    assert error is not None and "langfuse_span_scope" in error
