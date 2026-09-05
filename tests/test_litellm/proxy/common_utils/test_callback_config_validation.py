from litellm.proxy.common_utils.callback_config_validation import (
    callback_config_error,
)


def test_callback_config_error_rejects_invalid_langfuse_environment():
    for callback in ["langfuse", "langfuse_otel"]:
        error = callback_config_error(callback, {"langfuse_environment": "Production"})
        assert error is not None and "langfuse_environment" in error

    assert callback_config_error("langfuse", {"langfuse_environment": "team-a-prod"}) is None
    assert callback_config_error("langfuse", {"langfuse_public_key": "pk"}) is None
