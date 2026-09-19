import pytest

from litellm.proxy.common_utils.callback_config_validation import (
    callback_config_error,
    conflicting_span_scope_error,
    logging_metadata_config_error,
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


@pytest.mark.parametrize(
    "new_vars, stored, rejected",
    [
        ({"langfuse_span_scope": "llm_only"}, [{"langfuse_span_scope": "full"}], True),
        ({"langfuse_span_scope": "full"}, [{"langfuse_public_key": "pk"}, {"langfuse_span_scope": "llm_only"}], True),
        ({"langfuse_span_scope": "llm_only"}, [{"langfuse_span_scope": "llm_only"}], False),
        ({"langfuse_span_scope": "llm_only"}, [{"langfuse_public_key": "pk"}], False),
        ({"langfuse_span_scope": "llm_only"}, [], False),
        ({"langfuse_public_key": "pk"}, [{"langfuse_span_scope": "llm_only"}], False),
        (None, [{"langfuse_span_scope": "llm_only"}], False),
    ],
)
def test_one_span_scope_per_team(new_vars, stored, rejected):
    """The entries flatten last-wins, so a second scope would export whichever entry
    was stored last. An entry that names no scope leaves the stored one in charge."""
    error = conflicting_span_scope_error(new_vars, stored)
    assert (error is not None) is rejected
    if rejected:
        assert "langfuse_span_scope" in error and stored[-1]["langfuse_span_scope"] in error


def test_key_logging_entries_may_not_disagree_on_the_span_scope():
    disagreeing = {
        "logging": [
            {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": {"langfuse_span_scope": "full"}},
            {"callback_name": "langfuse_otel", "callback_type": "failure", "callback_vars": {"langfuse_span_scope": "llm_only"}},
        ]
    }
    error = logging_metadata_config_error(disagreeing)
    assert error is not None and "langfuse_span_scope" in error and "'full'" in error

    agreeing = {
        "logging": [
            {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": {"langfuse_span_scope": "llm_only"}},
            {"callback_name": "langfuse_otel", "callback_type": "failure", "callback_vars": {"langfuse_span_scope": "llm_only"}},
            {"callback_name": "otel", "callback_type": "success", "callback_vars": {}},
        ]
    }
    assert logging_metadata_config_error(agreeing) is None
