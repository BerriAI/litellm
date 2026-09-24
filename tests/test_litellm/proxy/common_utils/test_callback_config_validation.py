import pytest

from litellm.proxy.common_utils.callback_config_validation import (
    callback_config_error,
    conflicting_span_scope_error,
    cross_entry_family_error,
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
    error = conflicting_span_scope_error("langfuse_otel", new_vars, stored)
    assert (error is not None) is rejected
    if rejected:
        assert "langfuse_span_scope" in error and stored[-1]["langfuse_span_scope"] in error


def test_key_logging_entries_may_not_disagree_on_the_span_scope():
    disagreeing = {
        "logging": [
            {
                "callback_name": "langfuse_otel",
                "callback_type": "success",
                "callback_vars": {"langfuse_span_scope": "full"},
            },
            {
                "callback_name": "langfuse_otel",
                "callback_type": "failure",
                "callback_vars": {"langfuse_span_scope": "llm_only"},
            },
        ]
    }
    error = logging_metadata_config_error(disagreeing)
    assert error is not None and "langfuse_span_scope" in error and "'full'" in error

    agreeing = {
        "logging": [
            {
                "callback_name": "langfuse_otel",
                "callback_type": "success",
                "callback_vars": {"langfuse_span_scope": "llm_only"},
            },
            {
                "callback_name": "langfuse_otel",
                "callback_type": "failure",
                "callback_vars": {"langfuse_span_scope": "llm_only"},
            },
            {"callback_name": "otel", "callback_type": "success", "callback_vars": {}},
        ]
    }
    assert logging_metadata_config_error(agreeing) is None


@pytest.mark.parametrize("callback_name", ["langfuse_otel", "arize", "weave_otel", "newrelic"])
def test_otel_span_scope_is_accepted_on_every_destination_backend(callback_name):
    for scope in ("full", "no_internal", "llm_only"):
        assert callback_config_error(callback_name, {"otel_span_scope": scope}) is None


@pytest.mark.parametrize("bad", ["everything", "NO_INTERNAL", "no", "exclude", ""])
def test_callback_config_error_rejects_an_unknown_otel_span_scope(bad):
    error = callback_config_error("arize", {"otel_span_scope": bad})
    assert error is not None and "otel_span_scope" in error and "no_internal" in error


@pytest.mark.parametrize("callback_name", ["langfuse", "datadog", "otel", "arize_phoenix", None])
def test_otel_span_scope_on_a_callback_that_has_no_destination_is_rejected(callback_name):
    error = callback_config_error(callback_name, {"otel_span_scope": "no_internal"})
    assert error is not None and "otel_span_scope" in error and "langfuse_otel" in error


def test_langfuse_span_scope_and_otel_span_scope_disagreeing_on_one_entry_is_rejected():
    error = callback_config_error(
        "langfuse_otel", {"langfuse_span_scope": "llm_only", "otel_span_scope": "no_internal"}
    )
    assert error is not None and "langfuse_span_scope" in error and "otel_span_scope" in error

    assert (
        callback_config_error(
            "langfuse_otel", {"langfuse_span_scope": "llm_only", "otel_span_scope": "llm_only"}
        )
        is None
    )
    assert callback_config_error("langfuse_otel", {"otel_span_scope": "no_internal"}) is None


def test_otel_span_scope_llm_only_is_accepted_on_newrelic():
    assert callback_config_error("newrelic", {"otel_span_scope": "llm_only"}) is None


def test_key_logging_entries_of_one_backend_may_not_disagree_on_span_scope():
    def entry(callback_name, callback_type, value):
        return {
            "callback_name": callback_name,
            "callback_type": callback_type,
            "callback_vars": {"otel_span_scope": value},
        }

    disagreeing = {"logging": [entry("arize", "success", "no_internal"), entry("arize", "failure", "full")]}
    error = logging_metadata_config_error(disagreeing)
    assert error is not None and "otel_span_scope" in error and "'no_internal'" in error

    two_backends = {"logging": [entry("arize", "success", "no_internal"), entry("langfuse_otel", "success", "full")]}
    assert logging_metadata_config_error(two_backends) is None


@pytest.mark.parametrize(
    "new_vars, stored, rejected",
    [
        ({"otel_span_scope": "no_internal"}, [{"otel_span_scope": "full"}], True),
        ({"otel_span_scope": "no_internal"}, [{"otel_span_scope": "no_internal"}], False),
        ({"otel_span_scope": "no_internal"}, [{"arize_api_key": "k"}], False),
        ({"arize_api_key": "k"}, [{"otel_span_scope": "no_internal"}], False),
        (None, [{"otel_span_scope": "no_internal"}], False),
    ],
)
def test_one_span_scope_value_per_backend(new_vars, stored, rejected):
    error = conflicting_span_scope_error("arize", new_vars, stored)
    assert (error is not None) is rejected


@pytest.mark.parametrize(
    "callback_name, new_vars, stored, rejected",
    [
        ("langfuse_otel", {"otel_span_scope": "llm_only"}, [{"langfuse_span_scope": "full"}], True),
        ("newrelic", {"otel_span_scope": "llm_only"}, [{"otel_span_scope": "full"}], True),
        ("langfuse_otel", {"otel_span_scope": "llm_only"}, [{"langfuse_span_scope": "llm_only"}], False),
        ("langfuse_otel", {"otel_span_scope": "llm_only"}, [{"otel_span_scope": "llm_only"}], False),
        ("langfuse_otel", {"langfuse_span_scope": "llm_only"}, [{"otel_span_scope": "llm_only"}], False),
    ],
)
def test_split_span_scope_aliases_conflict_across_entries(callback_name, new_vars, stored, rejected):
    error = conflicting_span_scope_error(callback_name, new_vars, stored)
    assert (error is not None) is rejected


@pytest.mark.parametrize("var", ["arize_success_sampling_rate", "arize_error_sampling_rate"])
@pytest.mark.parametrize("bad", ["1.5", "-0.1", "abc", "nan", "inf"])
def test_callback_config_error_rejects_out_of_range_arize_sampling_rate(var, bad):
    error = callback_config_error("arize", {var: bad})
    assert error is not None and var in error and repr(bad) in error


@pytest.mark.parametrize("var", ["arize_success_sampling_rate", "arize_error_sampling_rate"])
@pytest.mark.parametrize("good", ["0", "1", "0.25", "", "None"])
def test_callback_config_error_accepts_in_range_arize_sampling_rate(var, good):
    assert callback_config_error("arize", {var: good}) is None


def test_arize_sampling_rate_rejected_on_non_arize_callback():
    error = callback_config_error("langfuse", {"arize_success_sampling_rate": "0.5"})
    assert error is not None
    assert "applies to the arize callback only" in error
    assert callback_config_error("arize", {"arize_success_sampling_rate": "0.5"}) is None


def test_arize_sampling_rates_are_not_family_credentials():
    """The rates choose what the Arize family exports, not where it sends, so an
    entry that repeats or adds a rate next to a stored Arize entry is not the
    credential-redirect shape cross_entry_family_error rejects."""
    stored = [{"arize_api_key": "k1", "arize_success_sampling_rate": "0.5"}]
    assert cross_entry_family_error({"arize_success_sampling_rate": "0.1"}, stored) is None
    assert cross_entry_family_error({"arize_error_sampling_rate": "0.5"}, stored) is None
