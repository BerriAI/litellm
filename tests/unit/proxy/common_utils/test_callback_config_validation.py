from collections.abc import Iterator
from typing import Final

import pytest

from litellm.proxy.common_utils.callback_config_validation import (
    callback_config_error,
    conflicting_capture_error,
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
    error = conflicting_span_scope_error(new_vars, stored)
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


class TestArizeOtlpProtocol:
    def test_protocol_on_a_non_arize_callback_is_rejected(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            for callback_name in ["langfuse", "langfuse_otel", "datadog", None]:
                error = callback_config_error(callback_name, {"arize_otlp_protocol": "grpc"})
                assert error is not None and "applies to the arize callback only" in error
        finally:
            is_otel_v2_enabled.cache_clear()

    def test_unknown_protocols_are_rejected(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            for bad in ["otlp_http", "HTTP/PROTOBUF", "http_json", "", "None"]:
                error = callback_config_error("arize", {"arize_otlp_protocol": bad})
                assert error is not None and "arize_otlp_protocol" in error and "http/protobuf" in error
        finally:
            is_otel_v2_enabled.cache_clear()

    def test_protocol_is_rejected_while_otel_v2_is_off(self, monkeypatch):
        monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            error = callback_config_error("arize", {"arize_otlp_protocol": "http/protobuf"})
            assert error is not None and "LITELLM_OTEL_V2" in error
        finally:
            is_otel_v2_enabled.cache_clear()

    def test_both_protocols_are_accepted_with_otel_v2_on(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            assert callback_config_error("arize", {"arize_otlp_protocol": "grpc"}) is None
            assert callback_config_error("arize", {"arize_otlp_protocol": "http/protobuf"}) is None
        finally:
            is_otel_v2_enabled.cache_clear()

    def test_protocol_is_not_a_family_credential(self):
        stored = [{"arize_api_key": "k1"}]
        assert cross_entry_family_error({"arize_otlp_protocol": "grpc"}, stored) is None
        assert cross_entry_family_error({"arize_otlp_protocol": "http/protobuf"}, stored) is None

    def test_protocol_on_a_key_logging_entry_is_validated(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            invalid_metadata = {
                "logging": [
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {"arize_otlp_protocol": "carrier-pigeon"},
                    }
                ]
            }
            error = logging_metadata_config_error(invalid_metadata)
            assert error is not None and "arize_otlp_protocol" in error

            valid_metadata = {
                "logging": [
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {"arize_otlp_protocol": "http/protobuf"},
                    }
                ]
            }
            assert logging_metadata_config_error(valid_metadata) is None
        finally:
            is_otel_v2_enabled.cache_clear()

    def test_failure_only_arize_callbacks_reject_the_protocol(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            error = callback_config_error("arize", {"arize_otlp_protocol": "http/protobuf"}, "failure")
            assert error is not None and "arize_otlp_protocol" in error and "failure" in error
            assert callback_config_error("arize", {"arize_otlp_protocol": "http/protobuf"}, "success") is None
            assert (
                callback_config_error("arize", {"arize_otlp_protocol": "http/protobuf"}, "success_and_failure") is None
            )
            assert callback_config_error("arize", {"arize_otlp_protocol": "http/protobuf"}) is None
        finally:
            is_otel_v2_enabled.cache_clear()

    def test_failure_only_key_logging_entry_rejects_the_protocol(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OTEL_V2", "true")
        from litellm.integrations.otel.model.config import is_otel_v2_enabled

        is_otel_v2_enabled.cache_clear()
        try:
            failure_metadata = {
                "logging": [
                    {
                        "callback_name": "arize",
                        "callback_type": "failure",
                        "callback_vars": {"arize_otlp_protocol": "http/protobuf"},
                    }
                ]
            }
            error = logging_metadata_config_error(failure_metadata)
            assert error is not None and "arize_otlp_protocol" in error

            success_metadata = {
                "logging": [
                    {
                        "callback_name": "arize",
                        "callback_type": "success",
                        "callback_vars": {"arize_otlp_protocol": "http/protobuf"},
                    }
                ]
            }
            assert logging_metadata_config_error(success_metadata) is None
        finally:
            is_otel_v2_enabled.cache_clear()


def test_arize_sampling_rates_are_not_family_credentials():
    """The rates choose what the Arize family exports, not where it sends, so an
    entry that repeats or adds a rate next to a stored Arize entry is not the
    credential-redirect shape cross_entry_family_error rejects."""
    stored = [{"arize_api_key": "k1", "arize_success_sampling_rate": "0.5"}]
    assert cross_entry_family_error({"arize_success_sampling_rate": "0.1"}, stored) is None
    assert cross_entry_family_error({"arize_error_sampling_rate": "0.5"}, stored) is None


@pytest.fixture
def otel_v2_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from litellm.integrations.otel.model.config import is_otel_v2_enabled

    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    yield
    is_otel_v2_enabled.cache_clear()


@pytest.mark.usefixtures("otel_v2_on")
@pytest.mark.parametrize("callback_name", ["langfuse_otel", "arize", "weave_otel", "newrelic"])
@pytest.mark.parametrize("value", ["no_content", "span_only"])
def test_capture_message_content_is_accepted_on_every_otel_v2_destination(callback_name: str, value: str) -> None:
    assert callback_config_error(callback_name, {"capture_message_content": value}) is None


def test_capture_message_content_is_rejected_while_otel_v2_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.otel.model.config import is_otel_v2_enabled

    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    is_otel_v2_enabled.cache_clear()
    try:
        error: Final = callback_config_error("langfuse_otel", {"capture_message_content": "no_content"})
    finally:
        is_otel_v2_enabled.cache_clear()
    assert error is not None and "capture_message_content" in error and "LITELLM_OTEL_V2" in error


@pytest.mark.parametrize("callback_name", ["langfuse", "datadog", "otel", "arize_phoenix", None])
def test_capture_message_content_is_rejected_where_it_would_never_take_effect(callback_name: str | None) -> None:
    error: Final = callback_config_error(callback_name, {"capture_message_content": "no_content"})
    assert error is not None and "capture_message_content" in error and "langfuse_otel" in error


@pytest.mark.usefixtures("otel_v2_on")
@pytest.mark.parametrize("value", ["full", "event_only", "span_and_event"])
def test_an_unsupported_capture_message_content_is_rejected_on_key_logging_metadata(value: str) -> None:
    error: Final = logging_metadata_config_error(
        {
            "logging": [
                {
                    "callback_name": "langfuse_otel",
                    "callback_type": "success",
                    "callback_vars": {"capture_message_content": value},
                }
            ]
        }
    )
    assert error is not None and "Invalid capture_message_content" in error


@pytest.mark.usefixtures("otel_v2_on")
def test_each_entry_keeps_its_own_capture_message_content() -> None:
    error: Final = logging_metadata_config_error(
        {
            "logging": [
                {
                    "callback_name": "langfuse_otel",
                    "callback_type": "success",
                    "callback_vars": {"capture_message_content": "span_only"},
                },
                {
                    "callback_name": "arize",
                    "callback_type": "success",
                    "callback_vars": {"capture_message_content": "no_content"},
                },
            ]
        }
    )
    assert error is None


@pytest.mark.usefixtures("otel_v2_on")
def test_a_failure_only_entry_rejects_capture_message_content() -> None:
    error: Final = callback_config_error("langfuse_otel", {"capture_message_content": "no_content"}, "failure")
    assert error is not None and "capture_message_content" in error and "success_and_failure" in error


@pytest.mark.parametrize(
    ("stored", "rejected"),
    [
        ([("langfuse_otel", {"capture_message_content": "no_content"})], True),
        ([("langfuse_otel", {"capture_message_content": "span_only"})], False),
        ([("langfuse_otel", {"langfuse_public_key": "pk"})], False),
        ([("newrelic", {"capture_message_content": "no_content"})], False),
    ],
)
def test_entries_for_one_backend_share_one_capture_message_content(
    stored: list[tuple[str, dict[str, str]]], rejected: bool
) -> None:
    """The backend's entries merge into one destination, so a second value would silently win or lose."""
    error: Final = conflicting_capture_error("langfuse_otel", {"capture_message_content": "span_only"}, stored)
    assert (error is not None) is rejected


@pytest.mark.usefixtures("otel_v2_on")
def test_key_logging_entries_for_one_backend_may_not_disagree_on_capture_message_content() -> None:
    def entry(callback_type: str, capture: str) -> dict[str, str | dict[str, str]]:
        return {
            "callback_name": "langfuse_otel",
            "callback_type": callback_type,
            "callback_vars": {"capture_message_content": capture},
        }

    error: Final = logging_metadata_config_error(
        {"logging": [entry("success", "no_content"), entry("success_and_failure", "span_only")]}
    )
    assert error is not None and "already set to 'no_content' by another langfuse_otel entry" in error
