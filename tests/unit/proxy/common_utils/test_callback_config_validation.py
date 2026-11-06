from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.proxy.common_utils.callback_config_validation import (
    callback_backend_name,
    callback_config_error,
    callback_definition,
    callback_definitions_error,
    conflicting_callback_var_error,
    conflicting_span_scope_error,
    cross_entry_family_error,
    logging_entry_keys,
    logging_metadata_config_error,
    reference_conflicts,
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


@pytest.mark.usefixtures("otel_v2_on")
@pytest.mark.parametrize("callback_name", ["langfuse_otel", "arize", "weave_otel", "newrelic"])
@pytest.mark.parametrize("value", ["redis,postgres", "none", " REDIS , postgresql "])
def test_excluded_services_is_accepted_on_every_otel_v2_destination(callback_name: str, value: str) -> None:
    assert callback_config_error(callback_name, {"excluded_services": value}) is None


def test_capture_message_content_is_rejected_while_otel_v2_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.otel.model.config import is_otel_v2_enabled

    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    is_otel_v2_enabled.cache_clear()
    try:
        error: Final = callback_config_error("langfuse_otel", {"capture_message_content": "no_content"})
    finally:
        is_otel_v2_enabled.cache_clear()
    assert error is not None and "capture_message_content" in error and "LITELLM_OTEL_V2" in error


def test_excluded_services_is_rejected_while_otel_v2_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.otel.model.config import is_otel_v2_enabled

    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    is_otel_v2_enabled.cache_clear()
    try:
        error: Final = callback_config_error("langfuse_otel", {"excluded_services": "redis"})
    finally:
        is_otel_v2_enabled.cache_clear()
    assert error is not None and "excluded_services" in error and "LITELLM_OTEL_V2" in error


@pytest.mark.parametrize("callback_name", ["langfuse", "datadog", "otel", "arize_phoenix", None])
def test_capture_message_content_is_rejected_where_it_would_never_take_effect(callback_name: str | None) -> None:
    error: Final = callback_config_error(callback_name, {"capture_message_content": "no_content"})
    assert error is not None and "capture_message_content" in error and "langfuse_otel" in error


@pytest.mark.usefixtures("otel_v2_on")
@pytest.mark.parametrize("callback_name", ["langfuse", "datadog", "otel", "arize_phoenix", None])
def test_excluded_services_is_rejected_where_it_would_never_take_effect(callback_name: str | None) -> None:
    error: Final = callback_config_error(callback_name, {"excluded_services": "redis"})
    assert error is not None and "excluded_services" in error and "langfuse_otel" in error


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
@pytest.mark.parametrize("callback_var", ["capture_message_content", "excluded_services"])
def test_key_logging_metadata_rejects_non_string_destination_callback_vars(callback_var: str) -> None:
    error: Final = logging_metadata_config_error(
        {
            "logging": [
                {
                    "callback_name": "newrelic",
                    "callback_type": "success_and_failure",
                    "callback_vars": {callback_var: None},
                }
            ]
        }
    )
    assert error == f"{callback_var} must be a string, got NoneType"


@pytest.mark.usefixtures("otel_v2_on")
def test_key_logging_metadata_with_non_string_callback_var_key_does_not_raise() -> None:
    error: Final = logging_metadata_config_error(
        {
            "logging": [
                {
                    "callback_name": "langfuse_otel",
                    "callback_vars": {1: "x", "excluded_services": "redis"},
                }
            ]
        }
    )
    assert error is None


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
    error: Final = conflicting_callback_var_error(
        "langfuse_otel", "capture_message_content", {"capture_message_content": "span_only"}, stored
    )
    assert (error is not None) is rejected


@pytest.mark.parametrize(
    ("stored", "rejected"),
    [
        ([("langfuse_otel", {"excluded_services": "none"})], True),
        ([("langfuse_otel", {"excluded_services": "redis"})], False),
        ([("langfuse_otel", {"langfuse_public_key": "pk"})], False),
        ([("newrelic", {"excluded_services": "none"})], False),
    ],
)
def test_entries_for_one_backend_share_one_excluded_services_value(
    stored: list[tuple[str, dict[str, str]]], rejected: bool
) -> None:
    error: Final = conflicting_callback_var_error(
        "langfuse_otel", "excluded_services", {"excluded_services": "redis"}, stored
    )
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


@pytest.mark.usefixtures("otel_v2_on")
def test_key_logging_entries_for_one_backend_may_not_disagree_on_excluded_services() -> None:
    def entry(callback_vars: dict[str, str]) -> dict[str, str | dict[str, str]]:
        return {"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": callback_vars}

    error: Final = logging_metadata_config_error(
        {"logging": [entry({"excluded_services": "redis"}), entry({"excluded_services": "none"})]}
    )
    assert error is not None and "already set to 'redis' by another langfuse_otel entry" in error


@pytest.mark.usefixtures("otel_v2_on")
@pytest.mark.parametrize(
    ("stored", "incoming", "conflicts"),
    [
        ("redis,postgres", "postgres,redis", False),
        ("redis", "REDIS", False),
        ("postgres", "postgresql", False),
        ("none", "NONE", False),
        ("redis", "postgres", True),
        ("none", "redis", True),
    ],
)
def test_key_logging_entries_compare_excluded_services_by_parsed_value(
    stored: str, incoming: str, conflicts: bool
) -> None:
    def entry(callback_type: str, excluded_services: str) -> dict[str, str | dict[str, str]]:
        return {
            "callback_name": "newrelic",
            "callback_type": callback_type,
            "callback_vars": {"newrelic_api_key": "license-key", "excluded_services": excluded_services},
        }

    error: Final = logging_metadata_config_error(
        {"logging": [entry("success", stored), entry("success_and_failure", incoming)]}
    )
    assert (error is not None) is conflicts
    if conflicts:
        assert error is not None and f"already set to {stored!r}" in error


@pytest.mark.usefixtures("otel_v2_on")
def test_a_failure_only_entry_rejects_excluded_services() -> None:
    error: Final = callback_config_error("langfuse_otel", {"excluded_services": "redis"}, "failure")
    assert error is not None and "excluded_services" in error and "success_and_failure" in error


_ARIZE_DEFINITION: Final = {
    "callback_type": "arize",
    "arize_api_key": "resolved-space-a-key",
    "arize_space_id": "space-a",
}


@pytest.fixture
def arize_space_a(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callback_settings", {"arize_space_a": dict(_ARIZE_DEFINITION)})


@pytest.mark.parametrize(
    ("callback_settings", "named_fields"),
    [
        ({"arize": {"callback_type": "arize", "arize_api_key": "k"}}, ("callback_settings.arize", "named after")),
        (
            {"space_a": {"callback_type": "not_a_callback"}},
            ("callback_settings.space_a.callback_type", "not_a_callback"),
        ),
        ({"space_a": {"callback_type": "arize", "arize_api_key": None}}, ("callback_settings.space_a.arize_api_key",)),
        ({"space_a": {"callback_type": "arize", "arize_api_key": ""}}, ("callback_settings.space_a.arize_api_key",)),
        (
            {"space_a": {"callback_type": "arize", "arize_space_id": "os.environ/UNSET"}},
            ("callback_settings.space_a.arize_space_id", "secret manager"),
        ),
        ({"space_a": {"callback_type": "arize", "arize_api_kye": "k"}}, ("callback_settings.space_a", "arize_api_kye")),
        (
            {"space_a": {"callback_type": "arize", "arize_otlp_protocol": "carrier-pigeon"}},
            ("callback_settings.space_a", "arize_otlp_protocol"),
        ),
    ],
)
def test_a_definition_a_key_could_never_use_fails_config_load(
    callback_settings: dict[str, object], named_fields: tuple[str, ...]
) -> None:
    error: Final = callback_definitions_error(callback_settings)
    assert error is not None
    for fragment in named_fields:
        assert fragment in error


def test_a_definition_error_names_the_field_and_never_its_value() -> None:
    error: Final = callback_definitions_error(
        {
            "space_a": {
                "callback_type": "arize",
                "arize_otlp_protocol": "carrier-pigeon",
                "arize_api_key": "sk-super-secret",
            }
        }
    )
    assert error is not None and "arize_otlp_protocol" in error
    assert "sk-super-secret" not in error


@pytest.mark.usefixtures("otel_v2_on")
def test_callback_settings_without_a_referenceable_definition_pass_config_load() -> None:
    assert (
        callback_definitions_error(
            {
                "otel": {"message_logging": False, "excluded_services": ["postgres"]},
                "my_api": {"callback_type": "generic_api", "url": "os.environ/UNSET"},
                "space_a": {
                    "callback_type": "arize",
                    "arize_api_key": "k",
                    "arize_space_id": 12345,
                    "excluded_services": "postgres,redis",
                },
            }
        )
        is None
    )


@pytest.mark.usefixtures("arize_space_a")
def test_a_reference_resolves_to_its_definition_and_a_plain_name_does_not() -> None:
    definition: Final = callback_definition("arize_space_a")
    assert definition is not None
    assert (definition.backend, dict(definition.callback_vars)) == (
        "arize",
        {"arize_api_key": "resolved-space-a-key", "arize_space_id": "space-a"},
    )
    assert callback_definition("arize") is None
    assert callback_definition(None) is None
    assert (callback_backend_name("arize_space_a"), callback_backend_name("langfuse")) == ("arize", "langfuse")


@pytest.mark.usefixtures("arize_space_a")
def test_a_reference_entry_may_not_carry_callback_vars() -> None:
    error: Final = logging_metadata_config_error(
        {
            "logging": [
                {
                    "callback_name": "arize_space_a",
                    "callback_type": "success",
                    "callback_vars": {"arize_space_id": "attacker-space"},
                }
            ]
        }
    )
    assert error is not None and "arize_space_a" in error and "callback_vars" in error


@pytest.mark.usefixtures("arize_space_a")
def test_a_reference_is_the_only_entry_for_its_backend() -> None:
    sibling: Final = {"callback_name": "arize", "callback_type": "success", "callback_vars": {"arize_space_id": "x"}}
    reference: Final = {"callback_name": "arize_space_a", "callback_type": "success"}
    error: Final = logging_metadata_config_error({"logging": [reference, sibling]})
    assert error is not None and "arize_space_a" in error and "arize" in error
    assert logging_metadata_config_error({"logging": [sibling]}) is None


@pytest.mark.usefixtures("arize_space_a")
def test_a_reference_sits_next_to_entries_for_other_backends() -> None:
    assert (
        logging_metadata_config_error(
            {
                "logging": [
                    {"callback_name": "arize_space_a", "callback_type": "success"},
                    {"callback_name": "arize_space_a", "callback_type": "failure", "callback_vars": {}},
                    {"callback_name": "langfuse", "callback_vars": {"langfuse_public_key": "pk"}},
                ]
            }
        )
        is not None
    ), "two references to one definition still share its backend"
    assert (
        logging_metadata_config_error(
            {
                "logging": [
                    {"callback_name": "arize_space_a", "callback_type": "success"},
                    {"callback_name": "langfuse", "callback_vars": {"langfuse_public_key": "pk"}},
                ]
            }
        )
        is None
    )


def test_logging_entry_keys_change_with_every_field_a_caller_could_edit() -> None:
    entry: Final = {"callback_name": "arize_space_a", "callback_type": "success", "callback_vars": {"b": "2", "a": "1"}}
    reordered: Final = {
        "callback_type": "success",
        "callback_vars": {"a": "1", "b": "2"},
        "callback_name": "arize_space_a",
    }
    assert logging_entry_keys([entry]) == logging_entry_keys([reordered])
    assert logging_entry_keys([entry]) != logging_entry_keys([{**entry, "callback_type": "failure"}])
    assert logging_entry_keys([entry]) != logging_entry_keys([{**entry, "callback_vars": {"a": "1"}}])
    assert logging_entry_keys([entry]) != logging_entry_keys([{**entry, "callback_name": "arize"}])
    assert logging_entry_keys([entry]) != logging_entry_keys([entry, entry])
    assert logging_entry_keys(None) == logging_entry_keys("not-a-list") == ()


@pytest.mark.parametrize("callback_type", [[], "bogus", 1, {"success": True}])
def test_a_logging_entry_with_a_malformed_callback_type_is_rejected_at_write_time(callback_type: object) -> None:
    """Stored as sent, such an entry fails model validation at request time and its logger silently never runs."""
    error: Final = logging_metadata_config_error(
        {"logging": [{"callback_name": "langfuse", "callback_type": callback_type, "callback_vars": {}}]}
    )
    assert error is not None and "callback_type" in error
    reference_error: Final = logging_metadata_config_error(
        {"logging": [{"callback_name": "arize_space_a", "callback_type": callback_type}]}
    )
    assert reference_error is not None and "callback_type" in reference_error


@pytest.mark.parametrize("callback_type", [None, "success", "failure", "success_and_failure"])
def test_a_logging_entry_with_a_valid_or_absent_callback_type_passes(callback_type: str | None) -> None:
    entry: Final = {"callback_name": "langfuse", "callback_vars": {"langfuse_public_key": "pk"}}
    with_type: Final = {**entry, "callback_type": callback_type}
    assert logging_metadata_config_error({"logging": [with_type]}) is None
    assert logging_metadata_config_error({"logging": [entry]}) is None


def test_logging_entry_keys_keep_a_malformed_callback_type_distinct_from_a_missing_one() -> None:
    entry: Final = {"callback_name": "arize_space_a", "callback_vars": {}}
    assert logging_entry_keys([entry]) == logging_entry_keys([{**entry, "callback_type": None}])
    assert logging_entry_keys([entry]) != logging_entry_keys([{**entry, "callback_type": []}])
    assert logging_entry_keys([entry]) != logging_entry_keys([{**entry, "callback_type": "success"}])


@pytest.fixture
def langfuse_team_a(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "callback_settings",
        {
            "langfuse_team_a": {
                "callback_type": "langfuse_otel",
                "langfuse_public_key": "pk-team-a",
                "langfuse_secret_key": "sk-team-a",
                "langfuse_host": "https://langfuse.team-a.example",
            }
        },
    )


@pytest.mark.usefixtures("langfuse_team_a")
def test_a_reference_may_not_sit_next_to_an_entry_of_the_same_credential_family() -> None:
    """The classic langfuse callback reads langfuse_* credentials from the merged map, so a sibling with its own
    host next to a langfuse_otel reference would receive the definition's keys."""
    reference: Final = {"callback_name": "langfuse_team_a", "callback_type": "success"}
    sibling: Final = {
        "callback_name": "langfuse",
        "callback_type": "success",
        "callback_vars": {"langfuse_host": "https://attacker.example"},
    }
    other_family: Final = {
        "callback_name": "datadog",
        "callback_type": "success",
        "callback_vars": {"dd_api_key": "dd"},
    }
    error: Final = logging_metadata_config_error({"logging": [reference, sibling]})
    assert error is not None and "langfuse_team_a" in error
    assert logging_metadata_config_error({"logging": [sibling, reference]}) is not None
    assert logging_metadata_config_error({"logging": [reference, other_family]}) is None
    assert reference_conflicts([reference, sibling]) == frozenset({"langfuse_team_a"})
    assert reference_conflicts([reference, other_family]) == frozenset()


@pytest.mark.usefixtures("otel_v2_on")
def test_a_failure_only_reference_is_held_to_its_definition_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "callback_settings",
        {
            "arize_space_a": {
                "callback_type": "arize",
                "arize_api_key": "k",
                "arize_space_id": "s",
                "excluded_services": "postgres,redis",
            }
        },
    )
    failure_only: Final = {"callback_name": "arize_space_a", "callback_type": "failure"}
    error: Final = logging_metadata_config_error({"logging": [failure_only]})
    assert error is not None and "excluded_services" in error and "callback_type" in error
    assert logging_metadata_config_error({"logging": [{**failure_only, "callback_type": "success"}]}) is None


@pytest.mark.parametrize("callback_type", [None, 5, ["arize"]])
def test_a_definition_whose_callback_type_is_not_a_name_fails_config_load(callback_type: object) -> None:
    error: Final = callback_definitions_error({"space_a": {"callback_type": callback_type, "arize_api_key": "k"}})
    assert error is not None and "callback_settings.space_a.callback_type" in error


def test_an_ordinary_logging_entry_without_callback_vars_is_rejected_at_write_time() -> None:
    """Stored as sent, the entry fails model validation at request time and its logger silently never runs."""
    error: Final = logging_metadata_config_error(
        {"logging": [{"callback_name": "datadog", "callback_type": "success"}]}
    )
    assert error is not None and "datadog" in error and "callback_vars" in error
    assert logging_metadata_config_error({"logging": [{"callback_name": "datadog", "callback_vars": {}}]}) is None


def test_logging_entry_keys_keep_missing_callback_vars_distinct_from_empty_ones() -> None:
    entry: Final = {"callback_name": "datadog", "callback_type": "success"}
    assert logging_entry_keys([entry]) != logging_entry_keys([{**entry, "callback_vars": {}}])
    assert logging_entry_keys([{**entry, "callback_vars": "x"}]) != logging_entry_keys([{**entry, "callback_vars": {}}])
