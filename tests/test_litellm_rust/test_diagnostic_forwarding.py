import logging
from collections.abc import Mapping
from io import StringIO
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm._logging import session_id_var, set_session_id, set_trace_id, trace_id_var
from litellm import diagnostics
from litellm.rust_bridge import _native
from litellm.rust_bridge.catalog import LoggerContext, decision
from litellm.rust_bridge.configuration import Decision
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension
_BODY: Final = TypeAdapter(dict[str, JsonValue])


def test_python_and_native_diagnostics_reach_one_configured_posthog_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    logger: Final = logging.Logger("LiteLLM.forwarding-contract", logging.INFO)
    output: Final = StringIO()
    logger.addHandler(logging.StreamHandler(output))
    session_token: Final = set_session_id("test-session")
    trace_token: Final = set_trace_id("test-trace")
    try:
        with recording_service() as collector, recording_service() as provider:
            collector.default_response = ResponseSpec(body={"status": 1})
            provider_response: Final[Mapping[str, JsonValue]] = {
                "pages": [{"index": 0, "markdown": "native", "images": [], "dimensions": None}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
            }
            provider.default_response = ResponseSpec(body=provider_response)
            try:
                for _ in range(2):
                    assert diagnostics.configure(
                        {
                            "enabled": True,
                            "policy": {"target_prefixes": [logger.name, "litellm_inference_ocr"]},
                            "destinations": [
                                {
                                    "transport": "posthog",
                                    "name": "events",
                                    "api_key": "phc_test",
                                    "endpoint": collector.base_url,
                                }
                            ],
                        },
                        loggers=(logger,),
                    )
                assert decision(LoggerContext()) is Decision.PYTHON
                logger.warning("Authorization: Bearer %s", "abcdefghijklmnop", extra={"api_key": "secret123"})
                kwargs: Final[Mapping[str, object]] = {
                    "model": "mistral/mistral-ocr-latest",
                    "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                    "api_key": "test-key",
                    "api_base": provider.base_url,
                }
                request: Final = LiteLLMOcrRequest(
                    "mistral/mistral-ocr-latest",
                    {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                    "test-key",
                    provider.base_url,
                    None,
                    "mistral",
                    None,
                    kwargs,
                )
                response: Final = _native.ocr(request, (), dict(kwargs))
                assert response.pages[0].markdown == "native"
                assert diagnostics.force_flush()
            finally:
                assert diagnostics.shutdown()
            body: Final = _BODY.validate_json(collector.requests[0].raw_body)
            events: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(body["batch"])
            property_records: Final = tuple(_BODY.validate_python(event["properties"]) for event in events)
            field_records: Final = tuple(_BODY.validate_python(properties["fields"]) for properties in property_records)
            assert len(events) == 2
            assert all(event["event"] == "litellm diagnostic" for event in events)
            assert all(properties["$process_person_profile"] is False for properties in property_records)
            assert all(
                (fields["session_id"], fields["trace_id"]) == ("test-session", "test-trace") for fields in field_records
            )
            python_event: Final = next(
                properties for properties in property_records if properties["message"] != "span closed"
            )
            assert python_event["message"] == "Authorization: REDACTED"
            native_fields: Final = next(
                fields for fields in field_records if fields.get("span_name") == "litellm.route"
            )
            assert (native_fields["route"], native_fields["outcome"]) == ("ocr", "success")
            assert b"abcdefghijklmnop" not in collector.requests[0].raw_body
            assert b"secret123" not in collector.requests[0].raw_body
            assert collector.requests[0].path == "/batch/"
            assert provider.requests[0].path == "/v1/ocr"
    finally:
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)
    assert output.getvalue() == "Authorization: Bearer abcdefghijklmnop\n"
    logger.error("after shutdown")
    assert not _native.NativeDiagnosticLogger().active()
    assert output.getvalue().endswith("after shutdown\n")


def test_named_destinations_apply_independent_sampling_without_changing_python_output() -> None:
    logger: Final = logging.Logger("LiteLLM.destinations", logging.INFO)
    output: Final = StringIO()
    logger.addHandler(logging.StreamHandler(output))
    with recording_service() as all_logs, recording_service() as errors:
        all_logs.default_response = ResponseSpec(body={"status": 1})
        errors.default_response = ResponseSpec(body={"status": 1})
        try:
            assert diagnostics.configure(
                {
                    "enabled": True,
                    "destinations": [
                        {"transport": "posthog", "name": "all", "api_key": "phc_test", "endpoint": all_logs.base_url},
                        {
                            "transport": "posthog",
                            "name": "errors",
                            "api_key": "phc_test",
                            "endpoint": errors.base_url,
                            "policy": {"sample_rate": 0},
                        },
                    ],
                },
                loggers=(logger,),
            )
            logger.warning("routine")
            logger.error("failure")
            assert diagnostics.force_flush()
        finally:
            assert diagnostics.shutdown()
        all_body: Final = _BODY.validate_json(all_logs.requests[0].raw_body)
        errors_body: Final = _BODY.validate_json(errors.requests[0].raw_body)
        all_events: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(all_body["batch"])
        error_events: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(errors_body["batch"])
        assert tuple(_BODY.validate_python(event["properties"])["message"] for event in all_events) == (
            "routine",
            "failure",
        )
        assert tuple(_BODY.validate_python(event["properties"])["message"] for event in error_events) == ("failure",)
    assert output.getvalue() == "routine\nfailure\n"
    assert diagnostics.shutdown()
