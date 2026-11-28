"""Per-field byte caps: no string bound for a Langfuse span attribute may exceed the cap.

The wiring tests are the regression guard for the ingestion guarantee: a multi-megabyte
message must reach the exported span attributes cut at ``LANGFUSE_MAX_FIELD_BYTES`` (default
32 KiB) with a ``...(truncated)`` marker, never whole.
"""

import json
from datetime import datetime, timezone
from typing import Final

import pytest
from langfuse import LangfuseOtelSpanAttributes as A

import litellm
import litellm.integrations.langfuse.langfuse as langfuse_module
from litellm.integrations.langfuse import langfuse_sdk
from litellm.integrations.langfuse.field_cap import (
    _TRUNCATION_SUFFIX,
    DEFAULT_MAX_FIELD_BYTES,
    MAX_FIELD_BYTES_METADATA_KEY,
    cap_payload,
    resolve_max_field_bytes,
)
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
    langfuse_client_init,
)


def test_default_cap_is_32_kib():
    assert DEFAULT_MAX_FIELD_BYTES == 32 * 1024


def test_resolve_prefers_request_metadata_over_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MAX_FIELD_BYTES", "4096")
    assert resolve_max_field_bytes(1024) == 1024


def test_resolve_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MAX_FIELD_BYTES", "4096")
    assert resolve_max_field_bytes() == 4096


def test_resolve_defaults_when_neither_set(monkeypatch):
    monkeypatch.delenv("LANGFUSE_MAX_FIELD_BYTES", raising=False)
    assert resolve_max_field_bytes() == DEFAULT_MAX_FIELD_BYTES


def test_resolve_env_string_zero_disables_capping(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MAX_FIELD_BYTES", "0")
    assert resolve_max_field_bytes() == 0


@pytest.mark.parametrize("raw", ["abc", "", True, [], 3.5, object()])
def test_resolve_invalid_values_fall_back_to_default(monkeypatch, raw):
    monkeypatch.delenv("LANGFUSE_MAX_FIELD_BYTES", raising=False)
    assert resolve_max_field_bytes(raw) == DEFAULT_MAX_FIELD_BYTES


def test_short_string_passes_through_unchanged():
    assert cap_payload("hello", DEFAULT_MAX_FIELD_BYTES) == "hello"


def test_long_ascii_string_is_cut_at_the_cap_with_marker():
    assert cap_payload("a" * 100, 32) == "a" * 16 + _TRUNCATION_SUFFIX


def test_multibyte_string_never_splits_a_rune():
    capped: Final = cap_payload("漢" * 100, 32)
    assert capped == "漢" * 2 + _TRUNCATION_SUFFIX


def test_capped_string_serialized_width_stays_within_cap():
    capped: Final = cap_payload("x" * 10_000, 256)
    assert len(json.dumps(capped)) <= 256


def test_cjk_string_wire_width_stays_within_cap_despite_uXXXX_escaping():
    capped: Final = cap_payload("漢" * 100_000, DEFAULT_MAX_FIELD_BYTES)
    assert capped == "漢" * 5_458 + _TRUNCATION_SUFFIX
    assert len(json.dumps(capped)) <= DEFAULT_MAX_FIELD_BYTES


def test_astral_code_points_count_double_width():
    capped: Final = cap_payload("🙂" * 1_000, 32)
    assert capped == "🙂" + _TRUNCATION_SUFFIX
    assert len(json.dumps(capped)) <= 32


def test_nested_containers_are_capped_recursively():
    payload: Final = {
        "messages": [{"role": "user", "content": "x" * 100}],
        "tools": ("y" * 100,),
        "count": 5,
        "nothing": None,
    }
    capped: Final = cap_payload(payload, 64)
    assert capped["messages"][0]["content"] == "x" * 48 + _TRUNCATION_SUFFIX
    assert capped["tools"][0] == "y" * 48 + _TRUNCATION_SUFFIX
    assert capped["count"] == 5
    assert capped["nothing"] is None


def test_cap_below_marker_size_leaves_only_the_marker():
    assert cap_payload("x" * 100, 16) == _TRUNCATION_SUFFIX


def test_cap_below_marker_width_returns_empty_string():
    assert cap_payload("x" * 100, 15) == ""
    assert cap_payload("x" * 100, 1) == ""


def test_circular_reference_resolves_to_the_safe_dumps_marker():
    payload: Final[dict[str, object]] = {"messages": []}
    payload["messages"].append({"role": "user", "content": payload})
    capped: Final = cap_payload(payload, DEFAULT_MAX_FIELD_BYTES)
    assert capped["messages"][0]["content"] == "CircularReference Detected"


def test_over_deep_nesting_resolves_to_the_safe_dumps_marker():
    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    payload: Final[dict[str, object]] = {"leaf": "x" * 10}
    for _ in range(200):
        payload = {"nested": payload}
    capped: Final = cap_payload(payload, DEFAULT_MAX_FIELD_BYTES)
    assert "MaxDepthExceeded" in repr(capped)
    assert safe_dumps(capped) == safe_dumps(payload)


def test_non_container_objects_pass_through():
    sentinel: Final = object()
    assert cap_payload(sentinel, 16) is sentinel


def test_zero_limit_returns_value_unchanged():
    payload: Final = {"a": "z" * 100}
    assert cap_payload(payload, 0) is payload


class _CapturedGeneration:
    id: Final = "f" * 16

    def __init__(self, attributes):
        self.attributes = attributes

    def end(self, end_time=None):
        return None


def _mock_mode(monkeypatch):
    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-field-cap")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-field-cap")
    monkeypatch.delenv("LANGFUSE_MAX_FIELD_BYTES", raising=False)
    langfuse_client_init.cache_clear()


def _log_with_capture(monkeypatch, metadata, content_size):
    captured = {}
    original_start_generation: Final = langfuse_sdk.start_generation

    def fake_start_generation(**kwargs):
        captured["attributes"] = dict(kwargs["attributes"])
        return original_start_generation(**kwargs)

    monkeypatch.setattr(langfuse_sdk, "start_generation", fake_start_generation)
    monkeypatch.setattr(langfuse_module, "log_provider_specific_information_as_span", lambda **kwargs: None)
    huge: Final = "x" * content_size
    now: Final = datetime.now(timezone.utc)
    logger = LangfusePromptManagement()
    try:
        logger.log_event_on_langfuse(
            kwargs={
                "litellm_call_id": "call-field-cap",
                "call_type": "completion",
                "model": "gpt-5.4",
                "litellm_params": {"metadata": metadata},
                "messages": [{"role": "user", "content": huge}],
                "optional_params": {"temperature": 0.5},
            },
            response_obj=litellm.ModelResponse(
                choices=[{"message": {"role": "assistant", "content": "y" * content_size}}]
            ),
            start_time=now,
            end_time=now,
        )
    finally:
        langfuse_sdk.release_langfuse_tracing(logger.tracing, grace_seconds=0.0)
        langfuse_client_init.cache_clear()
    return captured["attributes"]


def test_exported_generation_fields_are_capped_before_ingestion(monkeypatch):
    _mock_mode(monkeypatch)
    monkeypatch.setattr(litellm, "langfuse_default_tags", ["huge_tag"])
    attributes: Final = _log_with_capture(
        monkeypatch,
        metadata={"trace_id": "b" * 32, "trace_version": "v" * 100_000, "huge_tag": "t" * 100_000},
        content_size=DEFAULT_MAX_FIELD_BYTES * 4,
    )

    envelope_slack: Final = 256  # the JSON envelope around the capped string
    for key in (A.OBSERVATION_INPUT, A.OBSERVATION_OUTPUT):
        value = attributes[key]
        assert _TRUNCATION_SUFFIX in value, f"{key} was not truncated"
        assert len(value.encode("utf-8")) <= (DEFAULT_MAX_FIELD_BYTES + len(_TRUNCATION_SUFFIX) + envelope_slack), (
            f"{key} exceeded the cap"
        )

    for key in (A.VERSION, A.TRACE_TAGS):
        assert len(str(attributes[key])) <= DEFAULT_MAX_FIELD_BYTES + envelope_slack, f"{key} exceeded the cap"

    assert json.loads(attributes[A.OBSERVATION_MODEL_PARAMETERS]) == {"temperature": 0.5}


def test_metadata_override_can_disable_capping(monkeypatch):
    _mock_mode(monkeypatch)
    attributes: Final = _log_with_capture(
        monkeypatch,
        metadata={"trace_id": "c" * 32, MAX_FIELD_BYTES_METADATA_KEY: 0},
        content_size=DEFAULT_MAX_FIELD_BYTES * 4,
    )

    assert len(attributes[A.OBSERVATION_INPUT]) > DEFAULT_MAX_FIELD_BYTES
    assert attributes[A.OBSERVATION_INPUT].find(_TRUNCATION_SUFFIX) == -1
