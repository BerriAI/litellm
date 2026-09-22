import logging
from collections.abc import Iterator
from typing import Final

import pytest

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.internal_key_emission_guard import (
    internal_key_leak_counter,
    litellm_owned_key_paths,
    observe_internal_keys,
)


@pytest.fixture
def litellm_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    verbose_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        verbose_logger.removeHandler(caplog.handler)


def test_nested_owned_key_reports_dotted_path() -> None:
    body: Final = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"litellm_call_id": "x", "user_tag": 1},
    }
    assert litellm_owned_key_paths(body) == ("metadata.litellm_call_id",)


def test_top_level_owned_keys_report_in_insertion_order() -> None:
    body: Final = {"model": "m", "litellm_params": {}, "_litellm_trace": 1}
    assert litellm_owned_key_paths(body) == ("litellm_params", "_litellm_trace")


def test_sequence_index_owned_key_reports_indexed_path() -> None:
    body: Final = {"model": "m", "tools": [{"_litellm_x": 1}]}
    assert litellm_owned_key_paths(body) == ("tools[0]._litellm_x",)


def test_clean_body_reports_no_paths() -> None:
    body: Final = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }
    assert litellm_owned_key_paths(body) == ()


def test_user_supplied_extra_body_keys_are_not_flagged() -> None:
    transformed: Final = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    extra_body: Final = {"custom_flag": True, "nested": {"vendor_option": 1}}
    data: Final = {**transformed, **extra_body}
    assert litellm_owned_key_paths(data) == ()


def test_observe_internal_keys_counts_and_warns_on_dirty_body(litellm_caplog: pytest.LogCaptureFixture) -> None:
    before: Final = internal_key_leak_counter.value
    with litellm_caplog.at_level(logging.WARNING, logger="LiteLLM"):
        paths: Final = observe_internal_keys({"litellm_call_id": "x", "model": "m"}, "deepseek")
    assert paths == ("litellm_call_id",)
    assert internal_key_leak_counter.value == before + 1
    assert any("litellm_call_id" in record.getMessage() for record in litellm_caplog.records)


def test_observe_internal_keys_does_not_count_or_warn_on_clean_body(litellm_caplog: pytest.LogCaptureFixture) -> None:
    before: Final = internal_key_leak_counter.value
    with litellm_caplog.at_level(logging.WARNING, logger="LiteLLM"):
        paths: Final = observe_internal_keys({"model": "m", "temperature": 0.2}, "deepseek")
    assert paths == ()
    assert internal_key_leak_counter.value == before
    assert not litellm_caplog.records
