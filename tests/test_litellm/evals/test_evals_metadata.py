"""Regression tests: create_eval / create_run must forward the caller's metadata.

Both functions document a ``metadata`` parameter and both request TypedDicts
(``CreateEvalRequest``, ``CreateRunRequest``) declare a ``metadata`` field, but
neither builder ever set it, so the value was accepted and silently dropped. In
``create_run`` the assignment was present but commented out.

``update_eval`` in the same module already did the right thing, filtering the
internal LiteLLM metadata keys the proxy attaches before forwarding the rest.
These tests pin that same behaviour onto the two create paths: the caller's keys
reach the request body, and the internal keys never do.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.evals.main import INTERNAL_METADATA_KEYS

_DATA_SOURCE_CONFIG = {"type": "custom", "item_schema": {"type": "object"}}
_TESTING_CRITERIA = [{"type": "label_model", "name": "grader"}]
_DATA_SOURCE = {"type": "jsonl", "source": {"type": "file_id", "id": "file-123"}}


@pytest.fixture
def captured_body(monkeypatch):
    """Capture the request body handed to the HTTP layer, without a network call."""
    seen = {}

    def _capture(handler_name, response):
        def _handler(*args, **kwargs):
            seen["request_body"] = kwargs["request_body"]
            return response

        monkeypatch.setattr(litellm.evals.main.base_llm_http_handler, handler_name, _handler)

    _capture(
        "create_eval_handler",
        litellm.types.llms.openai_evals.Eval(
            id="eval-123",
            created_at=0,
            data_source_config=_DATA_SOURCE_CONFIG,
            testing_criteria=_TESTING_CRITERIA,
        ),
    )
    _capture(
        "create_run_handler",
        litellm.types.llms.openai_evals.Run(
            id="run-123",
            eval_id="eval-123",
            created_at=0,
            status="queued",
            model="gpt-4o",
            data_source=_DATA_SOURCE,
        ),
    )
    return seen


def test_create_eval_forwards_caller_metadata(captured_body):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        metadata={"team": "search", "run_by": "nightly"},
        api_key="sk-test",
    )
    assert captured_body["request_body"]["metadata"] == {
        "team": "search",
        "run_by": "nightly",
    }


def test_create_run_forwards_caller_metadata(captured_body):
    litellm.create_run(
        eval_id="eval-123",
        data_source=_DATA_SOURCE,
        metadata={"team": "search"},
        api_key="sk-test",
    )
    assert captured_body["request_body"]["metadata"] == {"team": "search"}


@pytest.mark.parametrize("internal_key", sorted(INTERNAL_METADATA_KEYS))
def test_create_eval_strips_internal_metadata_keys(captured_body, internal_key):
    # A dict value keeps every key valid for the logging path, which reads some
    # of these (e.g. requester_metadata) as mappings on the way through.
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        metadata={internal_key: {"leaked": True}, "team": "search"},
        api_key="sk-test",
    )
    assert captured_body["request_body"]["metadata"] == {"team": "search"}


def test_create_eval_omits_metadata_when_only_internal_keys(captured_body):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        metadata={"user_api_key_hash": "leaked", "endpoint": "/v1/evals"},
        api_key="sk-test",
    )
    assert "metadata" not in captured_body["request_body"]


def test_create_eval_omits_metadata_when_not_given(captured_body):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        api_key="sk-test",
    )
    assert "metadata" not in captured_body["request_body"]
