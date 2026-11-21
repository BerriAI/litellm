import pytest

import litellm
from litellm.evals.main import (
    INTERNAL_METADATA_KEY_PREFIXES,
    INTERNAL_METADATA_KEYS,
    _is_internal_metadata_key,
)
from litellm.types.llms.openai_evals import Eval, Run

_DATA_SOURCE_CONFIG = {"type": "custom", "item_schema": {"type": "object"}}
_TESTING_CRITERIA = [{"type": "label_model", "name": "grader"}]
_DATA_SOURCE = {"type": "jsonl", "source": {"type": "file_id", "id": "file-123"}}


@pytest.fixture
def captured_body(monkeypatch):
    seen = {}

    def _capture(handler_name, response):
        def _handler(*args, **kwargs):
            seen["request_body"] = kwargs["request_body"]
            return response

        monkeypatch.setattr(litellm.evals.main.base_llm_http_handler, handler_name, _handler)

    eval_response = Eval(
        id="eval-123",
        created_at=0,
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
    )
    _capture("create_eval_handler", eval_response)
    _capture("update_eval_handler", eval_response)
    _capture(
        "create_run_handler",
        Run(
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
        api_key="fake-key",
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
        api_key="fake-key",
    )
    assert captured_body["request_body"]["metadata"] == {"team": "search"}


@pytest.mark.parametrize("internal_key", sorted(INTERNAL_METADATA_KEYS))
def test_create_eval_strips_internal_metadata_keys(captured_body, internal_key):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        metadata={internal_key: {"leaked": True}, "team": "search"},
        api_key="fake-key",
    )
    assert captured_body["request_body"]["metadata"] == {"team": "search"}


def test_create_eval_omits_metadata_when_only_internal_keys(captured_body):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        metadata={"user_api_key_hash": "leaked", "endpoint": "/v1/evals"},
        api_key="fake-key",
    )
    assert "metadata" not in captured_body["request_body"]


def test_create_eval_omits_metadata_when_not_given(captured_body):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        api_key="fake-key",
    )
    assert "metadata" not in captured_body["request_body"]


@pytest.mark.parametrize(
    "internal_key",
    [
        "user_api_key_token",
        "user_api_key_team_metadata",
        "user_api_key_object_permission_id",
        "user_api_key_team_object_permission_id",
        "user_api_key_budget_reservation",
        "user_api_key_org_alias",
        "user_api_key_project_id",
        "litellm_parent_otel_span",
        "litellm_api_version",
    ],
)
def test_create_eval_strips_prefixed_internal_keys(captured_body, internal_key):
    litellm.create_eval(
        data_source_config=_DATA_SOURCE_CONFIG,
        testing_criteria=_TESTING_CRITERIA,
        metadata={internal_key: {"leaked": True}, "team": "search"},
        api_key="fake-key",
    )
    assert captured_body["request_body"]["metadata"] == {"team": "search"}


def test_every_user_api_key_metadata_key_is_covered():
    assert "user_api_key_" in INTERNAL_METADATA_KEY_PREFIXES
    for key in ("user_api_key_token", "user_api_key_hash", "user_api_key_some_field_added_in_2027"):
        assert _is_internal_metadata_key(key)
    assert not _is_internal_metadata_key("my_user_api_key_note")
    assert not _is_internal_metadata_key("team")


_PREVIOUS_DENYLIST = {
    "headers",
    "requester_metadata",
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_spend",
    "user_api_key_max_budget",
    "user_api_key_team_id",
    "user_api_key_user_id",
    "user_api_key_org_id",
    "user_api_key_team_alias",
    "user_api_key_end_user_id",
    "user_api_key_user_email",
    "user_api_key_request_route",
    "user_api_key_budget_reset_at",
    "user_api_key_auth_metadata",
    "user_api_key",
    "user_api_end_user_max_budget",
    "user_api_key_auth",
    "litellm_api_version",
    "global_max_parallel_requests",
    "user_api_key_team_max_budget",
    "user_api_key_team_spend",
    "user_api_key_model_max_budget",
    "user_api_key_user_spend",
    "user_api_key_user_max_budget",
    "user_api_key_metadata",
    "endpoint",
    "litellm_parent_otel_span",
    "requester_ip_address",
    "user_agent",
}


def test_prefix_filter_covers_everything_the_old_denylist_did():
    missed = sorted(k for k in _PREVIOUS_DENYLIST if not _is_internal_metadata_key(k))
    assert not missed, f"prefix filter no longer strips {missed}"


def test_proxy_metadata_control_fields_are_covered():
    pre_call_utils = pytest.importorskip("litellm.proxy.litellm_pre_call_utils")

    missed = sorted(k for k in pre_call_utils._UNTRUSTED_METADATA_CONTROL_FIELDS if not _is_internal_metadata_key(k))
    assert not missed, f"evals filter does not strip proxy-injected {missed}"


def test_update_eval_filtering_is_unchanged(captured_body):
    litellm.update_eval(
        eval_id="eval-123",
        metadata={"team": "search", "user_api_key_hash": "leaked", "user_api_key_token": "leaked"},
        api_key="fake-key",
    )
    assert captured_body["request_body"]["metadata"] == {"team": "search"}


def test_update_eval_omits_metadata_when_only_internal_keys(captured_body):
    litellm.update_eval(
        eval_id="eval-123",
        metadata={"user_api_key_hash": "leaked"},
        api_key="fake-key",
    )
    assert "metadata" not in captured_body["request_body"]
