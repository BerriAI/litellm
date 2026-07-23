"""
Tests for ``_ProxyDBLogger.async_log_failure_event``.

This is the per-deployment-attempt failure logger: the SDK fires
``async_log_failure_event`` for every failed ``litellm.acompletion`` attempt,
including attempts later rescued by a router retry/fallback. Without it, a
failure rescued by a fallback writes no row to ``LiteLLM_SpendLogs`` and is
invisible in the spend logs UI.

The kwargs are produced by a REAL router failure (mock_response error) so the
``standard_logging_object`` / metadata shape matches production, then fed to the
method with ``_insert_spend_log_to_db`` patched so we can inspect the row that
would be written.
"""
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import Router
from litellm.constants import (
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    MAX_STRING_LENGTH_PROMPT_IN_DB,
    REDACTED_BY_LITELM_STRING,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger

_INSERT_TARGET = (
    "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter._insert_spend_log_to_db"
)


@pytest.fixture(autouse=True)
def _restore_callbacks():
    """Each test mutates litellm.callbacks; restore afterwards."""
    saved = litellm.callbacks
    yield
    litellm.callbacks = saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deployment(model_id, mock_response):
    return {
        "model_name": "primary",
        "litellm_params": {
            "model": "openai/primary",
            "api_key": "sk-fake",
            "api_base": "https://api.openai.com/v1",
            "mock_response": mock_response,
        },
        "model_info": {"id": model_id},
    }


async def _capture_attempt_kwargs(model_id="PRIMARY_ID", call_id="CALLID"):
    """Run a single failing deployment through the router and capture the
    kwargs/timing that ``async_log_failure_event`` receives for that attempt."""
    captured = {}

    class _Cap(CustomLogger):
        async def async_log_failure_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            captured.setdefault(
                "args",
                {
                    "kwargs": kwargs,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )

    litellm.callbacks = [_Cap()]
    router = Router(
        model_list=[_deployment(model_id, "litellm.InternalServerError")],
        num_retries=0,
    )
    try:
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            litellm_call_id=call_id,
            metadata={
                "user_api_key_user_id": "u1",
                "user_api_key_team_id": "t1",
                "user_api_key_org_id": "o1",
                "user_api_key": "sk-hash",
            },
        )
    except Exception:
        pass
    await asyncio.sleep(0.3)
    assert "args" in captured, "expected an async_log_failure_event to fire"
    return captured["args"]


def _failure_payloads(mock_insert):
    """Return only the failure-status payloads passed to _insert_spend_log_to_db."""
    payloads = [c.kwargs["payload"] for c in mock_insert.call_args_list]
    return [p for p in payloads if p.get("status") == "failure"]


# ---------------------------------------------------------------------------
# Direct unit tests on async_log_failure_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writes_failure_row_with_expected_fields():
    args = await _capture_attempt_kwargs(model_id="PRIMARY_ID", call_id="CALLID")
    logger = _ProxyDBLogger()

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=args["kwargs"],
            response_obj=None,
            start_time=args["start_time"],
            end_time=args["end_time"],
        )

    mock_insert.assert_called_once()
    payload = mock_insert.call_args.kwargs["payload"]

    assert payload["status"] == "failure"
    assert payload["spend"] == 0.0
    assert payload["model_id"] == "PRIMARY_ID"

    # Unique, correlatable request_id (never collides with success/terminal row).
    assert payload["request_id"].startswith("CALLID_attempt_")
    assert payload["request_id"] != "CALLID"

    # error_information lifted from the SLO into the row metadata so the UI's
    # error_code / error_message filters work.
    md = json.loads(payload["metadata"])
    assert md["status"] == "failure"
    assert md["error_information"]["error_code"] == "500"
    assert "InternalServerError" in md["error_information"]["error_class"]


@pytest.mark.asyncio
async def test_request_id_unique_per_attempt():
    """Two failures sharing the same litellm_call_id must produce DISTINCT
    request_ids, else create_many(skip_duplicates=True) drops one."""
    args = await _capture_attempt_kwargs(call_id="SHARED")
    logger = _ProxyDBLogger()

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=args["kwargs"], response_obj=None, **_timing(args)
        )
        await logger.async_log_failure_event(
            kwargs=args["kwargs"], response_obj=None, **_timing(args)
        )

    rids = [c.kwargs["payload"]["request_id"] for c in mock_insert.call_args_list]
    assert len(rids) == 2
    assert rids[0] != rids[1]
    assert all(r.startswith("SHARED_attempt_") for r in rids)


@pytest.mark.asyncio
async def test_status_failure_even_when_litellm_metadata_present():
    """get_litellm_metadata_from_kwargs prefers litellm_metadata over metadata;
    status must be stamped on the effective dict so the row isn't 'success'.
    Inject a litellm_metadata WITHOUT status to prove the method stamps it."""
    args = await _capture_attempt_kwargs()
    lp = dict(args["kwargs"]["litellm_params"])
    lp["litellm_metadata"] = {
        "user_api_key_user_id": "u1",
        "model_group": "primary",
        "model_info": {"id": "PRIMARY_ID"},
    }
    kwargs = {**args["kwargs"], "litellm_params": lp}
    logger = _ProxyDBLogger()

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=kwargs, response_obj=None, **_timing(args)
        )

    mock_insert.assert_called_once()
    assert mock_insert.call_args.kwargs["payload"]["status"] == "failure"


@pytest.mark.asyncio
@patch(
    "litellm.proxy.spend_tracking.spend_tracking_utils._should_store_prompts_and_responses_in_spend_logs"
)
async def test_per_attempt_failure_sanitizes_error_information_before_spend_log(
    mock_should_store,
):
    """Per-attempt failure rows must honor spend-log prompt redaction and caps."""
    mock_should_store.return_value = False
    args = await _capture_attempt_kwargs()
    logger = _ProxyDBLogger()

    leaked_prompt = "super-secret-user-prompt"
    huge_traceback = "traceback: " + ("x" * (MAX_STRING_LENGTH_PROMPT_IN_DB * 2))
    error_information = {
        "error_code": "400",
        "error_class": "BadRequestError",
        "llm_provider": "openai",
        "error_message": (
            'ProviderError - {"error":{"message":"validation failed",'
            f'"input":[{{"role":"user","content":"{leaked_prompt}"}}]}}}}'
        ),
        "traceback": huge_traceback,
    }
    kwargs = {
        **args["kwargs"],
        "standard_logging_object": {
            **args["kwargs"]["standard_logging_object"],
            "error_information": error_information,
        },
    }

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=kwargs, response_obj=None, **_timing(args)
        )

    mock_insert.assert_called_once()
    payload = mock_insert.call_args.kwargs["payload"]
    md = json.loads(payload["metadata"])
    logged_error_information = md["error_information"]
    logged_error_json = json.dumps(logged_error_information)

    assert leaked_prompt not in logged_error_json
    assert REDACTED_BY_LITELM_STRING in logged_error_information["error_message"]
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD in logged_error_information["traceback"]


@pytest.mark.asyncio
async def test_respects_disable_error_logs():
    args = await _capture_attempt_kwargs()
    logger = _ProxyDBLogger()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"disable_error_logs": True},
    ), patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=args["kwargs"], response_obj=None, **_timing(args)
        )

    mock_insert.assert_not_called()


@pytest.mark.asyncio
async def test_missing_litellm_call_id_is_noop():
    logger = _ProxyDBLogger()
    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs={"litellm_params": {}},
            response_obj=None,
            start_time=None,
            end_time=None,
        )
    mock_insert.assert_not_called()


@pytest.mark.asyncio
async def test_respects_disable_spend_logs():
    """disable_spend_logs is the operator kill-switch for SpendLogs writes.
    Because we call _insert_spend_log_to_db directly (bypassing update_database's
    own guard), the failure path must honor it explicitly."""
    args = await _capture_attempt_kwargs()
    logger = _ProxyDBLogger()

    with patch("litellm.proxy.proxy_server.disable_spend_logs", True), patch(
        _INSERT_TARGET, new_callable=AsyncMock
    ) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=args["kwargs"], response_obj=None, **_timing(args)
        )

    mock_insert.assert_not_called()


@pytest.mark.asyncio
async def test_skips_call_without_identity():
    """Identity-less internal calls (e.g. semantic-cache embeddings) must NOT
    write blank failure rows — mirror the success path's _should_track_cost_callback
    gate. litellm_call_id is present but metadata carries no user_api_key*."""
    logger = _ProxyDBLogger()
    kwargs = {
        "litellm_call_id": "NOIDENTITY",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {"error_information": {"error_code": "500"}},
    }

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        await logger.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )

    mock_insert.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end through the router (real async_log_failure_event dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_rescued_failure_is_logged():
    """primary fails -> backup succeeds. The rescued failure must produce a
    failure row (distinct request_id), and the success path must not be blocked."""
    logger = _ProxyDBLogger()
    litellm.callbacks = [logger]

    router = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/primary",
                    "api_key": "x",
                    "mock_response": "litellm.InternalServerError",
                },
                "model_info": {"id": "PRIMARY_ID"},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "openai/backup",
                    "api_key": "x",
                    "mock_response": "rescued ok",
                },
                "model_info": {"id": "BACKUP_ID"},
            },
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=0,
    )

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        resp = await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            litellm_call_id="RESCUED",
            metadata={"user_api_key_user_id": "u1", "user_api_key": "sk-hash"},
        )
        await asyncio.sleep(0.5)

    assert resp.choices[0].message.content == "rescued ok"

    failures = _failure_payloads(mock_insert)
    assert len(failures) == 1
    assert failures[0]["model_id"] == "PRIMARY_ID"
    assert failures[0]["request_id"].startswith("RESCUED_attempt_")
    assert failures[0]["request_id"] != "RESCUED"


@pytest.mark.asyncio
async def test_all_fail_writes_distinct_rows_per_attempt():
    """primary + backup both fail. Each attempt gets its own failure row with a
    distinct request_id (no skip_duplicates collision)."""
    logger = _ProxyDBLogger()
    litellm.callbacks = [logger]

    router = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "openai/primary",
                    "api_key": "x",
                    "mock_response": "litellm.InternalServerError",
                },
                "model_info": {"id": "PRIMARY_ID"},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "openai/backup",
                    "api_key": "x",
                    "mock_response": "litellm.InternalServerError",
                },
                "model_info": {"id": "BACKUP_ID"},
            },
        ],
        fallbacks=[{"primary": ["backup"]}],
        num_retries=0,
    )

    with patch(_INSERT_TARGET, new_callable=AsyncMock) as mock_insert:
        try:
            await router.acompletion(
                model="primary",
                messages=[{"role": "user", "content": "hi"}],
                litellm_call_id="ALLFAIL",
                metadata={"user_api_key_user_id": "u1", "user_api_key": "sk-hash"},
            )
        except Exception:
            pass
        await asyncio.sleep(0.5)

    failures = _failure_payloads(mock_insert)
    model_ids = {p["model_id"] for p in failures}
    request_ids = {p["request_id"] for p in failures}
    assert {"PRIMARY_ID", "BACKUP_ID"}.issubset(model_ids)
    # one row per attempt, all distinct request_ids, all correlatable to call_id
    assert len(request_ids) == len(failures) >= 2
    assert all(r.startswith("ALLFAIL_attempt_") for r in request_ids)


def _timing(args):
    return {"start_time": args["start_time"], "end_time": args["end_time"]}
