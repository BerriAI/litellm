"""
Regression tests for BerriAI/litellm#45702: a native /v1/messages result is a
plain dict, so it carries no _hidden_params and the standard logging payload left
hidden_params["litellm_model_name"] as None. Deployment-level TPM/RPM accounting
(model_rate_limit_check, lowest_tpm_rpm_v2) keys its post-call increment on
<model_id>:<litellm_model_name>, so every such call was silently skipped and
enforce_model_rate_limits / usage-based-routing-v2 never saw the usage.
"""

import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import (
    Logging,
    get_standard_logging_object_payload,
)
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)

DEPLOYMENT_MODEL: Final = "anthropic/claude-opus-4-8"
DEPLOYMENT_ID: Final = "dep-anth"


def _call_kwargs() -> dict:
    # model_call_details keeps the full deployment model (provider prefix included);
    # Logging.model is the provider-stripped name, which must NOT be used here
    return {
        "model": DEPLOYMENT_MODEL,
        "litellm_params": {
            "model": DEPLOYMENT_MODEL,
            "metadata": {"model_group": "m-group", "model_info": {"id": DEPLOYMENT_ID}},
        },
        "call_type": "anthropic_messages",
        "litellm_call_id": "call-45702",
        "stream": False,
    }


def _anthropic_result() -> dict:
    # the native /v1/messages response reaches the payload builder as a plain dict
    return {
        "id": "msg_45702",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _logging_obj() -> Logging:
    return Logging(
        model=DEPLOYMENT_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="call-45702",
        function_id="fid-45702",
    )


def _payload(status: str = "success") -> dict:
    now: Final = datetime.datetime.now()
    return get_standard_logging_object_payload(
        kwargs=_call_kwargs(),
        init_response_obj=_anthropic_result(),
        start_time=now,
        end_time=now,
        logging_obj=_logging_obj(),
        status=status,
    )


def test_messages_plain_dict_result_backfills_litellm_model_name() -> None:
    payload: Final = _payload()
    assert payload["hidden_params"]["litellm_model_name"] == DEPLOYMENT_MODEL
    assert payload["model_id"] == DEPLOYMENT_ID


def test_messages_plain_dict_failure_payload_stays_unmarked() -> None:
    # failure payloads carry no model name on the object path either; keep it that way
    payload: Final = _payload(status="failure")
    assert payload["hidden_params"]["litellm_model_name"] is None


def test_object_result_hidden_params_win_over_backfill() -> None:
    response: Final = litellm.ModelResponse()
    response._hidden_params["litellm_model_name"] = "anthropic/claude-opus-4-8"
    kwargs: Final = _call_kwargs()
    kwargs["model"] = "m-group"
    now: Final = datetime.datetime.now()
    payload: Final = get_standard_logging_object_payload(
        kwargs=kwargs,
        init_response_obj=response,
        start_time=now,
        end_time=now,
        logging_obj=_logging_obj(),
        status="success",
    )
    assert payload["hidden_params"]["litellm_model_name"] == "anthropic/claude-opus-4-8"


@pytest.mark.asyncio
async def test_backfilled_name_keys_the_deployment_tpm_increment() -> None:
    # the pre-call check keys on <model_id>:<litellm_params.model>; the post-call
    # increment must land on the same key - a provider-stripped name would miss it
    mock_cache: Final = MagicMock()
    mock_cache.async_increment_cache = AsyncMock()
    check: Final = ModelRateLimitingCheck(dual_cache=mock_cache)

    await check.async_log_success_event({"standard_logging_object": _payload()}, None, None, None)

    mock_cache.async_increment_cache.assert_called_once()
    _, kwarg_params = mock_cache.async_increment_cache.call_args
    deployment: Final = {
        "litellm_params": {"model": DEPLOYMENT_MODEL},
        "model_info": {"id": DEPLOYMENT_ID},
    }
    tpm_key, _ = check._get_cache_keys(  # pyright: ignore[reportPrivateUsage]  # same seam the pre-call check uses
        deployment, litellm.utils.get_utc_datetime().strftime("%H-%M")
    )
    assert kwarg_params["key"] == tpm_key
