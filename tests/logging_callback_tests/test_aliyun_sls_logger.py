"""
Unit tests for AliyunSLSLogger.

Tests cover:
- __init__ reads from litellm.aliyun_sls_callback_params (programmatic config)
- __init__ falls back to os.environ (UI-configured params)
- __init__ raises ValueError when required params are missing
- _build_contents builds the expected log fields
- async_log_success_event / async_log_failure_event call _put_logs
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import litellm

# ---------------------------------------------------------------------------
# Stub aliyun.log so the module can be imported without the real SDK.
# ---------------------------------------------------------------------------
_aliyun_pkg = types.ModuleType("aliyun")
_aliyun_log = types.ModuleType("aliyun.log")
_aliyun_log.LogClient = MagicMock
_aliyun_log.LogItem = MagicMock
_aliyun_log.PutLogsRequest = MagicMock
_aliyun_pkg.log = _aliyun_log
sys.modules.setdefault("aliyun", _aliyun_pkg)
sys.modules.setdefault("aliyun.log", _aliyun_log)

# Import after stubs are in place.
from litellm.integrations.aliyun_sls import AliyunSLSLogger  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_PARAMS = {
    "aliyun_sls_region": "cn-hangzhou",
    "aliyun_sls_project": "my-project",
    "aliyun_sls_logstore": "my-logstore",
    "aliyun_sls_access_key_id": "ak-id",
    "aliyun_sls_access_key_secret": "ak-secret",
}


def _build_logger(monkeypatch, *, env: dict | None = None, callback_params: dict | None = None):
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", callback_params)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    with patch("litellm.integrations.aliyun_sls.LogClient", MagicMock()):
        return AliyunSLSLogger()


# ---------------------------------------------------------------------------
# __init__ — config sources
# ---------------------------------------------------------------------------


def test_init_from_callback_params(monkeypatch):
    """Params from litellm.aliyun_sls_callback_params are used."""
    params = {
        "aliyun_sls_region": "cn-beijing",
        "aliyun_sls_project": "proj",
        "aliyun_sls_logstore": "store",
        "aliyun_sls_access_key_id": "kid",
        "aliyun_sls_access_key_secret": "ksecret",
    }
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", params)
    mock_client_cls = MagicMock()
    with patch("litellm.integrations.aliyun_sls.LogClient", mock_client_cls):
        logger = AliyunSLSLogger()

    mock_client_cls.assert_called_once_with("cn-beijing.log.aliyuncs.com", "kid", "ksecret")
    assert logger.project == "proj"
    assert logger.logstore == "store"


def test_init_from_env_vars(monkeypatch):
    """Falls back to os.environ when litellm.aliyun_sls_callback_params is None."""
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", None)
    for k, v in _ENV_PARAMS.items():
        monkeypatch.setenv(k, v)

    mock_client_cls = MagicMock()
    with patch("litellm.integrations.aliyun_sls.LogClient", mock_client_cls):
        logger = AliyunSLSLogger()

    mock_client_cls.assert_called_once_with(
        "cn-hangzhou.log.aliyuncs.com", "ak-id", "ak-secret"
    )
    assert logger.project == "my-project"
    assert logger.logstore == "my-logstore"


def test_init_custom_endpoint_skips_region(monkeypatch):
    """Custom endpoint overrides the region-derived endpoint; region not required."""
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", None)
    env = {**_ENV_PARAMS, "aliyun_sls_endpoint": "custom.endpoint.com"}
    env.pop("aliyun_sls_region")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("aliyun_sls_region", raising=False)

    mock_client_cls = MagicMock()
    with patch("litellm.integrations.aliyun_sls.LogClient", mock_client_cls):
        AliyunSLSLogger()

    mock_client_cls.assert_called_once_with("custom.endpoint.com", "ak-id", "ak-secret")


def test_init_missing_region_raises(monkeypatch):
    """Missing region (and no custom endpoint) raises ValueError."""
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", None)
    for k, v in {k: v for k, v in _ENV_PARAMS.items() if k != "aliyun_sls_region"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("aliyun_sls_region", raising=False)
    monkeypatch.delenv("aliyun_sls_endpoint", raising=False)

    with patch("litellm.integrations.aliyun_sls.LogClient", MagicMock()):
        with pytest.raises(ValueError, match="aliyun_sls_region"):
            AliyunSLSLogger()


def test_init_missing_project_raises(monkeypatch):
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", None)
    for k, v in {k: v for k, v in _ENV_PARAMS.items() if k != "aliyun_sls_project"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("aliyun_sls_project", raising=False)

    with patch("litellm.integrations.aliyun_sls.LogClient", MagicMock()):
        with pytest.raises(ValueError, match="aliyun_sls_project"):
            AliyunSLSLogger()


def test_init_missing_logstore_raises(monkeypatch):
    monkeypatch.setattr(litellm, "aliyun_sls_callback_params", None)
    for k, v in {k: v for k, v in _ENV_PARAMS.items() if k != "aliyun_sls_logstore"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("aliyun_sls_logstore", raising=False)

    with patch("litellm.integrations.aliyun_sls.LogClient", MagicMock()):
        with pytest.raises(ValueError, match="aliyun_sls_logstore"):
            AliyunSLSLogger()


# ---------------------------------------------------------------------------
# _build_contents
# ---------------------------------------------------------------------------


def test_build_contents_no_payload(monkeypatch):
    """Without standard_logging_object only status is returned."""
    logger = _build_logger(monkeypatch, env=_ENV_PARAMS)
    result = logger._build_contents({}, None, None, None, "success")
    assert result == [("status", "success")]


def test_build_contents_with_payload(monkeypatch):
    logger = _build_logger(monkeypatch, env=_ENV_PARAMS)
    payload = {
        "model": "gpt-4o",
        "call_type": "completion",
        "id": "resp-123",
        "api_base": "https://api.openai.com",
        "response_time": 1234,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "response_cost": 0.005,
        "metadata": {
            "user_api_key_user_id": "uid-1",
            "user_api_key_team_id": "team-1",
            "user_api_key_alias": "my-key",
        },
        "error_str": None,
    }
    result = logger._build_contents(
        {"standard_logging_object": payload}, None, None, None, "success"
    )
    result_dict = dict(result)

    assert result_dict["status"] == "success"
    assert result_dict["model"] == "gpt-4o"
    assert result_dict["call_type"] == "completion"
    assert result_dict["response_id"] == "resp-123"
    assert result_dict["prompt_tokens"] == "10"
    assert result_dict["completion_tokens"] == "20"
    assert result_dict["total_tokens"] == "30"
    assert result_dict["total_cost"] == "0.005"
    assert result_dict["user"] == "uid-1"
    assert result_dict["team_id"] == "team-1"
    assert result_dict["key_alias"] == "my-key"
    assert "error" not in result_dict


def test_build_contents_includes_error(monkeypatch):
    logger = _build_logger(monkeypatch, env=_ENV_PARAMS)
    payload = {
        "model": "gpt-4o",
        "call_type": "completion",
        "error_str": "timeout",
        "metadata": {},
    }
    result = logger._build_contents(
        {"standard_logging_object": payload}, None, None, None, "failure"
    )
    result_dict = dict(result)
    assert result_dict["status"] == "failure"
    assert result_dict["error"] == "timeout"


# ---------------------------------------------------------------------------
# async_log_success_event / async_log_failure_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_log_success_event_calls_put_logs(monkeypatch):
    logger = _build_logger(monkeypatch, env=_ENV_PARAMS)
    logger._put_logs = MagicMock()

    await logger.async_log_success_event(
        kwargs={"standard_logging_object": None},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    logger._put_logs.assert_called_once()
    contents = logger._put_logs.call_args[0][0]
    assert ("status", "success") in contents


@pytest.mark.asyncio
async def test_async_log_failure_event_calls_put_logs(monkeypatch):
    logger = _build_logger(monkeypatch, env=_ENV_PARAMS)
    logger._put_logs = MagicMock()

    await logger.async_log_failure_event(
        kwargs={"standard_logging_object": None},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    logger._put_logs.assert_called_once()
    contents = logger._put_logs.call_args[0][0]
    assert ("status", "failure") in contents
