import pytest
from fastapi import HTTPException

from litellm.proxy.utils import _premium_user_check


def normalize(value):
    return value


def test_premium_user_check_happy_path_no_raise_when_premium(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    summary = {
        "result": _premium_user_check(),
        "premium_user": True,
        "raised": False,
    }
    assert summary == {
        "result": None,
        "premium_user": True,
        "raised": False,
    }


def test_premium_user_check_happy_path_with_feature_no_raise(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "premium_user", True, raising=False)
    summary = {
        "result": _premium_user_check(feature="model-routing"),
        "premium_user": True,
        "feature": "model-routing",
    }
    assert summary == {
        "result": None,
        "premium_user": True,
        "feature": "model-routing",
    }


def test_premium_user_check_raises_when_not_premium(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "premium_user", False, raising=False)
    with pytest.raises(HTTPException) as exc_info:
        _premium_user_check()
    snapshot = {
        "status_code": exc_info.value.status_code,
        "is_dict_detail": isinstance(exc_info.value.detail, dict),
        "has_error_key": "error" in exc_info.value.detail,
    }
    assert snapshot == {
        "status_code": 403,
        "is_dict_detail": True,
        "has_error_key": True,
    }


def test_premium_user_check_raises_with_feature_message(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "premium_user", False, raising=False)
    with pytest.raises(HTTPException) as exc_info:
        _premium_user_check(feature="custom-callbacks")
    error_msg = exc_info.value.detail["error"]
    snapshot = {
        "status_code": exc_info.value.status_code,
        "feature_in_message": "custom-callbacks" in error_msg,
        "enterprise_in_message": "LiteLLM Enterprise" in error_msg,
    }
    assert snapshot == {
        "status_code": 403,
        "feature_in_message": True,
        "enterprise_in_message": True,
    }
