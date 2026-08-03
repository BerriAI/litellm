"""Pin behavior of top-of-file and bottom-of-region helpers.

Covers ``print_verbose``, ``_get_email_logger_class``,
``_accepts_litellm_call_info``, ``_enrich_http_exception_with_guardrail_context``,
``on_backoff``, ``jsonify_object``, ``_lookup_deprecated_key``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy import utils as utils_mod
from litellm.proxy.utils import (
    _accepts_litellm_call_info,
    _enrich_http_exception_with_guardrail_context,
    _get_email_logger_class,
    _lookup_deprecated_key,
    jsonify_object,
    on_backoff,
    print_verbose,
)


# ---------------------------------------------------------------------------
# print_verbose
# ---------------------------------------------------------------------------


def test_print_verbose_when_set_verbose_true_prints_redacted(monkeypatch, capsys):
    monkeypatch.setattr(litellm, "set_verbose", True)
    print_verbose("hello world")
    captured = capsys.readouterr()
    snapshot = {
        "out_has_prefix": "LiteLLM Proxy:" in captured.out,
        "out_has_payload": "hello world" in captured.out,
        "no_stderr": captured.err == "",
    }
    assert snapshot == {"out_has_prefix": True, "out_has_payload": True, "no_stderr": True}


def test_print_verbose_when_set_verbose_false_no_stdout(monkeypatch, capsys):
    monkeypatch.setattr(litellm, "set_verbose", False)
    print_verbose("quiet")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_verbose_handles_unprintable_object_raises(monkeypatch):
    monkeypatch.setattr(litellm, "set_verbose", True)

    class Bomb:
        def __str__(self):
            raise RuntimeError("bad str")

    with pytest.raises(RuntimeError):
        print_verbose(Bomb())


# ---------------------------------------------------------------------------
# _get_email_logger_class
# ---------------------------------------------------------------------------


def test_get_email_logger_class_priority_matrix(monkeypatch):
    """Truth table for ``_get_email_logger_class`` priority: SendGrid >
    Resend > SMTP > Base."""
    sg = object()
    rs = object()
    smtp = object()
    base = object()
    monkeypatch.setattr(utils_mod, "BaseEmailLogger", base)
    monkeypatch.setattr(utils_mod, "SendGridEmailLogger", sg)
    monkeypatch.setattr(utils_mod, "ResendEmailLogger", rs)
    monkeypatch.setattr(utils_mod, "SMTPEmailLogger", smtp)
    for k in ("SENDGRID_API_KEY", "RESEND_API_KEY", "SMTP_HOST"):
        monkeypatch.delenv(k, raising=False)

    fallback = _get_email_logger_class() is base
    monkeypatch.setenv("SMTP_HOST", "smtp.example")
    smtp_choice = _get_email_logger_class() is smtp
    monkeypatch.setenv("RESEND_API_KEY", "rs-x")
    resend_choice = _get_email_logger_class() is rs
    monkeypatch.setenv("SENDGRID_API_KEY", "sg-x")
    sendgrid_choice = _get_email_logger_class() is sg
    snapshot = {
        "fallback_to_base": fallback,
        "smtp_when_smtp_only": smtp_choice,
        "resend_beats_smtp": resend_choice,
        "sendgrid_wins": sendgrid_choice,
    }
    assert snapshot == {
        "fallback_to_base": True,
        "smtp_when_smtp_only": True,
        "resend_beats_smtp": True,
        "sendgrid_wins": True,
    }


def test_get_email_logger_class_error_when_no_enterprise_module(monkeypatch):
    monkeypatch.setattr(utils_mod, "BaseEmailLogger", None)
    # Returns ``None`` rather than raising; this is the documented failure
    # mode when the optional enterprise package is missing.
    assert _get_email_logger_class() is None
    # Sentinel: monkey-patch SendGrid env but keep BaseEmailLogger None;
    # function still must return None and not blow up on the optional path.
    monkeypatch.setenv("SENDGRID_API_KEY", "sg-x")
    assert _get_email_logger_class() is None


# ---------------------------------------------------------------------------
# _accepts_litellm_call_info
# ---------------------------------------------------------------------------


class _CbAcceptsInfo:
    async def async_post_call_response_headers_hook(self, *, litellm_call_info=None):
        return None


class _CbRejectsInfo:
    async def async_post_call_response_headers_hook(self, *, response):
        return None


def test_accepts_litellm_call_info_matrix(monkeypatch):
    monkeypatch.setattr(utils_mod, "_CALLBACK_ACCEPTS_CALL_INFO", {})
    cache = {id(_CbAcceptsInfo): True}
    monkeypatch.setattr(utils_mod, "_CALLBACK_ACCEPTS_CALL_INFO", cache)
    snapshot = {
        "cache_hit_returns_true": _accepts_litellm_call_info(_CbAcceptsInfo()),
        "cache_size_after_hit": len(cache),
        "cache_keyed_by_type_id": id(_CbAcceptsInfo) in cache,
    }
    assert snapshot == {
        "cache_hit_returns_true": True,
        "cache_size_after_hit": 1,
        "cache_keyed_by_type_id": True,
    }


def test_accepts_litellm_call_info_signature_inspection(monkeypatch):
    monkeypatch.setattr(utils_mod, "_CALLBACK_ACCEPTS_CALL_INFO", {})
    snapshot = {
        "accepts_param_true": _accepts_litellm_call_info(_CbAcceptsInfo()),
        "rejects_param_false": _accepts_litellm_call_info(_CbRejectsInfo()),
        "cache_populated": len(utils_mod._CALLBACK_ACCEPTS_CALL_INFO) == 2,
    }
    assert snapshot == {
        "accepts_param_true": True,
        "rejects_param_false": False,
        "cache_populated": True,
    }


def test_accepts_litellm_call_info_error_on_callback_without_hook_raises(monkeypatch):
    monkeypatch.setattr(utils_mod, "_CALLBACK_ACCEPTS_CALL_INFO", {})

    class _Bad:
        pass

    with pytest.raises(AttributeError):
        _accepts_litellm_call_info(_Bad())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _enrich_http_exception_with_guardrail_context
# ---------------------------------------------------------------------------


def test_enrich_http_exception_adds_guardrail_name_and_mode():
    detail = {"error": "blocked"}
    exc = HTTPException(status_code=400, detail=detail)
    cb = MagicMock()
    cb.guardrail_name = "presidio"
    cb.event_hook = "pre_call"

    _enrich_http_exception_with_guardrail_context(exc, cb)
    snapshot = {
        "error": detail["error"],
        "guardrail_name": detail["guardrail_name"],
        "guardrail_mode": detail["guardrail_mode"],
    }
    assert snapshot == {
        "error": "blocked",
        "guardrail_name": "presidio",
        "guardrail_mode": "pre_call",
    }


def test_enrich_http_exception_does_not_overwrite_existing_keys():
    detail = {"error": "blocked", "guardrail_name": "explicit", "guardrail_mode": "during_call"}
    exc = HTTPException(status_code=400, detail=detail)
    cb = MagicMock()
    cb.guardrail_name = "should-not-overwrite"
    cb.event_hook = "should-not-overwrite"
    _enrich_http_exception_with_guardrail_context(exc, cb)
    assert detail == {"error": "blocked", "guardrail_name": "explicit", "guardrail_mode": "during_call"}


def test_enrich_http_exception_no_op_for_non_http_exception():
    other = ValueError("not http")
    _enrich_http_exception_with_guardrail_context(other, MagicMock(guardrail_name="g"))


def test_enrich_http_exception_no_op_for_non_dict_detail():
    exc = HTTPException(status_code=400, detail="just a string")
    _enrich_http_exception_with_guardrail_context(exc, MagicMock(guardrail_name="g"))
    assert exc.detail == "just a string"


def test_enrich_http_exception_error_handling_does_not_raise():
    """``_enrich_http_exception_with_guardrail_context`` swallows mismatched
    inputs (non-HTTPException, non-dict detail, no guardrail_name) and never
    raises — verified by passing each pathological input in turn."""
    # Bare exception with no detail at all should not blow up.
    bare = Exception("bare")
    _enrich_http_exception_with_guardrail_context(bare, MagicMock(guardrail_name=None))
    # HTTPException with non-dict detail.
    s = HTTPException(status_code=500, detail="str-detail")
    _enrich_http_exception_with_guardrail_context(s, MagicMock(guardrail_name="g"))
    assert s.detail == "str-detail"


def test_enrich_http_exception_with_falsy_attrs_does_not_set():
    detail = {"error": "blocked"}
    exc = HTTPException(status_code=400, detail=detail)
    cb = MagicMock()
    cb.guardrail_name = None
    cb.event_hook = None
    _enrich_http_exception_with_guardrail_context(exc, cb)
    assert detail == {"error": "blocked"}


# ---------------------------------------------------------------------------
# on_backoff
# ---------------------------------------------------------------------------


def test_on_backoff_invokes_print_verbose(monkeypatch):
    captured = []
    monkeypatch.setattr(utils_mod, "print_verbose", lambda s: captured.append(s))
    on_backoff({"tries": 3})
    snapshot = {"len": len(captured), "first_has_attempt": "attempt" in captured[0], "first_has_3": "3" in captured[0]}
    assert snapshot == {"len": 1, "first_has_attempt": True, "first_has_3": True}


def test_on_backoff_missing_tries_key_raises():
    with pytest.raises(KeyError):
        on_backoff({})


# ---------------------------------------------------------------------------
# jsonify_object
# ---------------------------------------------------------------------------


def test_jsonify_object_serializes_nested_dicts():
    src = {"plain": "x", "nested": {"a": 1, "b": 2}, "n": 42}
    out = jsonify_object(src)
    expected = {"plain": "x", "nested": '{"a": 1, "b": 2}', "n": 42}
    assert out == expected
    # Source is not mutated.
    assert src == {"plain": "x", "nested": {"a": 1, "b": 2}, "n": 42}


def test_jsonify_object_failed_serialization_marks_value(monkeypatch):
    class Unserialiseable:
        pass

    src = {"name": "x", "bad": {"obj": Unserialiseable()}, "count": 1}
    out = jsonify_object(src)
    assert out == {"name": "x", "bad": "failed-to-serialize-json", "count": 1}


def test_jsonify_object_non_dict_input_raises():
    with pytest.raises(AttributeError):
        jsonify_object("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _lookup_deprecated_key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_deprecated_key_returns_active_token_id_and_caches(monkeypatch):
    from litellm.caching.dual_cache import LimitedSizeOrderedDict

    fresh = LimitedSizeOrderedDict(max_size=1000)
    monkeypatch.setattr(utils_mod, "_deprecated_key_cache", fresh)

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    deprecated_row = MagicMock()
    deprecated_row.active_token_id = "active-123"
    deprecated_row.revoke_at = future

    db = MagicMock()
    db.litellm_deprecatedverificationtoken.find_first = AsyncMock(return_value=deprecated_row)

    result = await _lookup_deprecated_key(db=db, hashed_token="hash-abc")
    cached_value = fresh.get("hash-abc")
    snapshot = {
        "result": result,
        "cache_active_token_id": cached_value[0],
        "cache_has_3_tuple": isinstance(cached_value, tuple) and len(cached_value) == 3,
    }
    assert snapshot == {
        "result": "active-123",
        "cache_active_token_id": "active-123",
        "cache_has_3_tuple": True,
    }


@pytest.mark.asyncio
async def test_lookup_deprecated_key_returns_none_when_not_found(monkeypatch):
    from litellm.caching.dual_cache import LimitedSizeOrderedDict

    monkeypatch.setattr(utils_mod, "_deprecated_key_cache", LimitedSizeOrderedDict(max_size=10))
    db = MagicMock()
    db.litellm_deprecatedverificationtoken.find_first = AsyncMock(return_value=None)
    assert await _lookup_deprecated_key(db=db, hashed_token="missing") is None


@pytest.mark.asyncio
async def test_lookup_deprecated_key_db_error_returns_none(monkeypatch):
    from litellm.caching.dual_cache import LimitedSizeOrderedDict

    monkeypatch.setattr(utils_mod, "_deprecated_key_cache", LimitedSizeOrderedDict(max_size=10))
    db = MagicMock()
    db.litellm_deprecatedverificationtoken.find_first = AsyncMock(side_effect=RuntimeError("db down"))
    result = await _lookup_deprecated_key(db=db, hashed_token="x")
    assert result is None


@pytest.mark.asyncio
async def test_lookup_deprecated_key_uses_cache_within_ttl(monkeypatch):
    from litellm.caching.dual_cache import LimitedSizeOrderedDict

    cache = LimitedSizeOrderedDict(max_size=10)
    now_ts = datetime.now(timezone.utc).timestamp()
    cache["hashY"] = ("active-from-cache", now_ts + 100, now_ts + 1000)
    monkeypatch.setattr(utils_mod, "_deprecated_key_cache", cache)

    db = MagicMock()
    db.litellm_deprecatedverificationtoken.find_first = AsyncMock(return_value=None)
    result = await _lookup_deprecated_key(db=db, hashed_token="hashY")
    assert result == "active-from-cache"
    db.litellm_deprecatedverificationtoken.find_first.assert_not_called()
