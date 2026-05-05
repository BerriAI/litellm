"""Unit tests for the shared VCR helpers in ``tests/_vcr_conftest_common``.

The most important guarantee here is that the custom ``safe_body`` matcher
gracefully handles JSON Lines (and other non-strict-JSON) request bodies
without raising ``json.JSONDecodeError`` — vcrpy's default ``body`` matcher
crashes on those because it unconditionally runs ``json.loads`` for any
``application/json`` request body.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# Tests live in ``tests/test_litellm/`` but ``_vcr_conftest_common`` lives in
# the parent ``tests/`` package. Make sure both are importable regardless of
# how pytest is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._vcr_conftest_common import (  # noqa: E402
    SAFE_BODY_MATCHER_NAME,
    _safe_body_matcher,
    vcr_config_dict,
)


def _req(body):
    return SimpleNamespace(body=body, headers={"Content-Type": "application/json"})


def test_safe_body_matcher_is_in_match_on():
    cfg = vcr_config_dict()
    assert SAFE_BODY_MATCHER_NAME in cfg["match_on"]
    assert "body" not in cfg["match_on"]


def test_safe_body_matcher_accepts_identical_bytes():
    _safe_body_matcher(_req(b"hello"), _req(b"hello"))


def test_safe_body_matcher_accepts_str_bytes_equivalent():
    _safe_body_matcher(_req("hello"), _req(b"hello"))


def test_safe_body_matcher_handles_jsonl_without_crashing():
    """vcrpy's default ``body`` matcher raises ``JSONDecodeError`` on JSONL.

    The Bedrock batch S3 PUT sends a JSON Lines body under
    ``Content-Type: application/json``. The safe matcher must compare such
    bodies as bytes and never invoke ``json.loads``.
    """
    jsonl = (
        b'{"recordId": "request-1", "modelInput": {}}\n'
        b'{"recordId": "request-2", "modelInput": {}}\n'
    )
    _safe_body_matcher(_req(jsonl), _req(jsonl))


def test_safe_body_matcher_rejects_different_jsonl_bodies():
    a = b'{"recordId": "request-1"}\n{"recordId": "request-2"}\n'
    b = b'{"recordId": "request-1"}\n{"recordId": "request-3"}\n'
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(a), _req(b))


def test_safe_body_matcher_rejects_different_bytes():
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(b"a"), _req(b"b"))


def test_safe_body_matcher_treats_none_bodies_as_equal():
    _safe_body_matcher(_req(None), _req(None))


def test_safe_body_matcher_does_not_normalize_json_key_order():
    """The safe matcher is strictly more conservative than vcrpy's default.

    Two semantically-equal JSON bodies with different key order are
    treated as *different* requests (cache miss, never a false hit).
    """
    with pytest.raises(AssertionError):
        _safe_body_matcher(_req(b'{"a":1,"b":2}'), _req(b'{"b":2,"a":1}'))


def test_default_vcrpy_body_matcher_crashes_on_jsonl_for_documentation():
    """Document the behavior we are working around.

    vcrpy's stock body matcher raises ``json.JSONDecodeError`` (not even
    a clean ``AssertionError``) when given a JSONL payload typed as
    ``application/json``. This is precisely the crash that broke
    ``tests/batches_tests/test_bedrock_files_and_batches.py::test_async_create_file``
    and is the reason ``safe_body`` exists.
    """
    import json

    from vcr.matchers import body as vcrpy_body  # type: ignore

    jsonl = b'{"recordId": "request-1"}\n{"recordId": "request-2"}\n'
    with pytest.raises(json.JSONDecodeError):
        vcrpy_body(_req(jsonl), _req(jsonl))
