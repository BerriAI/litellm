"""Regression tests for ``litellm.proxy.db.log_db_metrics`` exception classification.

``_is_exception_related_to_db`` runs inside the DB failure handler on the request
path. It used to ``from prisma.errors import PrismaError`` on every call; prisma
is an optional dependency, so on a proxy without it that import raised
``ImportError`` straight out of the failure handler (masking the real error) and
re-ran the import finder/loader machinery every time. The classification is now
resolved once and degrades gracefully when prisma is absent.
"""

import httpx

import litellm.proxy.db.log_db_metrics as m


def _reset_cache():
    m._db_exception_types = None


def test_httpx_connection_errors_classified_as_db():
    _reset_cache()
    assert m._is_exception_related_to_db(httpx.ConnectError("boom")) is True
    assert m._is_exception_related_to_db(httpx.TimeoutException("slow")) is True


def test_non_db_exception_returns_false_without_raising():
    _reset_cache()
    # Must not raise even when prisma is not installed.
    assert m._is_exception_related_to_db(ValueError("nope")) is False
    assert m._is_exception_related_to_db(KeyError("nope")) is False


def test_exception_types_resolved_once_and_cached():
    _reset_cache()
    first = m._db_exception_types_cached()
    second = m._db_exception_types_cached()
    assert first is second
    assert httpx.ConnectError in first
    assert httpx.TimeoutException in first


def test_prisma_error_classified_as_db_when_available():
    _reset_cache()
    try:
        from prisma.errors import PrismaError
    except Exception:
        import pytest

        pytest.skip("prisma not installed")

    class _FakePrismaError(PrismaError):
        def __init__(self):
            pass

    assert m._is_exception_related_to_db(_FakePrismaError()) is True
