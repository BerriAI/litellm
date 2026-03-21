"""
Tests for PrismaModelProxy and PrismaWrapper DB metrics instrumentation.

Verifies that actual Prisma CRUD operations are instrumented with
ServiceTypes.DB_READ / DB_WRITE via ServiceLogging, and that
non-CRUD / passthrough attributes are forwarded unchanged.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.db.prisma_client import (
    _DB_READ_METHODS,
    _DB_WRITE_METHODS,
    _PASSTHROUGH_ATTRS,
    _RAW_METHOD_SERVICE_TYPES,
    PrismaModelProxy,
    PrismaWrapper,
    _wrap_raw_query_method,
)
from litellm.types.services import ServiceTypes


def _make_mock_service_logger():
    """Create a mock ServiceLogging with async hooks."""
    logger = MagicMock()
    logger.async_service_success_hook = AsyncMock()
    logger.async_service_failure_hook = AsyncMock()
    return logger


def _make_mock_model():
    """Create a mock Prisma model with all CRUD methods as AsyncMocks."""
    model = MagicMock()
    for method_name in _DB_READ_METHODS | _DB_WRITE_METHODS:
        setattr(model, method_name, AsyncMock(return_value={"id": "test"}))
    return model


# ---------------------------------------------------------------------------
# PrismaModelProxy tests
# ---------------------------------------------------------------------------


class TestPrismaModelProxy:
    @pytest.mark.parametrize("method_name", sorted(_DB_READ_METHODS))
    @pytest.mark.asyncio
    async def test_should_log_read_methods_as_db_read(self, method_name):
        model = _make_mock_model()
        logger = _make_mock_service_logger()
        proxy = PrismaModelProxy(model, "litellm_usertable", logger)

        wrapped = getattr(proxy, method_name)
        result = await wrapped(where={"id": "x"})

        assert result == {"id": "test"}
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_READ
        assert call_kwargs["call_type"] == f"litellm_usertable.{method_name}"
        assert call_kwargs["duration"] >= 0

    @pytest.mark.parametrize("method_name", sorted(_DB_WRITE_METHODS))
    @pytest.mark.asyncio
    async def test_should_log_write_methods_as_db_write(self, method_name):
        model = _make_mock_model()
        logger = _make_mock_service_logger()
        proxy = PrismaModelProxy(model, "litellm_teamtable", logger)

        wrapped = getattr(proxy, method_name)
        result = await wrapped(data={"name": "test"})

        assert result == {"id": "test"}
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_WRITE
        assert call_kwargs["call_type"] == f"litellm_teamtable.{method_name}"

    @pytest.mark.asyncio
    async def test_should_log_failure_on_exception(self):
        model = _make_mock_model()
        model.find_unique = AsyncMock(side_effect=RuntimeError("connection lost"))
        logger = _make_mock_service_logger()
        proxy = PrismaModelProxy(model, "litellm_usertable", logger)

        with pytest.raises(RuntimeError, match="connection lost"):
            await proxy.find_unique(where={"id": "x"})

        logger.async_service_failure_hook.assert_called_once()
        call_kwargs = logger.async_service_failure_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_READ
        assert call_kwargs["call_type"] == "litellm_usertable.find_unique"
        assert isinstance(call_kwargs["error"], RuntimeError)

    def test_should_passthrough_non_crud_attributes(self):
        model = _make_mock_model()
        model.some_property = "hello"
        logger = _make_mock_service_logger()
        proxy = PrismaModelProxy(model, "litellm_usertable", logger)

        assert proxy.some_property == "hello"

    @pytest.mark.asyncio
    async def test_should_capture_duration(self):
        async def slow_find(*args, **kwargs):
            await asyncio.sleep(0.05)
            return {"id": "slow"}

        model = _make_mock_model()
        model.find_unique = slow_find
        logger = _make_mock_service_logger()
        proxy = PrismaModelProxy(model, "litellm_usertable", logger)

        await proxy.find_unique(where={"id": "x"})

        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["duration"] >= 0.04


# ---------------------------------------------------------------------------
# _wrap_raw_query_method tests
# ---------------------------------------------------------------------------


class TestWrapRawQueryMethod:
    @pytest.mark.asyncio
    async def test_should_log_query_raw_as_db_read(self):
        logger = _make_mock_service_logger()
        raw_method = AsyncMock(return_value=[{"count": 1}])

        wrapped = _wrap_raw_query_method(
            method=raw_method,
            service_type=ServiceTypes.DB_READ,
            call_type="query_raw",
            service_logger_obj=logger,
        )

        result = await wrapped("SELECT 1")
        assert result == [{"count": 1}]
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_READ
        assert call_kwargs["call_type"] == "query_raw"

    @pytest.mark.asyncio
    async def test_should_log_execute_raw_as_db_write(self):
        logger = _make_mock_service_logger()
        raw_method = AsyncMock(return_value=5)

        wrapped = _wrap_raw_query_method(
            method=raw_method,
            service_type=ServiceTypes.DB_WRITE,
            call_type="execute_raw",
            service_logger_obj=logger,
        )

        result = await wrapped("UPDATE foo SET bar = 1")
        assert result == 5
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_WRITE

    @pytest.mark.asyncio
    async def test_should_log_failure_for_raw_method(self):
        logger = _make_mock_service_logger()
        raw_method = AsyncMock(side_effect=ConnectionError("db unreachable"))

        wrapped = _wrap_raw_query_method(
            method=raw_method,
            service_type=ServiceTypes.DB_READ,
            call_type="query_raw",
            service_logger_obj=logger,
        )

        with pytest.raises(ConnectionError, match="db unreachable"):
            await wrapped("SELECT 1")

        logger.async_service_failure_hook.assert_called_once()
        call_kwargs = logger.async_service_failure_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_READ
        assert isinstance(call_kwargs["error"], ConnectionError)


# ---------------------------------------------------------------------------
# PrismaWrapper instrumentation tests
# ---------------------------------------------------------------------------


class TestPrismaWrapperInstrumentation:
    def _make_wrapper_with_mock_prisma(self):
        """Build a PrismaWrapper with a mock Prisma client and service logger."""
        mock_prisma = MagicMock()

        mock_user_model = MagicMock()
        mock_user_model.find_unique = AsyncMock(return_value={"id": "u1"})
        mock_user_model.create = AsyncMock(return_value={"id": "u1"})
        mock_prisma.litellm_usertable = mock_user_model

        mock_prisma.query_raw = AsyncMock(return_value=[{"count": 42}])
        mock_prisma.query_first = AsyncMock(return_value={"id": "first"})
        mock_prisma.execute_raw = AsyncMock(return_value=3)

        mock_prisma.connect = AsyncMock()
        mock_prisma.disconnect = AsyncMock()
        mock_prisma.batch_ = MagicMock()
        mock_prisma.tx = MagicMock()

        logger = _make_mock_service_logger()

        wrapper = PrismaWrapper(
            original_prisma=mock_prisma,
            iam_token_db_auth=False,
            service_logger_obj=logger,
        )

        return wrapper, mock_prisma, logger

    def test_should_wrap_model_in_proxy(self):
        wrapper, _, _ = self._make_wrapper_with_mock_prisma()
        model_proxy = wrapper.litellm_usertable
        assert isinstance(model_proxy, PrismaModelProxy)

    @pytest.mark.asyncio
    async def test_should_instrument_model_find_unique(self):
        wrapper, mock_prisma, logger = self._make_wrapper_with_mock_prisma()
        result = await wrapper.litellm_usertable.find_unique(where={"id": "u1"})

        assert result == {"id": "u1"}
        mock_prisma.litellm_usertable.find_unique.assert_called_once_with(
            where={"id": "u1"}
        )
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_READ

    @pytest.mark.asyncio
    async def test_should_instrument_model_create(self):
        wrapper, mock_prisma, logger = self._make_wrapper_with_mock_prisma()
        result = await wrapper.litellm_usertable.create(data={"name": "new"})

        assert result == {"id": "u1"}
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_WRITE

    @pytest.mark.asyncio
    async def test_should_wrap_query_raw_as_read(self):
        wrapper, _, logger = self._make_wrapper_with_mock_prisma()
        wrapped = wrapper.query_raw
        result = await wrapped("SELECT 1")

        assert result == [{"count": 42}]
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_READ
        assert call_kwargs["call_type"] == "query_raw"

    @pytest.mark.asyncio
    async def test_should_wrap_execute_raw_as_write(self):
        wrapper, _, logger = self._make_wrapper_with_mock_prisma()
        wrapped = wrapper.execute_raw
        result = await wrapped("DELETE FROM foo")

        assert result == 3
        logger.async_service_success_hook.assert_called_once()
        call_kwargs = logger.async_service_success_hook.call_args[1]
        assert call_kwargs["service"] == ServiceTypes.DB_WRITE
        assert call_kwargs["call_type"] == "execute_raw"

    @pytest.mark.parametrize("attr_name", sorted(_PASSTHROUGH_ATTRS))
    def test_should_passthrough_control_attributes(self, attr_name):
        wrapper, mock_prisma, logger = self._make_wrapper_with_mock_prisma()
        result = getattr(wrapper, attr_name)

        assert result is getattr(mock_prisma, attr_name)
        logger.async_service_success_hook.assert_not_called()

    def test_should_skip_instrumentation_when_no_logger(self):
        mock_prisma = MagicMock()
        mock_model = MagicMock()
        mock_model.find_unique = AsyncMock()
        mock_prisma.litellm_usertable = mock_model

        wrapper = PrismaWrapper(
            original_prisma=mock_prisma,
            iam_token_db_auth=False,
            service_logger_obj=None,
        )

        result = wrapper.litellm_usertable
        assert not isinstance(result, PrismaModelProxy)
        assert result is mock_model
