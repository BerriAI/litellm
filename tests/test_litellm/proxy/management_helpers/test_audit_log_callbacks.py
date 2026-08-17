"""
Tests for audit log callback dispatch.

Tests the flow: create_audit_log_for_update -> _dispatch_audit_log_to_callbacks -> CustomLogger.async_log_audit_log_event
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LiteLLM_AuditLogs, LitellmTableNames, UserAPIKeyAuth
from litellm.proxy.management_helpers.audit_logs import (
    _audit_log_task_done_callback,
    _build_audit_log_payload,
    _dispatch_audit_log_to_callbacks,
    create_audit_log_for_update,
    create_object_audit_log,
)
from litellm.types.utils import StandardAuditLogPayload


@pytest.fixture(autouse=True)
def reset_audit_log_callbacks():
    """Reset audit_log_callbacks before and after each test."""
    original = litellm.audit_log_callbacks
    litellm.audit_log_callbacks = []
    yield
    litellm.audit_log_callbacks = original


def _make_audit_log(
    action: str = "created",
    table_name: LitellmTableNames = LitellmTableNames.TEAM_TABLE_NAME,
) -> LiteLLM_AuditLogs:
    return LiteLLM_AuditLogs(
        id="test-audit-id",
        updated_at=datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc),
        changed_by="user-123",
        changed_by_api_key="sk-abc",
        action=action,
        table_name=table_name,
        object_id="team-456",
        updated_values=json.dumps({"name": "new-team"}),
        before_value=json.dumps({"name": "old-team"}),
    )


class TestBuildAuditLogPayload:
    def test_builds_correct_payload(self):
        audit_log = _make_audit_log()
        payload = _build_audit_log_payload(audit_log)

        assert payload["id"] == "test-audit-id"
        assert payload["updated_at"] == "2026-03-09T12:00:00+00:00"
        assert payload["changed_by"] == "user-123"
        assert payload["changed_by_api_key"] == "sk-abc"
        assert payload["action"] == "created"
        assert payload["table_name"] == "LiteLLM_TeamTable"
        assert payload["object_id"] == "team-456"
        assert payload["updated_values"] == json.dumps({"name": "new-team"})
        assert payload["before_value"] == json.dumps({"name": "old-team"})

    def test_includes_denormalized_alias_keys(self):
        """Callback payloads carry the five alias keys so S3/Datadog/custom sinks receive them."""
        audit_log = _make_audit_log().model_copy(
            update={
                "object_alias": "ml-team",
                "object_team_id": "team-1",
                "object_team_alias": "ml-team",
                "changed_by_user_email": "admin@example.com",
                "changed_by_key_alias": "admin-key",
            }
        )

        payload = _build_audit_log_payload(audit_log)

        assert payload["object_alias"] == "ml-team"
        assert payload["object_team_id"] == "team-1"
        assert payload["object_team_alias"] == "ml-team"
        assert payload["changed_by_user_email"] == "admin@example.com"
        assert payload["changed_by_key_alias"] == "admin-key"

    def test_handles_none_values(self):
        audit_log = LiteLLM_AuditLogs(
            id="test-id",
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            changed_by=None,
            changed_by_api_key=None,
            action="deleted",
            table_name=LitellmTableNames.KEY_TABLE_NAME,
            object_id="key-789",
            updated_values=None,
            before_value=None,
        )
        payload = _build_audit_log_payload(audit_log)

        assert payload["changed_by"] == ""
        assert payload["changed_by_api_key"] == ""
        assert payload["before_value"] is None
        assert payload["updated_values"] is None


class TestDispatchAuditLogToCallbacks:
    @pytest.mark.asyncio
    async def test_dispatches_to_custom_logger_instance(self):
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()
        litellm.audit_log_callbacks = [mock_logger]

        audit_log = _make_audit_log()
        await _dispatch_audit_log_to_callbacks(audit_log)

        # Let asyncio.create_task run
        await asyncio.sleep(0.1)

        mock_logger.async_log_audit_log_event.assert_called_once()
        payload = mock_logger.async_log_audit_log_event.call_args[0][0]
        assert payload["id"] == "test-audit-id"
        assert payload["action"] == "created"

    @pytest.mark.asyncio
    async def test_no_dispatch_when_callbacks_empty(self):
        litellm.audit_log_callbacks = []
        audit_log = _make_audit_log()
        # Should return immediately without error
        await _dispatch_audit_log_to_callbacks(audit_log)

    @pytest.mark.asyncio
    async def test_resolves_string_callback(self):
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()

        litellm.audit_log_callbacks = ["s3_v2"]

        with patch(
            "litellm.proxy.management_helpers.audit_logs._resolve_audit_log_callback",
            return_value=mock_logger,
        ):
            audit_log = _make_audit_log()
            await _dispatch_audit_log_to_callbacks(audit_log)
            await asyncio.sleep(0.1)

            mock_logger.async_log_audit_log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_nonblocking_on_callback_failure(self):
        """Callback errors should not propagate."""
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock(side_effect=RuntimeError("boom"))
        litellm.audit_log_callbacks = [mock_logger]

        audit_log = _make_audit_log()
        # Should not raise
        await _dispatch_audit_log_to_callbacks(audit_log)
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_skips_unresolvable_string_callback(self):
        litellm.audit_log_callbacks = ["nonexistent_callback"]

        with patch(
            "litellm.proxy.management_helpers.audit_logs._resolve_audit_log_callback",
            return_value=None,
        ):
            audit_log = _make_audit_log()
            # Should not raise
            await _dispatch_audit_log_to_callbacks(audit_log)


class TestCreateAuditLogForUpdateWithCallbacks:
    @pytest.mark.asyncio
    async def test_dispatches_to_callbacks_after_db_write(self):
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()
        litellm.audit_log_callbacks = [mock_logger]

        with (
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.store_audit_logs", True),
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        ):
            mock_prisma.db.litellm_auditlog.create = AsyncMock()

            audit_log = _make_audit_log()
            await create_audit_log_for_update(audit_log)
            await asyncio.sleep(0.1)

            # DB write should happen
            mock_prisma.db.litellm_auditlog.create.assert_called_once()
            # Callback should also be called
            mock_logger.async_log_audit_log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_dispatch_when_not_premium(self):
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()
        litellm.audit_log_callbacks = [mock_logger]

        with (
            patch("litellm.proxy.proxy_server.premium_user", False),
            patch("litellm.store_audit_logs", True),
        ):
            audit_log = _make_audit_log()
            await create_audit_log_for_update(audit_log)
            await asyncio.sleep(0.1)

            mock_logger.async_log_audit_log_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_dispatch_when_store_audit_logs_false(self):
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()
        litellm.audit_log_callbacks = [mock_logger]

        with patch("litellm.store_audit_logs", False):
            audit_log = _make_audit_log()
            await create_audit_log_for_update(audit_log)
            await asyncio.sleep(0.1)

            mock_logger.async_log_audit_log_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_even_when_prisma_client_is_none(self):
        """Callbacks should fire even if DB is unavailable."""
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()
        litellm.audit_log_callbacks = [mock_logger]

        with (
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.store_audit_logs", True),
            patch("litellm.proxy.proxy_server.prisma_client", None),
        ):
            audit_log = _make_audit_log()
            await create_audit_log_for_update(audit_log)
            await asyncio.sleep(0.1)

            # Callback should still be called despite no DB
            mock_logger.async_log_audit_log_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_even_when_db_write_fails(self):
        """Callbacks should fire even if the DB write raises."""
        mock_logger = MagicMock(spec=CustomLogger)
        mock_logger.async_log_audit_log_event = AsyncMock()
        litellm.audit_log_callbacks = [mock_logger]

        with (
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.store_audit_logs", True),
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        ):
            mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=RuntimeError("DB connection lost"))

            audit_log = _make_audit_log()
            await create_audit_log_for_update(audit_log)
            await asyncio.sleep(0.1)

            # Callback should still be called despite DB failure
            mock_logger.async_log_audit_log_event.assert_called_once()


class TestAuditLogTaskDoneCallback:
    def test_logs_exception_from_failed_task(self):
        """Done callback should log task exceptions."""
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.exception.return_value = RuntimeError("callback failed")

        with patch("litellm.proxy.management_helpers.audit_logs.verbose_proxy_logger") as mock_logger:
            _audit_log_task_done_callback(mock_task)
            mock_logger.error.assert_called_once()
            assert "callback failed" in str(mock_logger.error.call_args)

    def test_no_log_on_success(self):
        """Done callback should not log when task succeeds."""
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.exception.return_value = None

        with patch("litellm.proxy.management_helpers.audit_logs.verbose_proxy_logger") as mock_logger:
            _audit_log_task_done_callback(mock_task)
            mock_logger.error.assert_not_called()

    def test_handles_cancelled_task(self):
        """Done callback should handle cancelled tasks gracefully."""
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.exception.side_effect = asyncio.CancelledError()

        with patch("litellm.proxy.management_helpers.audit_logs.verbose_proxy_logger") as mock_logger:
            _audit_log_task_done_callback(mock_task)
            mock_logger.error.assert_not_called()


class TestS3LoggerAuditLogEvent:
    @pytest.mark.asyncio
    async def test_queues_audit_log_with_correct_s3_key(self):
        with patch("litellm.integrations.s3_v2.S3Logger.__init__", return_value=None):
            from litellm.integrations.s3_v2 import S3Logger

            logger = S3Logger()
            logger.s3_path = "my-prefix"
            logger.log_queue = []
            logger.batch_size = 100

            audit_log = StandardAuditLogPayload(
                id="audit-123",
                updated_at="2026-03-09T12:00:00+00:00",
                changed_by="user-1",
                changed_by_api_key="sk-abc",
                action="created",
                table_name="LiteLLM_TeamTable",
                object_id="team-1",
                before_value=None,
                updated_values='{"name": "new"}',
            )

            await logger.async_log_audit_log_event(audit_log)

            assert len(logger.log_queue) == 1
            element = logger.log_queue[0]
            assert element.s3_object_key.startswith("my-prefix/audit_logs/")
            assert "audit-123" in element.s3_object_key
            assert element.s3_object_key.endswith(".json")
            assert element.s3_object_download_filename == "audit-audit-123.json"
            assert element.payload["id"] == "audit-123"
            assert element.payload["action"] == "created"

    @pytest.mark.asyncio
    async def test_s3_key_format_no_path(self):
        with patch("litellm.integrations.s3_v2.S3Logger.__init__", return_value=None):
            from litellm.integrations.s3_v2 import S3Logger

            logger = S3Logger()
            logger.s3_path = None
            logger.log_queue = []
            logger.batch_size = 100

            audit_log = StandardAuditLogPayload(
                id="audit-456",
                updated_at="2026-03-09T12:00:00+00:00",
                changed_by="user-1",
                changed_by_api_key="sk-abc",
                action="deleted",
                table_name="LiteLLM_VerificationToken",
                object_id="key-1",
                before_value=None,
                updated_values=None,
            )

            await logger.async_log_audit_log_event(audit_log)

            assert len(logger.log_queue) == 1
            element = logger.log_queue[0]
            assert element.s3_object_key.startswith("audit_logs/")
            assert "audit-456" in element.s3_object_key


class TestS3AuditCallbackParamsDecoupling:
    """`s3_audit_callback_params` should give the audit-log path its own
    S3Logger instance, distinct from the singleton serving normal logs."""

    @pytest.fixture(autouse=True)
    def _isolate_caches_and_globals(self):
        from litellm.litellm_core_utils import litellm_logging as ll_logging
        from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

        original_s3 = litellm.s3_callback_params
        original_audit = getattr(litellm, "s3_audit_callback_params", None)
        ll_audit_logs._audit_log_callback_cache.clear()
        ll_logging._in_memory_loggers.clear()
        yield
        litellm.s3_callback_params = original_s3
        litellm.s3_audit_callback_params = original_audit
        ll_audit_logs._audit_log_callback_cache.clear()
        ll_logging._in_memory_loggers.clear()

    def test_opt_in_constructs_separate_instance_with_audit_config(self):
        """Audit config set → audit resolver returns a fresh S3Logger pointing
        at the audit bucket, distinct from the normal-log singleton."""
        from litellm.integrations.s3_v2 import S3Logger
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )
        from litellm.proxy.management_helpers.audit_logs import (
            _resolve_audit_log_callback,
        )

        litellm.s3_callback_params = {"s3_bucket_name": "normal-bucket"}
        litellm.s3_audit_callback_params = {"s3_bucket_name": "audit-bucket"}

        with patch("asyncio.create_task"):
            audit_instance = _resolve_audit_log_callback("s3_v2")
            normal_instance = _init_custom_logger_compatible_class(
                logging_integration="s3_v2",
                internal_usage_cache=None,
                llm_router=None,
            )

        assert isinstance(audit_instance, S3Logger)
        assert isinstance(normal_instance, S3Logger)
        assert id(audit_instance) != id(normal_instance)
        assert audit_instance.s3_bucket_name == "audit-bucket"
        assert normal_instance.s3_bucket_name == "normal-bucket"

    def test_opt_out_preserves_singleton_behavior(self):
        """No `s3_audit_callback_params` → audit and normal share the singleton
        (existing behavior, regression guard)."""
        from litellm.integrations.s3_v2 import S3Logger
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )
        from litellm.proxy.management_helpers.audit_logs import (
            _resolve_audit_log_callback,
        )

        litellm.s3_callback_params = {"s3_bucket_name": "shared-bucket"}
        litellm.s3_audit_callback_params = None

        with patch("asyncio.create_task"):
            normal_instance = _init_custom_logger_compatible_class(
                logging_integration="s3_v2",
                internal_usage_cache=None,
                llm_router=None,
            )
            audit_instance = _resolve_audit_log_callback("s3_v2")

        assert isinstance(audit_instance, S3Logger)
        assert id(audit_instance) == id(normal_instance)
        assert audit_instance.s3_bucket_name == "shared-bucket"

    def test_empty_dict_opts_in(self):
        """`s3_audit_callback_params = {}` is opt-in (truthy-by-presence) and
        produces a separate instance with no bucket configured (env/IAM-only)."""
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )
        from litellm.proxy.management_helpers.audit_logs import (
            _resolve_audit_log_callback,
        )

        litellm.s3_callback_params = {"s3_bucket_name": "normal-bucket"}
        litellm.s3_audit_callback_params = {}

        with patch("asyncio.create_task"):
            audit_instance = _resolve_audit_log_callback("s3_v2")
            normal_instance = _init_custom_logger_compatible_class(
                logging_integration="s3_v2",
                internal_usage_cache=None,
                llm_router=None,
            )

        assert id(audit_instance) != id(normal_instance)
        assert audit_instance.s3_bucket_name is None
        assert normal_instance.s3_bucket_name == "normal-bucket"

    def test_reset_audit_log_callback_cache_clears_audit_instance(self):
        """`reset_audit_log_callback_cache()` must drop the cached audit
        instance so a config reload picks up the new params."""
        from litellm.proxy.management_helpers.audit_logs import (
            _audit_log_callback_cache,
            _resolve_audit_log_callback,
            reset_audit_log_callback_cache,
        )

        litellm.s3_audit_callback_params = {"s3_bucket_name": "first"}
        with patch("asyncio.create_task"):
            first = _resolve_audit_log_callback("s3_v2")
            assert first is not None and "s3_v2" in _audit_log_callback_cache

            reset_audit_log_callback_cache()
            assert "s3_v2" not in _audit_log_callback_cache

            litellm.s3_audit_callback_params = {"s3_bucket_name": "second"}
            second = _resolve_audit_log_callback("s3_v2")
            assert second is not None
            assert id(second) != id(first)
            assert second.s3_bucket_name == "second"


class _FakeFindUniqueTable:
    def __init__(self, row=None):
        self.row = row
        self.where_calls = []

    async def find_unique(self, where):
        self.where_calls.append(where)
        return self.row


class _FakeAuditCreateTable:
    def __init__(self):
        self.created = []

    async def create(self, data):
        self.created.append(data)


class _FakeDb:
    def __init__(self, team_row=None, user_row=None, key_row=None):
        self.litellm_auditlog = _FakeAuditCreateTable()
        self.litellm_teamtable = _FakeFindUniqueTable(team_row)
        self.litellm_usertable = _FakeFindUniqueTable(user_row)
        self.litellm_verificationtoken = _FakeFindUniqueTable(key_row)


class _FakePrismaClient:
    def __init__(self, db):
        self.db = db


def _gates(fake_prisma):
    return (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma),
    )


class TestAuditLogAliasDenormalization:
    @pytest.mark.asyncio
    async def test_team_delete_denormalizes_from_blob_without_lookups(self):
        """A team delete row gets object_alias/object_team_id/object_team_alias straight from its blob."""
        db = _FakeDb(
            user_row=SimpleNamespace(user_email="admin@example.com"),
            key_row=SimpleNamespace(key_alias="admin-key", user_id="admin-user"),
        )
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a1",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    changed_by="admin-user",
                    changed_by_api_key="hash-admin",
                    action="deleted",
                    table_name=LitellmTableNames.TEAM_TABLE_NAME,
                    object_id="team-1",
                    before_value=json.dumps({"team_id": "team-1", "team_alias": "ml-team"}),
                )
            )

        data = db.litellm_auditlog.created[0]
        assert data["object_alias"] == "ml-team"
        assert data["object_team_id"] == "team-1"
        assert data["object_team_alias"] == "ml-team"
        assert data["changed_by_user_email"] == "admin@example.com"
        assert data["changed_by_key_alias"] == "admin-key"
        assert db.litellm_teamtable.where_calls == []
        assert db.litellm_usertable.where_calls == [{"user_id": "admin-user"}]
        assert db.litellm_verificationtoken.where_calls == [{"token": "hash-admin"}]

    @pytest.mark.asyncio
    async def test_key_create_resolves_aliases_via_lookup_not_masked_blob(self):
        """Key rows resolve object_alias through the key table because the blob's key_alias is masked,
        and resolve object_team_alias through the team table from the blob's team_id."""
        db = _FakeDb(
            team_row=SimpleNamespace(team_alias="team-nine"),
            key_row=SimpleNamespace(key_alias="prod-key", user_id=None),
        )
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a2",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    action="created",
                    table_name=LitellmTableNames.KEY_TABLE_NAME,
                    object_id="hash-1",
                    updated_values=json.dumps({"key_alias": "prod-key", "team_id": "team-9"}),
                )
            )

        data = db.litellm_auditlog.created[0]
        assert json.loads(data["updated_values"])["key_alias"] == "********"
        assert data["object_alias"] == "prod-key"
        assert data["object_team_id"] == "team-9"
        assert data["object_team_alias"] == "team-nine"
        assert db.litellm_verificationtoken.where_calls == [{"token": "hash-1"}]
        assert db.litellm_teamtable.where_calls == [{"team_id": "team-9"}]

    @pytest.mark.asyncio
    async def test_none_actor_fields_fall_through_to_credential_confirmed_lookup(self):
        """A None from the auth context still falls through to the lookup, but the email side only
        resolves through the credential-confirmed gate: a service key with no owning user gets one
        key lookup and never a user lookup."""
        db = _FakeDb(
            user_row=SimpleNamespace(user_email="someone@example.com"),
            key_row=SimpleNamespace(key_alias=None, user_id=None),
        )
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a2b",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    changed_by="service-account",
                    changed_by_api_key="hash-service",
                    action="updated",
                    table_name=LitellmTableNames.TEAM_TABLE_NAME,
                    object_id="team-1",
                    updated_values=json.dumps({"team_alias": "ml-team"}),
                    changed_by_user_email=None,
                    changed_by_key_alias=None,
                )
            )

        data = db.litellm_auditlog.created[0]
        assert "changed_by_user_email" not in data
        assert "changed_by_key_alias" not in data
        assert db.litellm_verificationtoken.where_calls == [{"token": "hash-service"}]
        assert db.litellm_usertable.where_calls == []

    @pytest.mark.asyncio
    async def test_spoofed_changed_by_is_not_resolved_to_an_email(self):
        """A changed_by that is not the credential's own user (litellm-changed-by header) is never
        decorated with a real user's email."""
        db = _FakeDb(
            user_row=SimpleNamespace(user_email="victim@example.com"),
            key_row=SimpleNamespace(key_alias="attacker-key", user_id="attacker-user"),
        )
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a2c",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    changed_by="victim-user",
                    changed_by_api_key="hash-attacker",
                    action="updated",
                    table_name=LitellmTableNames.TEAM_TABLE_NAME,
                    object_id="team-1",
                    updated_values=json.dumps({"team_alias": "ml-team"}),
                )
            )

        data = db.litellm_auditlog.created[0]
        assert "changed_by_user_email" not in data
        assert data["changed_by_key_alias"] == "attacker-key"
        assert db.litellm_usertable.where_calls == []

    @pytest.mark.asyncio
    async def test_user_row_prefers_alias_then_email_updated_blob_first(self):
        """User rows derive object_alias as user_alias then user_email, updated_values before before_value."""
        db = _FakeDb()
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a3",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    action="updated",
                    table_name=LitellmTableNames.USER_TABLE_NAME,
                    object_id="user-1",
                    before_value=json.dumps({"user_alias": "Old Alias", "user_email": "old@example.com"}),
                    updated_values=json.dumps({"user_email": "new@example.com"}),
                )
            )

        assert db.litellm_auditlog.created[0]["object_alias"] == "new@example.com"

    @pytest.mark.asyncio
    async def test_caller_provided_actor_fields_skip_lookups(self):
        """Actor email and key alias passed by the caller are kept verbatim with no DB lookups."""
        db = _FakeDb(
            user_row=SimpleNamespace(user_email="wrong@example.com"),
            key_row=SimpleNamespace(key_alias="wrong-key", user_id="admin-user"),
        )
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a4",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    changed_by="admin-user",
                    changed_by_api_key="hash-admin",
                    action="updated",
                    table_name=LitellmTableNames.TEAM_TABLE_NAME,
                    object_id="team-1",
                    updated_values=json.dumps({"team_alias": "ml-team"}),
                    changed_by_user_email="right@example.com",
                    changed_by_key_alias="right-key",
                )
            )

        data = db.litellm_auditlog.created[0]
        assert data["changed_by_user_email"] == "right@example.com"
        assert data["changed_by_key_alias"] == "right-key"
        assert db.litellm_usertable.where_calls == []
        assert db.litellm_verificationtoken.where_calls == []

    @pytest.mark.asyncio
    async def test_deleted_actor_stays_absent(self):
        """When actor lookups miss, the actor columns are left unset rather than written as empty."""
        db = _FakeDb()
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_audit_log_for_update(
                LiteLLM_AuditLogs(
                    id="a5",
                    updated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    changed_by="gone-user",
                    changed_by_api_key="gone-hash",
                    action="deleted",
                    table_name=LitellmTableNames.TEAM_TABLE_NAME,
                    object_id="team-1",
                    before_value=json.dumps({"team_id": "team-1", "team_alias": "ml-team"}),
                )
            )

        data = db.litellm_auditlog.created[0]
        assert "changed_by_user_email" not in data
        assert "changed_by_key_alias" not in data
        assert data["object_alias"] == "ml-team"

    @pytest.mark.asyncio
    async def test_create_object_audit_log_uses_auth_context(self):
        """create_object_audit_log stamps actor email and key alias from UserAPIKeyAuth without lookups."""
        db = _FakeDb()
        p1, p2, p3 = _gates(_FakePrismaClient(db))
        with p1, p2, p3:
            await create_object_audit_log(
                object_id="team-1",
                action="updated",
                litellm_changed_by=None,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="hash-admin",
                    user_id="admin-user",
                    user_email="admin@example.com",
                    key_alias="admin-key",
                ),
                litellm_proxy_admin_name="default_user_id",
                table_name=LitellmTableNames.TEAM_TABLE_NAME,
                after_value=json.dumps({"team_id": "team-1", "team_alias": "ml-team"}),
            )

        data = db.litellm_auditlog.created[0]
        assert data["changed_by"] == "admin-user"
        assert data["changed_by_user_email"] == "admin@example.com"
        assert data["changed_by_key_alias"] == "admin-key"
        assert data["object_alias"] == "ml-team"
        assert db.litellm_usertable.where_calls == []
        assert db.litellm_verificationtoken.where_calls == []
