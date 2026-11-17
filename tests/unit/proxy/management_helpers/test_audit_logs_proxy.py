import os
import traceback
from litellm._uuid import uuid
from datetime import datetime, timezone
from collections.abc import Coroutine
from typing import Any, Final

from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute


import contextlib
import io
import logging
import time

# this file is to test litellm/proxy

import asyncio

load_dotenv()

import pytest
import litellm

from litellm.proxy.proxy_server import (
    LitellmUserRoles,
    audio_transcriptions,
    chat_completion,
    completion,
    embeddings,
    model_list,
    moderations,
    user_api_key_auth,
)

from litellm.proxy.utils import PrismaClient, ProxyLogging, hash_token, update_spend


from starlette.datastructures import URL

from litellm.proxy.management_helpers.audit_logs import (
    create_audit_log_for_update,
    drain_audit_tasks,
    get_audit_log_changed_by,
    track_audit_task,
)
from litellm.proxy._types import LiteLLM_AuditLogs, LitellmTableNames, UserAPIKeyAuth
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardAuditLogPayload
from unittest.mock import AsyncMock, MagicMock, patch

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
import json
from tests._master_key import MASTER_KEY


def test_get_audit_log_changed_by_prefers_authenticated_user():
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="authenticated-user",
    )

    assert (
        get_audit_log_changed_by(
            litellm_changed_by="spoofed-user",
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name="proxy-admin",
        )
        == "authenticated-user"
    )


def test_get_audit_log_changed_by_honors_header_with_admin_opt_in():
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="service-account",
        metadata={"allow_litellm_changed_by_header": True},
    )

    assert (
        get_audit_log_changed_by(
            litellm_changed_by="delegated-user",
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name="proxy-admin",
        )
        == "delegated-user"
    )


def test_get_audit_log_changed_by_honors_header_with_team_opt_in():
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="service-account",
        team_metadata={"allow_litellm_changed_by_header": True},
    )

    assert (
        get_audit_log_changed_by(
            litellm_changed_by="delegated-user",
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name="proxy-admin",
        )
        == "delegated-user"
    )


def test_get_audit_log_changed_by_ignores_header_without_opt_in_when_user_id_missing():
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    assert (
        get_audit_log_changed_by(
            litellm_changed_by="spoofed-user",
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name="proxy-admin",
        )
        == "proxy-admin"
    )


def test_get_audit_log_changed_by_honors_header_with_opt_in_when_user_id_missing():
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"allow_litellm_changed_by_header": True},
    )

    assert (
        get_audit_log_changed_by(
            litellm_changed_by="delegated-user",
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name="proxy-admin",
        )
        == "delegated-user"
    )


@pytest.mark.asyncio
async def test_create_internal_user_audit_log_uses_changed_by_helper():
    from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="service-account",
        metadata={"allow_litellm_changed_by_header": True},
    )

    with (
        patch("litellm.store_audit_logs", True),
        patch(
            "litellm.proxy.hooks.user_management_event_hooks.create_audit_log_for_update",
            new_callable=AsyncMock,
        ) as mock_create_audit_log_for_update,
    ):
        await UserManagementEventHooks.create_internal_user_audit_log(
            user_id="target-user",
            action="updated",
            litellm_changed_by="delegated-user",
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name="proxy-admin",
            before_value='{"before": true}',
            after_value='{"after": true}',
        )

    request_data = mock_create_audit_log_for_update.await_args.kwargs["request_data"]
    assert request_data.changed_by == "delegated-user"
    assert request_data.changed_by_api_key == "test-key"
    assert request_data.object_id == "target-user"
    assert request_data.action == "updated"


@pytest.mark.asyncio
async def test_create_audit_log_for_update_premium_user():
    """
    Basic unit test for create_audit_log_for_update

    Test that the audit log is created when a premium user updates a team
    """
    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
    ):
        mock_prisma.db.litellm_auditlog.create = AsyncMock()

        request_data = LiteLLM_AuditLogs(
            id="test_id",
            updated_at=datetime.now(),
            changed_by="test_changed_by",
            action="updated",
            table_name=LitellmTableNames.TEAM_TABLE_NAME,
            object_id="test_object_id",
            updated_values=json.dumps({"key": "value"}),
            before_value=json.dumps({"old_key": "old_value"}),
        )

        await create_audit_log_for_update(request_data)

        mock_prisma.db.litellm_auditlog.create.assert_called_once_with(
            data={
                "id": "test_id",
                "updated_at": request_data.updated_at,
                "changed_by": request_data.changed_by,
                "action": request_data.action,
                "table_name": request_data.table_name,
                "object_id": request_data.object_id,
                "updated_values": request_data.updated_values,
                "before_value": request_data.before_value,
            }
        )


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming PrismaClient is a class that needs to be instantiated
    prisma_client = PrismaClient(database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj)

    return prisma_client


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@pytest.mark.asyncio()
async def test_create_audit_log_in_db(prisma_client):
    print("prisma client=", prisma_client)

    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    setattr(litellm.proxy.proxy_server, "premium_user", True)
    setattr(litellm, "store_audit_logs", True)

    await litellm.proxy.proxy_server.prisma_client.connect()
    audit_log_id = f"audit_log_id_{uuid.uuid4()}"

    # create a audit log for /key/generate
    request_data = LiteLLM_AuditLogs(
        id=audit_log_id,
        updated_at=datetime.now(),
        changed_by="test_changed_by",
        action="updated",
        table_name=LitellmTableNames.TEAM_TABLE_NAME,
        object_id="test_object_id",
        updated_values=json.dumps({"key": "value"}),
        before_value=json.dumps({"old_key": "old_value"}),
    )

    await create_audit_log_for_update(request_data)

    await asyncio.sleep(1)

    # now read the last log from the db
    last_log = await prisma_client.db.litellm_auditlog.find_first(where={"id": audit_log_id})

    assert last_log.id == audit_log_id

    setattr(litellm, "store_audit_logs", False)


@pytest.mark.asyncio
async def test_track_audit_task_lifecycle():
    task_completed: Final = asyncio.Event()

    async def _sample_coroutine() -> None:
        await asyncio.sleep(0)
        task_completed.set()

    task: Final = track_audit_task(asyncio.create_task(_sample_coroutine()))
    assert not task.done()

    await drain_audit_tasks(timeout=1.0)
    assert task_completed.is_set()
    assert task.done()

    completed_task: Final = asyncio.create_task(asyncio.sleep(0))
    await completed_task
    assert track_audit_task(completed_task) is completed_task


@pytest.mark.asyncio
async def test_drain_audit_tasks_waits_for_all_tasks():
    event1: Final = asyncio.Event()
    event2: Final = asyncio.Event()

    async def _worker(event: asyncio.Event) -> None:
        await asyncio.sleep(0)
        event.set()

    task1: Final = track_audit_task(asyncio.create_task(_worker(event1)))
    task2: Final = track_audit_task(asyncio.create_task(_worker(event2)))

    assert not task1.done()
    assert not task2.done()

    await drain_audit_tasks(timeout=1.0)

    assert event1.is_set()
    assert event2.is_set()
    assert task1.done()
    assert task2.done()


@pytest.mark.asyncio
async def test_drain_audit_tasks_handles_failing_task_cleanly():
    async def _failing_worker() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("test audit log write failure")

    task: Final = track_audit_task(asyncio.create_task(_failing_worker()))
    assert not task.done()

    await drain_audit_tasks(timeout=1.0)
    assert task.done()
    assert isinstance(task.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_drain_audit_tasks_timeout_does_not_cancel_pending_tasks():
    async def _long_running_worker() -> None:
        await asyncio.Event().wait()

    task: Final = track_audit_task(asyncio.create_task(_long_running_worker()))
    try:
        assert not task.done()
        await drain_audit_tasks(timeout=0)
        assert not task.cancelled()
        assert not task.done()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_drain_audit_tasks_persists_management_audit_write_before_db_disconnect():
    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []
    is_disconnected: Final = [False]

    async def _mock_create(data: dict[str, Any]) -> None:
        if is_disconnected[0]:
            raise RuntimeError("Database already disconnected")
        await asyncio.sleep(0)
        writes.append(data)

    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=_mock_create)

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):
        request_data: Final = LiteLLM_AuditLogs(
            id=str(uuid.uuid4()),
            updated_at=datetime.now(timezone.utc),
            changed_by="admin-user",
            changed_by_api_key="sk-admin",
            table_name=LitellmTableNames.KEY_TABLE_NAME,
            object_id="token-123",
            action="blocked",
            updated_values="{}",
            before_value='{"key": "test"}',
        )

        task: Final = track_audit_task(asyncio.create_task(create_audit_log_for_update(request_data=request_data)))
        assert not task.done()

        await drain_audit_tasks(timeout=2.0)
        is_disconnected[0] = True

        assert task.done()
        assert len(writes) == 1
        assert writes[0]["object_id"] == "token-123"
        assert writes[0]["action"] == "blocked"


@pytest.mark.asyncio
async def test_create_audit_log_auto_registers_in_drain():
    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []

    async def _mock_create(data: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        writes.append(data)

    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=_mock_create)

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):
        request_data: Final = LiteLLM_AuditLogs(
            id=str(uuid.uuid4()),
            updated_at=datetime.now(timezone.utc),
            changed_by="admin-user",
            changed_by_api_key="sk-admin",
            table_name=LitellmTableNames.KEY_TABLE_NAME,
            object_id="token-456",
            action="deleted",
            updated_values="{}",
            before_value='{"key": "test-deleted"}',
        )

        task: Final = asyncio.create_task(create_audit_log_for_update(request_data=request_data))
        await asyncio.sleep(0)

        await drain_audit_tasks(timeout=2.0)

        assert task.done()
        assert len(writes) == 1
        assert writes[0]["object_id"] == "token-456"
        assert writes[0]["action"] == "deleted"


@pytest.mark.asyncio
async def test_drain_audit_tasks_discards_timed_out_tasks_on_timeout():
    async def _hung_worker() -> None:
        await asyncio.Event().wait()

    hung_task: Final = track_audit_task(asyncio.create_task(_hung_worker()))
    try:
        await drain_audit_tasks(timeout=0)
        assert not hung_task.done()

        second_drain: Final = asyncio.create_task(drain_audit_tasks(timeout=60))
        for _ in range(3):
            await asyncio.sleep(0)
        assert second_drain.done()
    finally:
        hung_task.cancel()
        try:
            await hung_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_drain_audit_tasks_captures_tasks_queued_during_drain():
    finished_tasks: Final[list[str]] = []

    async def _first_worker() -> None:
        await asyncio.sleep(0)
        track_audit_task(asyncio.create_task(_second_worker()))
        finished_tasks.append("first")

    async def _second_worker() -> None:
        await asyncio.sleep(0)
        finished_tasks.append("second")

    track_audit_task(asyncio.create_task(_first_worker()))
    await drain_audit_tasks(timeout=2.0)

    assert finished_tasks == ["first", "second"]


class _AuditCallbackBlockedUntilReleased(CustomLogger):
    def __init__(self, release: asyncio.Event) -> None:
        super().__init__()
        self.release: Final = release
        self.logged_object_ids: Final[list[str]] = []

    async def async_log_audit_log_event(self, audit_log: StandardAuditLogPayload) -> None:
        await self.release.wait()
        self.logged_object_ids.append(audit_log["object_id"])


class _WarningRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: Final[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _deleted_key_audit_log(object_id: str) -> LiteLLM_AuditLogs:
    return LiteLLM_AuditLogs(
        id=str(uuid.uuid4()),
        updated_at=datetime.now(timezone.utc),
        changed_by="admin-user",
        table_name=LitellmTableNames.KEY_TABLE_NAME,
        object_id=object_id,
        action="deleted",
        updated_values="{}",
        before_value="{}",
    )


@pytest.mark.asyncio
async def test_drain_waits_for_a_slow_audit_log_callback():
    release: Final = asyncio.Event()
    callback: Final = _AuditCallbackBlockedUntilReleased(release)

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.audit_log_callbacks", [callback]),
    ):
        await create_audit_log_for_update(request_data=_deleted_key_audit_log("token-callback"))
        drain: Final = asyncio.create_task(drain_audit_tasks(timeout=60))
        for _ in range(3):
            await asyncio.sleep(0)
        drain_returned_before_the_callback_finished: Final = drain.done()
        release.set()
        await drain

    assert not drain_returned_before_the_callback_finished
    assert callback.logged_object_ids == ["token-callback"]


@pytest.mark.asyncio
async def test_drain_timeout_warning_names_each_dropped_audit_row():
    mock_prisma: Final = MagicMock()
    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=_io_that_never_finishes)
    recorder: Final = _WarningRecorder()
    verbose_proxy_logger.addHandler(recorder)

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):
        writer: Final = asyncio.create_task(
            create_audit_log_for_update(request_data=_deleted_key_audit_log("token-dropped"))
        )
        try:
            for _ in range(3):
                await asyncio.sleep(0)
            await drain_audit_tasks(timeout=0.05)
        finally:
            verbose_proxy_logger.removeHandler(recorder)
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer

    assert recorder.messages == [
        "Timed out draining 1 audit tasks on shutdown, dropping: "
        f"deleted {LitellmTableNames.KEY_TABLE_NAME.value} token-dropped to the database"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_audit_logs", "premium_user", "expected_write_count"),
    [(True, True, 1), (False, True, 0), (True, False, 0)],
    ids=["row_written", "logging_disabled", "not_premium"],
)
async def test_drain_does_not_wait_for_code_that_runs_after_an_inline_audit_write(
    store_audit_logs: bool, premium_user: bool, expected_write_count: int
):
    release_caller: Final = asyncio.Event()
    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []
    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=lambda data: writes.append(data))

    async def _caller_that_outlives_the_audit_call() -> None:
        await create_audit_log_for_update(
            request_data=LiteLLM_AuditLogs(
                id=str(uuid.uuid4()),
                updated_at=datetime.now(timezone.utc),
                changed_by="test-user",
                table_name=LitellmTableNames.KEY_TABLE_NAME,
                object_id="test-obj",
                action="updated",
                updated_values="{}",
                before_value="{}",
            )
        )
        await release_caller.wait()

    with (
        patch("litellm.store_audit_logs", store_audit_logs),
        patch("litellm.proxy.proxy_server.premium_user", premium_user),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):
        caller: Final = asyncio.create_task(_caller_that_outlives_the_audit_call())
        for _ in range(3):
            await asyncio.sleep(0)
        drain: Final = asyncio.create_task(drain_audit_tasks(timeout=60))
        for _ in range(3):
            await asyncio.sleep(0)
        drain_returned_while_caller_was_running: Final = drain.done()
        release_caller.set()
        await caller
        await drain

    assert drain_returned_while_caller_was_running
    assert len(writes) == expected_write_count


async def _io_that_never_finishes(*args: object, **kwargs: object) -> None:
    await asyncio.Event().wait()


async def _drain_while_hook_io_hangs(hook: Coroutine[object, object, None]) -> None:
    hook_task: Final = asyncio.create_task(hook)
    for _ in range(3):
        await asyncio.sleep(0)
    await drain_audit_tasks(timeout=2.0)
    hook_task.cancel()


@pytest.mark.asyncio
async def test_key_generated_audit_write_is_drained_while_the_invite_email_hangs():
    from litellm.proxy._types import GenerateKeyRequest, GenerateKeyResponse
    from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks

    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []
    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=lambda data: writes.append(data))

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(KeyManagementEventHooks, "_send_key_created_email", side_effect=_io_that_never_finishes),
    ):
        await _drain_while_hook_io_hangs(
            KeyManagementEventHooks.async_key_generated_hook(
                data=GenerateKeyRequest(send_invite_email=True),
                response=GenerateKeyResponse(key="sk-new", token_id="hashed-new"),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_id="admin"),
            )
        )

    assert [(write["object_id"], write["action"]) for write in writes] == [("hashed-new", "created")]


@pytest.mark.asyncio
async def test_key_updated_audit_write_is_drained_while_the_secret_manager_rename_hangs():
    from litellm.proxy._types import LiteLLM_VerificationToken, UpdateKeyRequest
    from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks

    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []
    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=lambda data: writes.append(data))

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(
            KeyManagementEventHooks, "_rename_virtual_key_in_secret_manager", side_effect=_io_that_never_finishes
        ),
    ):
        await _drain_while_hook_io_hangs(
            KeyManagementEventHooks.async_key_updated_hook(
                data=UpdateKeyRequest(key="sk-existing", key_alias="new-alias"),
                existing_key_row=LiteLLM_VerificationToken(token="hashed-existing", key_alias="old-alias"),
                response=None,
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_id="admin"),
            )
        )

    assert [(write["object_id"], write["action"]) for write in writes] == [(hash_token("sk-existing"), "updated")]


@pytest.mark.asyncio
async def test_key_rotated_audit_write_is_drained_while_the_secret_manager_rotation_hangs():
    from litellm.proxy._types import GenerateKeyResponse, LiteLLM_VerificationToken
    from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks

    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []
    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=lambda data: writes.append(data))

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(
            KeyManagementEventHooks, "_rotate_virtual_key_in_secret_manager", side_effect=_io_that_never_finishes
        ),
    ):
        await _drain_while_hook_io_hangs(
            KeyManagementEventHooks.async_key_rotated_hook(
                data=None,
                existing_key_row=LiteLLM_VerificationToken(token="hashed-old"),
                response=GenerateKeyResponse(key="sk-rotated", token_id="hashed-new"),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_id="admin"),
            )
        )

    assert [(write["object_id"], write["action"]) for write in writes] == [("hashed-old", "rotated")]


@pytest.mark.asyncio
async def test_user_created_audit_write_is_drained_while_its_user_read_is_still_running():
    from litellm.proxy._types import NewUserRequest, NewUserResponse
    from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks

    mock_prisma: Final = MagicMock()
    writes: Final[list[dict[str, Any]]] = []
    release_user_read: Final = asyncio.Event()

    async def _slow_user_read(*args: object, **kwargs: object) -> dict[str, str]:
        await release_user_read.wait()
        return {"user_id": "new-user"}

    mock_prisma.db.litellm_auditlog.create = AsyncMock(side_effect=lambda data: writes.append(data))
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(side_effect=_slow_user_read)

    with (
        patch("litellm.store_audit_logs", True),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(UserManagementEventHooks, "async_send_user_invitation_email", side_effect=_io_that_never_finishes),
    ):
        hook: Final = asyncio.create_task(
            UserManagementEventHooks.async_user_created_hook(
                data=NewUserRequest(user_id="new-user", send_invite_email=True),
                response=NewUserResponse(key="sk-new", user_id="new-user"),
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_id="admin"),
            )
        )
        for _ in range(3):
            await asyncio.sleep(0)
        drain: Final = asyncio.create_task(drain_audit_tasks(timeout=60))
        for _ in range(3):
            await asyncio.sleep(0)
        drain_waited_for_the_user_read: Final = not drain.done()
        release_user_read.set()
        await drain
        hook.cancel()

    assert drain_waited_for_the_user_read
    assert [(write["object_id"], write["action"]) for write in writes] == [("new-user", "created")]
