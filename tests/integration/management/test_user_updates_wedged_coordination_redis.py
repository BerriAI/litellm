import os
import signal
import time
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from redis import Redis

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import owned_redis

_USERS: Final = 60
_HANDLER_BUDGET_SECONDS: Final = 2.5


def _timed_post(candidate: Gateway, path: str, body: dict[str, object]) -> float:
    started: Final = time.monotonic()
    response: Final = candidate.request("POST", path, body)
    elapsed: Final = time.monotonic() - started
    assert response.status_code == 200, f"POST {path}: {response.status_code} {response.text}"
    return elapsed


@pytest.mark.timeout(180)
@pytest.mark.covers(
    "mgmt.user.update.budget_change_returns_promptly_with_wedged_coordination_redis",
    "mgmt.user.bulk_update.budget_change_returns_promptly_with_wedged_coordination_redis",
)
def test_user_budget_updates_return_promptly_while_coordination_redis_is_wedged(
    gateway: Gateway, tmp_path: Path
) -> None:
    original: Final = os.environ["DATABASE_URL"]
    identity: Final = "integration_wedged_redis_" + uuid.uuid4().hex
    parsed: Final = urlsplit(original)
    database_url: Final = urlunsplit((parsed.scheme, parsed.netloc, "/" + identity, "", ""))
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(identity)))
        try:
            with (
                owned_redis(tmp_path) as coordination,
                owned_proxy(
                    gateway,
                    tmp_path,
                    {
                        "DATABASE_URL": database_url,
                        "REDIS_HOST": coordination.host,
                        "REDIS_PORT": str(coordination.port),
                    },
                    config=Path("tests/integration/coordination_redis_proxy_config.yaml"),
                ) as candidate,
            ):
                users: Final = tuple(f"{identity}_u{index}" for index in range(_USERS))
                for user_id in users:
                    candidate.post("/user/new", {"user_id": user_id, "auto_create_key": False, "max_budget": 10.0})
                with Redis(host=coordination.host, port=coordination.port, socket_timeout=1) as client:
                    eventually(
                        lambda: client.pubsub_numsub("litellm_proxy.auth_cache_invalidation")[0][1],
                        lambda count: count >= 1,
                        seconds=8,
                    )
                assert (
                    _timed_post(candidate, "/user/update", {"user_id": users[0], "max_budget": 11.0})
                    < _HANDLER_BUDGET_SECONDS
                )
                coordination.signal(signal.SIGSTOP)
                try:
                    control: Final = _timed_post(candidate, "/user/update", {"user_id": users[0], "tpm_limit": 1000})
                    assert control < _HANDLER_BUDGET_SECONDS, (
                        f"control update without a cache-relevant field took {control:.2f}s"
                    )
                    single: Final = _timed_post(candidate, "/user/update", {"user_id": users[0], "max_budget": 98.0})
                    assert single < _HANDLER_BUDGET_SECONDS, (
                        f"/user/update with max_budget took {single:.2f}s with a wedged coordination Redis"
                    )
                    bulk: Final = _timed_post(
                        candidate, "/user/bulk_update", {"all_users": True, "user_updates": {"max_budget": 79.0}}
                    )
                    assert bulk < 2 * _HANDLER_BUDGET_SECONDS, (
                        f"/user/bulk_update over {_USERS} users took {bulk:.2f}s with a wedged coordination Redis"
                    )
                finally:
                    coordination.signal(signal.SIGCONT)
                info: Final = object_value(candidate.get("/user/info", {"user_id": users[-1]})["user_info"])
                assert info["max_budget"] == 79.0, info
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(identity)))
