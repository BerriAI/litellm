from __future__ import annotations

import re
from pathlib import Path

import pytest
from integration._support.redis_process import owned_redis
from redis import Redis


def _redis_pid(host: str, port: int) -> int:
    with Redis(host=host, port=port) as client:
        return int(client.info("server")["process_id"])


def test_owned_redis_restart_refuses_a_port_another_redis_took_while_it_was_down(tmp_path: Path) -> None:
    with owned_redis(tmp_path / "first") as first:
        first.stop()
        with owned_redis(tmp_path / "squatter", lambda: first.port) as squatter:
            with pytest.raises(
                AssertionError, match=f"Another Redis already serves {re.escape(first.host)}:{first.port}"
            ):
                first.start()
            assert first.process is None
            assert _redis_pid(squatter.host, squatter.port) == squatter.server_pid
