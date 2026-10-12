from __future__ import annotations

import re
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
from integration._support.redis_process import PORT_ATTEMPTS, free_loopback_port, owned_redis
from redis import Redis


def _redis_pid(host: str, port: int) -> int:
    with Redis(host=host, port=port) as client:
        return int(client.info("server")["process_id"])


def test_owned_redis_moves_off_a_port_another_owned_redis_already_serves(tmp_path: Path) -> None:
    with owned_redis(tmp_path / "first") as first:
        candidates: Final = chain((first.port,), iter(free_loopback_port, None))
        with owned_redis(tmp_path / "second", lambda: next(candidates)) as second:
            assert second.port != first.port
            assert first.serves()
            assert second.serves()
            assert second.server_pid != first.server_pid
            assert _redis_pid(second.host, second.port) == second.server_pid
            assert _redis_pid(first.host, first.port) == first.server_pid
        assert first.serves()


def test_owned_redis_gives_up_with_the_last_cause_after_a_bounded_number_of_ports(tmp_path: Path) -> None:
    with owned_redis(tmp_path / "first") as first:
        ports: Final = iter((first.port,) * PORT_ATTEMPTS)
        with pytest.raises(
            AssertionError,
            match=(
                f"after {PORT_ATTEMPTS} attempts: Another Redis already serves {re.escape(first.host)}:{first.port}"
            ),
        ):
            with owned_redis(tmp_path / "second", lambda: next(ports)):
                pytest.fail("Owned Redis unexpectedly reused a port already serving Redis")
        assert next(ports, None) is None
        assert first.serves()
