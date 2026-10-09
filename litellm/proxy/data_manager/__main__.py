"""Run spend-log cleanup in a separate process using proxy config from CONFIG_FILE_PATH.

Proxies sharing the database must set LITELLM_DATA_MANAGER_ENABLED=true.
"""

import asyncio
import os
import signal
import sys
from collections.abc import Sequence
from typing import Final

from litellm.proxy.data_manager.config import DATA_MANAGER_JOB_ROLE


def _install_stop_signals(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)


async def run_data_manager() -> None:
    from fastapi import FastAPI

    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.proxy_server import proxy_startup_event

    stop: Final = asyncio.Event()
    _install_stop_signals(asyncio.get_running_loop(), stop)
    async with proxy_startup_event(FastAPI()):
        verbose_proxy_logger.info("data manager: running")
        await stop.wait()


def main(argv: Sequence[str]) -> None:
    if argv:
        sys.exit("usage: python -m litellm.proxy.data_manager")

    os.environ["LITELLM_JOB_ROLE"] = DATA_MANAGER_JOB_ROLE
    from litellm.proxy.collector import apply_log_level
    from litellm.proxy.db.db_url_settings import DatabaseURLSettings

    apply_log_level(os.environ.get("LITELLM_LOG"))
    DatabaseURLSettings.from_env().apply_to_env()
    asyncio.run(run_data_manager())


if __name__ == "__main__":
    main(sys.argv[1:])
