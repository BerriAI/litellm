"""Refreshable credentials whose refresh only completes while the event loop keeps serving."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Final

from botocore.credentials import RefreshableCredentials

REFRESH_RELEASE_TIMEOUT_SECONDS: Final = 2.0
REFRESH_START_TIMEOUT_SECONDS: Final = 10.0


class EventLoopProbe:
    """Blocks inside botocore's credential refresh until a coroutine on the loop releases it.

    Signing on the event loop thread can never be released, so `served_during_refresh` reads False there
    and True only when the refresh ran on another thread while the loop stayed responsive.
    """

    def __init__(self) -> None:
        self.refresh_started: Final = threading.Event()
        self.loop_served: Final = threading.Event()
        self.served_during_refresh: bool | None = None

    def refresh(self) -> dict[str, str | None]:
        self.refresh_started.set()
        served: Final = self.loop_served.wait(timeout=REFRESH_RELEASE_TIMEOUT_SECONDS)
        if self.served_during_refresh is None:
            self.served_during_refresh = served
        return {
            "access_key": "AKIAREFRESHED",
            "secret_key": "refreshed-secret",
            "token": None,
            "expiry_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }

    def credentials(self) -> RefreshableCredentials:
        return RefreshableCredentials(
            access_key="AKIASTALE",
            secret_key="stale-secret",
            token=None,
            expiry_time=datetime.now(timezone.utc) + timedelta(seconds=60),
            refresh_using=self.refresh,
            method="event-loop-probe",
        )

    async def release_refresh_from_the_loop(self) -> None:
        deadline: Final = time.monotonic() + REFRESH_START_TIMEOUT_SECONDS
        while not self.refresh_started.is_set():
            if time.monotonic() > deadline:
                raise TimeoutError("signing finished without ever starting a credential refresh")
            await asyncio.sleep(0.005)
        self.loop_served.set()
