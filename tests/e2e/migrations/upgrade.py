from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final
from uuid import uuid4

from e2e_http import Result, Success, unwrap
from models import (
    KeyGenerateBody,
    KeyGenerateResponse,
    KeyInfoParams,
    KeyInfoResponse,
    ModelsListParams,
    ModelsListResponse,
)
from pydantic import BaseModel

from .containers import Replica, until
from .database import Database

CACHED_PLAN: Final = "cached plan must not change result type"


def provision(replica: Replica) -> tuple[str, str]:
    alias: Final = f"upgrade-{uuid4().hex}"
    key: Final = unwrap(
        replica.transport.post(
            "/key/generate",
            headers=replica.transport.master,
            json=KeyGenerateBody(key_alias=alias),
            response_type=KeyGenerateResponse,
        )
    ).key
    return key, alias


def confirm(replica: Replica, key: str, alias: str) -> None:
    info: Final = unwrap(
        replica.transport.get(
            "/key/info",
            headers=replica.transport.master,
            params=KeyInfoParams(key=key),
            response_type=KeyInfoResponse,
        )
    )
    assert info.info.key_alias == alias, "Key minted on one release did not resolve on the other"


@dataclass(slots=True)
class Outcomes:
    served: int = 0
    failures: list[str] = field(default_factory=list)

    def record(self, result: Result[BaseModel]) -> None:
        match result:
            case Success():
                self.served += 1
            case _:
                self.failures.append(result.model_dump_json())


@contextmanager
def auth_traffic(replica: Replica, key: str, interval: float = 0.05) -> Generator[Outcomes]:
    outcomes: Final = Outcomes()
    stop: Final = threading.Event()

    def drive() -> None:
        while not stop.is_set():
            outcomes.record(
                replica.transport.get(
                    "/v1/models",
                    headers=replica.transport.bearer(key),
                    params=ModelsListParams(),
                    response_type=ModelsListResponse,
                    timeout=10,
                )
            )
            stop.wait(interval)

    thread: Final = threading.Thread(target=drive, name="upgrade-auth-traffic", daemon=True)
    thread.start()
    try:
        yield outcomes
    finally:
        stop.set()
        thread.join(30)
        assert not thread.is_alive(), "Auth traffic thread did not stop"
    assert not outcomes.failures, (
        f"Virtual-key auth failed on {replica.name} after the traffic window closed: {outcomes.failures[:5]}"
    )


def keep_serving(outcomes: Outcomes, description: str, calls: int = 20) -> int:
    target: Final = outcomes.served + calls
    until(description, lambda: outcomes.served >= target or bool(outcomes.failures))
    assert not outcomes.failures, f"Virtual-key auth failed during {description}: {outcomes.failures[:5]}"
    return outcomes.served


def migration_names(database: Database) -> frozenset[str]:
    return frozenset(str(row[0]) for row in database.query("SELECT migration_name FROM _prisma_migrations"))


def assert_history_clean(database: Database) -> None:
    assert database.query(
        "SELECT count(*) FROM _prisma_migrations WHERE finished_at IS NULL OR rolled_back_at IS NOT NULL"
    ) == ((0,),), "The upgrade left an unfinished or rolled-back migration behind"
    assert database.query(
        "SELECT count(*) FROM (SELECT migration_name FROM _prisma_migrations GROUP BY migration_name "
        "HAVING count(*) > 1) duplicated"
    ) == ((0,),), "A migration was recorded more than once, so it ran on more than one replica"


def assert_upgraded(before: frozenset[str], after: frozenset[str]) -> frozenset[str]:
    applied: Final = after - before
    assert applied, (
        "The candidate applied no migrations the baseline release had not: the pinned "
        "LITELLM_MIGRATION_BASELINE_IMAGE is at or ahead of the candidate, so this suite proves nothing"
    )
    assert not before - after, "The upgrade removed migration history the baseline release had already applied"
    return applied
