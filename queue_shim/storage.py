from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Final

from redis.asyncio import Redis

__all__ = [
    "Job",
    "Storage",
    "claim_job",
    "enqueue_job",
    "get_job",
    "idem_key",
    "list_pending",
    "mark_done",
    "mark_failed",
    "requeue_job",
]

QUEUE_KEY: Final = "queue:pending"
JOB_PREFIX: Final = "job:"
CLAIM_TTL: Final = float(os.environ.get("CLAIM_TTL", "300"))


def idem_key(model: str, body: dict[str, object]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{model}\x00{payload}".encode("utf-8")).hexdigest()


def job_key(key: str) -> str:
    suffix = key[len(JOB_PREFIX):] if key.startswith(JOB_PREFIX) else key
    return f"{JOB_PREFIX}{suffix}"


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    idem_key: str
    model: str
    body: dict[str, object]
    stream: bool
    status: str
    result: str | None = None

    @staticmethod
    def from_hash(job_id: str, h: dict[str, str]) -> Job:
        return Job(
            job_id=job_id,
            idem_key=h.get("idem_key", ""),
            model=h.get("model", ""),
            body=json.loads(h.get("body", "{}")),
            stream=h.get("stream") == "true",
            status=h.get("status", "pending"),
            result=h.get("result"),
        )


async def get_job(client: Redis, key: str) -> Job | None:
    h = await client.hgetall(job_key(key))
    if not h:
        return None
    return Job.from_hash(key, h)


async def enqueue_job(client: Redis, model: str, body: dict[str, object], stream: bool) -> tuple[str, bool]:
    key = idem_key(model, body)
    jk = job_key(key)
    if await client.exists(jk):
        return key, False
    now = time.time()
    pipe = client.pipeline()
    pipe.hset(
        jk,
        mapping={
            "status": "pending",
            "model": model,
            "body": json.dumps(body, sort_keys=True, separators=(",", ":")),
            "stream": "true" if stream else "false",
            "idem_key": key,
            "created_at": str(now),
        },
    )
    pipe.zadd(QUEUE_KEY, {jk: now})
    await pipe.execute()
    return key, True


async def claim_job(client: Redis, key: str, worker: str, ttl: float = CLAIM_TTL) -> Job | None:
    jk = job_key(key)
    now = time.time()

    status = await client.hget(jk, "status")
    if status in ("done", "failed"):
        return None
    claim_expires = await client.hget(jk, "claim_expires") or "0"
    if status == "running" and float(claim_expires) > now:
        return None

    async with client.pipeline(transaction=False) as pipe:
        pipe.hget(jk, "status")
        pipe.hget(jk, "claim_expires")
        results = await pipe.execute()
        s = results[0]
        ce = results[1] or "0"
        if s in ("done", "failed") or (s == "running" and float(ce) > now):
            return None
        pipe.hset(jk, mapping={"status": "running", "claimed_by": worker, "claim_expires": str(now + ttl)})
        await pipe.execute()
    return await get_job(client, key)


async def requeue_job(client: Redis, key: str) -> None:
    jk = job_key(key)
    created = await client.hget(jk, "created_at")
    pipe = client.pipeline()
    pipe.hset(jk, mapping={"status": "pending", "claimed_by": "", "claim_expires": ""})
    pipe.zadd(QUEUE_KEY, {jk: float(created) if created else time.time()})
    await pipe.execute()


async def list_pending(client: Redis, limit: int = 10) -> list[str]:
    return await client.zrangebyscore(QUEUE_KEY, "-inf", "+inf", start=0, num=limit)


async def mark_done(client: Redis, key: str, result: str) -> None:
    jk = job_key(key)
    pipe = client.pipeline()
    pipe.hset(jk, mapping={"status": "done", "result": result, "claimed_by": "", "claim_expires": ""})
    pipe.zrem(QUEUE_KEY, jk)
    await pipe.execute()


async def mark_failed(client: Redis, key: str, error: str) -> None:
    jk = job_key(key)
    pipe = client.pipeline()
    pipe.hset(jk, mapping={"status": "failed", "result": error, "claimed_by": "", "claim_expires": ""})
    pipe.zrem(QUEUE_KEY, jk)
    await pipe.execute()


class Storage:
    """Async facade over the Redis job primitives."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, key: str) -> Job | None:
        return await get_job(self._client, key)

    async def enqueue(self, model: str, body: dict[str, object], stream: bool) -> tuple[str, bool]:
        return await enqueue_job(self._client, model, body, stream)

    async def claim(self, key: str, worker: str, ttl: float = CLAIM_TTL) -> Job | None:
        return await claim_job(self._client, key, worker, ttl)

    async def requeue(self, key: str) -> None:
        await requeue_job(self._client, key)

    async def list_pending(self, limit: int = 10) -> list[str]:
        return await list_pending(self._client, limit)

    async def mark_done(self, key: str, result: str) -> None:
        await mark_done(self._client, key, result)

    async def mark_failed(self, key: str, error: str) -> None:
        await mark_failed(self._client, key, error)
