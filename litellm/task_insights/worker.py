"""The Task Insights analysis worker: `python -m litellm.task_insights.worker`.

It runs queued discovery and buckets new requests against the frozen taxonomy. It is the only part of Task Insights
that calls models, and it starts no model call by itself: discovery runs only after an admin queues it, and bucketing
only once a taxonomy is frozen and new requests with stored prompt text arrive. Every call goes through the gateway
with the analysis tag, so it is billed like other traffic and never analyzed itself.
"""

import argparse
import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Final

import httpx
from openai import AsyncOpenAI
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.repositories.task_insights_repository import TaskInsightsRepository
from litellm.task_insights.bucketing import TaskDecider, bucket_pending
from litellm.task_insights.database import connected_database
from litellm.task_insights.discovery import discover
from litellm.task_insights.gateway import ANALYSIS_HEADERS
from litellm.task_insights.jev import GatewayJevDecider


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LITELLM_TASK_INSIGHTS_", frozen=True, extra="ignore")

    gateway_url: str = "http://litellm:4000/v1"
    gateway_key: SecretStr | None = None
    embedding_model: str = "task-insights-embedding"
    synthesis_model: str = "task-insights-synthesis"
    decision_model: str = "task-insights-decision"
    min_probability: float = Field(default=0.5, ge=0, le=1)
    min_fit: float = Field(default=0.5, ge=0, le=1)
    bucket_batch_size: int = Field(default=16, ge=1, le=200)
    poll_seconds: float = Field(default=15, gt=0)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def run_once(
    repository: TaskInsightsRepository,
    settings: WorkerSettings,
    decider: TaskDecider | None,
    gateway: AsyncOpenAI | None,
    now: Callable[[], datetime],
) -> int:
    if gateway is None:
        if (await repository.get_state()).discovery_status == "requested":
            await repository.claim_discovery()
            await repository.fail_discovery(
                "Set LITELLM_TASK_INSIGHTS_GATEWAY_KEY so the worker can call the embedding and synthesis models",
                now(),
            )
    else:
        from litellm.task_insights.clustering import UmapHdbscanClusterer
        from litellm.task_insights.gateway import GatewayEmbedder, GatewaySynthesizer

        await discover(
            repository,
            GatewayEmbedder(gateway, settings.embedding_model),
            UmapHdbscanClusterer(),
            GatewaySynthesizer(gateway, settings.synthesis_model),
            now=now,
        )
    if decider is None:
        return 0
    return await bucket_pending(repository, decider, now=now, batch_size=settings.bucket_batch_size)


async def _serve(once: bool) -> None:
    settings: Final = WorkerSettings()
    async with connected_database() as database, httpx.AsyncClient(timeout=120) as http:
        repository: Final = TaskInsightsRepository(database)
        await repository.recover_discovery()
        key: Final = settings.gateway_key.get_secret_value() if settings.gateway_key is not None else None
        if key is None:
            verbose_proxy_logger.warning(
                "Task Insights worker: LITELLM_TASK_INSIGHTS_GATEWAY_KEY is unset, no model calls"
            )
        gateway: Final = (
            AsyncOpenAI(base_url=settings.gateway_url, api_key=key, http_client=http, default_headers=ANALYSIS_HEADERS)
            if key is not None
            else None
        )
        decider: Final = (
            GatewayJevDecider(
                http,
                url=f"{settings.gateway_url.rstrip('/')}/systemone",
                api_key=key,
                model=settings.decision_model,
                min_probability=settings.min_probability,
                min_fit=settings.min_fit,
            )
            if key is not None
            else None
        )
        while True:
            bucketed = await run_once(repository, settings, decider, gateway, _now)
            if once:
                return
            if bucketed < settings.bucket_batch_size:
                await asyncio.sleep(settings.poll_seconds)


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Task Insights analysis worker")
    parser.add_argument("--once", action="store_true", help="Run one discovery and bucketing pass, then exit")
    asyncio.run(_serve(parser.parse_args().once))


if __name__ == "__main__":
    main()
