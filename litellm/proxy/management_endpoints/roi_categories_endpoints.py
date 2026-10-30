import asyncio
import base64
import binascii
import os
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Final, Protocol, TypeAlias
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.roi_calculator.sync_store import SyncStore
from litellm.proxy.roi_categories.models import (
    CONFIG_KEY,
    Categories,
    ContextText,
    ContextUpload,
    Example,
    Report,
    RetryDiscovery,
    Settings,
    Setup,
    Status,
)
from litellm.proxy.roi_categories.service import discover, process_batch
from litellm.proxy.roi_categories.storage import CategoryStorage
from litellm.rag.ingestion.file_parsers.pdf_parser import extract_text_from_pdf
from litellm.repositories.config_repository import ConfigRepository
from litellm.router_strategy.complexity_router.jev_classifier import RouterJevClassifierClient
from litellm.rust_bridge.clickhouse import spend_storage_config
from litellm.types.roi_calculator import ROISyncStatus
from litellm.types.utils import ModelResponse


class ROIJobScheduler(Protocol):
    def add_job(
        self,
        func: Callable[[], Awaitable[None]],
        trigger: str,
        *,
        seconds: int,
        id: str,
        max_instances: int,
        replace_existing: bool,
    ) -> object: ...


router: Final = APIRouter(prefix="/roi/categories", tags=["ROI categories"])
_RUNNING: Final = ROISyncStatus(
    running=True,
    phase="spend",
    stage="Categorizing traffic",
    done=0,
    total=0,
    estimated=0,
    reused=0,
    needs_attention=0,
    error=None,
)


async def admin(auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(403, "ROI categories are available to proxy administrators only")
    return auth


async def repository(_: Annotated[UserAPIKeyAuth, Depends(admin)]) -> ConfigRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(503, "ROI setup requires PostgreSQL")
    return ConfigRepository(prisma_client, use_writer=True)


async def load(repository: ConfigRepository) -> Settings | None:
    record: Final = await repository.get_param(CONFIG_KEY)
    return Settings.model_validate(record.param_value) if record else None


async def storage(_: Annotated[UserAPIKeyAuth, Depends(admin)]) -> AsyncIterator[CategoryStorage]:
    if not os.environ.get("CLICKHOUSE_URL"):
        raise HTTPException(503, "Connect ClickHouse and enable spend logging to use ROI categories")
    async with httpx.AsyncClient() as client:
        try:
            yield CategoryStorage(spend_storage_config(), client)
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "ClickHouse is unavailable or returned invalid category data") from None


@asynccontextmanager
async def lease(repository: ConfigRepository) -> AsyncGenerator[None]:
    store: Final = SyncStore(repository.prisma_client, namespace="roi_categories")
    owner: Final = str(uuid4())
    if not await store.acquire(owner, _RUNNING):
        raise HTTPException(409, "ROI processing is in progress. Try again shortly.")
    try:
        yield
    finally:
        _ = await store.finish(owner, _RUNNING.model_copy(update={"running": False, "phase": "complete"}))


Repository: TypeAlias = Annotated[ConfigRepository, Depends(repository)]
Storage: TypeAlias = Annotated[CategoryStorage, Depends(storage)]


@router.get("/status", response_model=Status)
async def category_status(repository: Repository) -> Status:
    """Return ClickHouse configuration status and the current discovery or classification phase."""
    return Status(configured=bool(os.environ.get("CLICKHOUSE_URL")), settings=await load(repository))


@router.post("/setup", response_model=Settings)
async def setup_categories(body: Setup, repository: Repository, storage: Storage) -> Settings:
    """Opt into background discovery from new logged traffic using configured gateway models."""
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None or any(
        name not in llm_router.registry().names() for name in (body.discovery_model, body.classifier_model)
    ):
        raise HTTPException(400, "Select discovery and Jev models configured on this gateway")
    if litellm.turn_off_message_logging:
        raise HTTPException(409, "ROI discovery requires eligible message logging")
    async with lease(repository):
        if await load(repository) is not None:
            raise HTTPException(409, "ROI has already been configured")
        await storage.ensure_schema()
        settings: Final = Settings(
            sample_size=body.sample_size,
            discovery_model=body.discovery_model,
            classifier_model=body.classifier_model,
            context=body.context,
            team_id=body.team_id,
            version=str(uuid4()),
            started_at=datetime.now(timezone.utc),
        )
        _ = await repository.set_param(CONFIG_KEY, settings.model_dump(mode="json"))
        return settings


@router.post("/activate", response_model=Settings)
async def activate_categories(body: Categories, repository: Repository) -> Settings:
    """Approve a fixed category set and begin classifying subsequent requests with Jev."""
    async with lease(repository):
        settings: Final = await load(repository)
        if settings is None or settings.phase != "review":
            raise HTTPException(409, "Discover categories before activating them")
        active: Final = settings.model_copy(
            update={
                "categories": body.categories,
                "phase": "active",
                "activated_at": datetime.now(timezone.utc),
                "context": "",
            }
        )
        _ = await repository.set_param(CONFIG_KEY, active.model_dump(mode="json"))
        return active


@router.post("/retry", response_model=Settings)
async def retry_discovery(body: RetryDiscovery, repository: Repository) -> Settings:
    """Retry a failed discovery with the existing sample and business context."""
    async with lease(repository):
        settings: Final = await load(repository)
        if settings is None or settings.phase != "error":
            raise HTTPException(409, "Only failed discovery can be retried")
        updated: Final = settings.model_copy(update={"phase": "sampling", "error": None})
        _ = await repository.set_param(CONFIG_KEY, updated.model_dump(mode="json"))
        return updated


@router.post("/context", response_model=ContextText)
async def extract_context(body: ContextUpload, _: Annotated[UserAPIKeyAuth, Depends(admin)]) -> ContextText:
    """Extract bounded business context from a PDF, UTF-8 text, or Markdown file."""
    try:
        content: Final = base64.b64decode(body.content_base64, validate=True)
    except binascii.Error:
        raise HTTPException(422, "File content must be valid base64") from None
    name: Final = body.name
    if len(content) > 1024 * 1024:
        raise HTTPException(413, "Each context file must be at most 1 MiB")
    if name.lower().endswith(".pdf"):
        text: Final = await asyncio.to_thread(extract_text_from_pdf, content)
        if not text:
            raise HTTPException(422, "No text could be extracted from this PDF. Upload a text or Markdown file.")
        return ContextText(name=name, text=redact_string(text[:32000]))
    if not name.lower().endswith((".txt", ".md")):
        raise HTTPException(415, "Use PDF, text, or Markdown files")
    try:
        return ContextText(name=name, text=redact_string(content.decode("utf-8")[:32000]))
    except UnicodeDecodeError:
        raise HTTPException(422, "Text files must use UTF-8 encoding") from None


async def active_settings(repository: ConfigRepository, start: datetime, end: datetime) -> Settings:
    if end <= start or (end - start).days > 90:
        raise HTTPException(400, "Choose a date range of up to 90 days")
    settings: Final = await load(repository)
    if settings is None:
        raise HTTPException(409, "Set up ROI categories first")
    return settings


@router.get("/report", response_model=Report)
async def category_report(
    repository: Repository,
    storage: Storage,
    start: AwareDatetime,
    end: AwareDatetime,
    model: Annotated[str, Query(max_length=200)] = "",
    user: Annotated[str, Query(max_length=200)] = "",
) -> Report:
    """Return daily category spend and model breakdowns for the selected UTC interval."""
    settings: Final = await active_settings(repository, start, end)
    buckets: Final = await storage.report(settings, start, end, model, user)
    if len(buckets) > 10000:
        raise HTTPException(400, "Narrow the date range or model filter")
    sessions: Final = await storage.sessions(settings, start, end, model, user)
    if len(sessions) > 1000:
        raise HTTPException(400, "More than 1,000 sessions. Narrow the date range or model filter.")
    return Report(categories=settings.categories, buckets=buckets, sessions=sessions, version=settings.version)


@router.get("/examples", response_model=tuple[Example, ...])
async def category_examples(
    repository: Repository,
    storage: Storage,
    start: AwareDatetime,
    end: AwareDatetime,
    category_id: Annotated[str, Query(max_length=48)] = "",
    model: Annotated[str, Query(max_length=200)] = "",
    user: Annotated[str, Query(max_length=200)] = "",
    session_key: Annotated[str, Query(max_length=2000)] = "",
) -> tuple[Example, ...]:
    """Return up to five highest-spend request excerpts for a category in the selected interval."""
    settings: Final = await active_settings(repository, start, end)
    return await storage.examples(settings, start, end, category_id, model, user, session_key)


async def complete(model: str, instructions: str, evidence: str) -> str:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise RuntimeError("Gateway router is unavailable")
    response: Final = ModelResponse.model_validate(
        await llm_router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # Router SDK kwargs are untyped
            model=model,
            messages=[{"role": "system", "content": instructions}, {"role": "user", "content": evidence}],
            response_format={"type": "json_object"},
            max_tokens=3000,
            timeout=25,
            num_retries=0,
            metadata={"tags": ["litellm-roi-analysis"]},
            turn_off_message_logging=True,
        )
    )
    return str(response.choices[0].message.content or "")


async def run_roi_tick() -> None:
    from litellm.proxy.proxy_server import llm_router, prisma_client

    if prisma_client is None or llm_router is None or not os.environ.get("CLICKHOUSE_URL"):
        return
    repository: Final = ConfigRepository(prisma_client, use_writer=True)
    try:
        if await load(repository) is None:
            return
        async with asyncio.timeout(45), lease(repository):
            settings: Final = await load(repository)
            if settings is None or settings.phase in ("review", "error") or litellm.turn_off_message_logging:
                return
            async with httpx.AsyncClient() as client:
                storage: Final = CategoryStorage(spend_storage_config(), client)
                now: Final = datetime.now(timezone.utc)
                if settings.phase == "sampling":
                    try:
                        updated: Final = await discover(settings, await storage.sample(settings, now), complete)
                    except Exception:  # noqa: BLE001  # Model and storage failures must remain outside the request path
                        failed: Final = settings.model_copy(
                            update={
                                "phase": "error",
                                "error": "Discovery failed. Check model access and ClickHouse connectivity.",
                            }
                        )
                        _ = await repository.set_param(CONFIG_KEY, failed.model_dump(mode="json"))
                    else:
                        _ = await repository.set_param(CONFIG_KEY, updated.model_dump(mode="json"))
                    return
                await process_batch(
                    settings, storage, RouterJevClassifierClient(llm_router, settings.classifier_model), now
                )
    except HTTPException:
        return
    except Exception:  # noqa: BLE001  # Model and storage failures must remain outside the request path
        verbose_proxy_logger.warning("ROI background processing could not finish; it will retry on the next interval")
