"""
Daily rollup for per-model PTU (provisioned throughput) flat cost.

v1 reads PTU config straight off the model deployment
(``LiteLLM_ProxyModelTable.model_info``): a deployment carrying ``ptu_count``
and ``cost_per_ptu_per_hour`` accrues flat cost of
``ptu_count * cost_per_ptu_per_hour * active_hours`` for a given UTC day, where
``active_hours`` is the overlap between the day and the optional
``[ptu_effective_from, ptu_effective_to)`` window (a window opening at 23:00
charges one hour that day). The amount is written to ``LiteLLM_DailyTeamSpend``
under a sentinel api_key so the rows are distinguishable from per-request rows
and share the existing unique constraint.
"""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    PTU_LAPSED_ALERT_LIMIT,
    PTU_PRUNE_SKEW_GRACE_SECONDS,
    PTU_ROLLUP_JOB_ID,
    PTU_ROLLUP_LOCK_TTL_SECONDS,
    PTU_ROLLUP_MAX_BACKFILL_DAYS,
    PTU_SENTINEL_API_KEY,
)
from litellm.litellm_core_utils.ptu_pricing import ptu_terms
from litellm.proxy.spend_tracking.ptu_feature_flag import is_ptu_cost_attribution_enabled

if TYPE_CHECKING:
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient

_HOURS_PER_DAY: Final = 24
_PRUNE_ID_CHUNK_SIZE: Final = 5_000
_UPSERT_ATTEMPTS: Final = 3
_UPSERT_RETRY_BACKOFF_SECONDS: Final = 0.5


@dataclass(frozen=True, slots=True)
class RollupResult:
    day: date
    models_processed: int
    rows_written: int
    rows_failed: int = 0
    lapsed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackfillResult:
    start: date
    end: date
    days_scanned: int
    rows_written: int
    rows_failed: int = 0


@dataclass(frozen=True, slots=True)
class PTUModel:
    """A model deployment carrying valid manual PTU config."""

    model_id: str
    model_name: str
    team_id: str
    ptu_count: int
    cost_per_ptu_per_hour: float
    effective_from: datetime | None = None
    effective_to: datetime | None = None


def _public_model_name(row: object, model_info: Mapping[str, object]) -> str:
    """The name an operator recognises for this deployment.

    Creating a team-scoped deployment rewrites model_name to a synthetic routing key
    (``model_name_<team_id>_<uuid4>``) and keeps the chosen name in
    ``model_info.team_public_model_name``. PTU config is only accepted alongside a
    team_id, so every PTU deployment carries that synthetic name; keying the sentinel
    row on it would file each charge under a UUID that no usage view can resolve and
    that never lines up with the same model's request rows.
    """
    public_name: Final = model_info.get("team_public_model_name")
    if isinstance(public_name, str) and public_name:
        return public_name
    return str(getattr(row, "model_name", "") or "")


def _decode_model_info(raw: object) -> "Mapping[str, object] | None":
    """A deployment's model_info as a mapping, decoding a JSON string, else None.

    Valid JSON that is not an object decodes to a list or a scalar, which every caller
    would then read fields off, so it is rejected here rather than raised past them.
    """
    if isinstance(raw, str):
        try:
            decoded: Final = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return decoded if isinstance(decoded, dict) else None
    if isinstance(raw, Mapping):
        return raw
    return None


@dataclass(frozen=True, slots=True)
class _PTUDeployment:
    """A deployment in the shape ``_parse_ptu_model`` reads, whatever declared it.

    A ``LiteLLM_ProxyModelTable`` row already has it. A router entry does not: its id
    lives in ``model_info.id`` rather than on the entry itself.
    """

    model_id: str
    model_name: str
    model_info: Mapping[str, object]


def _router_deployment(deployment: Mapping[str, object]) -> _PTUDeployment | None:
    """A router ``model_list`` entry in the shape the parser reads, else None.

    An id is required rather than defaulted because it keys the sentinel row: every
    deployment without one would collapse onto a single row per team and only the last
    would be billed. The mapping is copied because the router rewrites entries in place
    while the rollup runs.
    """
    model_info: Final = _decode_model_info(deployment.get("model_info"))
    if model_info is None:
        return None
    model_id: Final = model_info.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    return _PTUDeployment(
        model_id=model_id,
        model_name=str(deployment.get("model_name") or ""),
        model_info=MappingProxyType(dict(model_info)),
    )


def _parse_ptu_model(row: object) -> PTUModel | None:
    """Return a PTUModel when the deployment carries valid manual PTU config, else None.

    Valid means model_info has a positive ptu_count, a non-negative
    cost_per_ptu_per_hour, and a team_id (1 model -> 1 team).
    """
    model_info: Final = _decode_model_info(getattr(row, "model_info", None))
    if model_info is None:
        return None
    terms: Final = ptu_terms(model_info)
    if terms is None:
        return None
    return PTUModel(
        model_id=str(getattr(row, "model_id", "") or ""),
        model_name=_public_model_name(row, model_info),
        team_id=terms.team_id,
        ptu_count=terms.ptu_count,
        cost_per_ptu_per_hour=terms.cost_per_ptu_per_hour,
        effective_from=terms.effective_from,
        effective_to=terms.effective_to,
    )


def _active_hours_on_day(model: PTUModel, day: date) -> float:
    """Hours the model's PTU window overlaps ``day`` (UTC), clamped to [0, 24]."""
    day_start: Final = datetime.combine(day, time.min, tzinfo=timezone.utc)
    day_end: Final = day_start + timedelta(days=1)
    start: Final = max(day_start, model.effective_from) if model.effective_from else day_start
    end: Final = min(day_end, model.effective_to) if model.effective_to else day_end
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 3600.0


def _compute_daily_flat_cost(model: PTUModel, day: date) -> float:
    """Flat cost for ``day``: ptu_count * cost_per_ptu_per_hour * active_hours."""
    return float(model.ptu_count) * model.cost_per_ptu_per_hour * _active_hours_on_day(model, day)


@dataclass(frozen=True, slots=True)
class _PTUCharge:
    """One sentinel row's worth of flat cost for a deployment on a day.

    ``model_id`` is the row's identity and goes in the unique key; ``model_name`` is what
    an operator reads and rides alongside it. A deployment can be renamed, so keying on
    the name would let two runs holding different config views write the same day twice.
    """

    team_id: str
    model_id: str
    model_name: str
    flat_cost: float


def _aggregate_charges(ptu_models: tuple[PTUModel, ...], day: date) -> tuple[_PTUCharge, ...]:
    """One charge per deployment that accrues cost on ``day``. Zero-cost deployments are
    dropped, which keeps a day outside a window from writing a row.

    Deployments sharing a public name inside a team no longer need collapsing: each keys
    its own row on its own id, and the read path merges them back under the shared name.
    """
    return tuple(
        _PTUCharge(
            team_id=model.team_id,
            model_id=model.model_id,
            model_name=model.model_name,
            flat_cost=_compute_daily_flat_cost(model, day),
        )
        for model in sorted(ptu_models, key=lambda m: (m.team_id, m.model_id))
        if _compute_daily_flat_cost(model, day) > 0
    )


async def _upsert_ptu_daily_row(
    prisma_client: "PrismaClient",
    *,
    team_id: str,
    model_id: str,
    model_name: str,
    date_str: str,
    flat_cost: float,
) -> None:
    """Idempotent upsert of a sentinel-api_key row on LiteLLM_DailyTeamSpend.

    ``model`` holds the deployment id because it is part of the table's unique key and a
    rename must not move the row. ``model_group`` carries the operator-facing name, which
    is outside the key and is what the usage views display.
    """
    where: Final = {  # mutable-ok: prisma upsert filter payload
        "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint": {  # mutable-ok: prisma composite-key filter
            "team_id": team_id,
            "date": date_str,
            "api_key": PTU_SENTINEL_API_KEY,
            "model": model_id,
            "custom_llm_provider": "",
            "mcp_namespaced_tool_name": "",
            "endpoint": "",
        }
    }
    now: Final = datetime.now(timezone.utc)
    await prisma_client.db.litellm_dailyteamspend.upsert(
        where=where,
        data={  # mutable-ok: prisma upsert data payload
            "create": {  # mutable-ok: prisma create payload
                "team_id": team_id,
                "date": date_str,
                "api_key": PTU_SENTINEL_API_KEY,
                "model": model_id,
                "model_group": model_name,
                "custom_llm_provider": "",
                "mcp_namespaced_tool_name": "",
                "endpoint": "",
                "ptu_flat_cost": flat_cost,
            },
            "update": {  # mutable-ok: prisma update payload
                "model_group": model_name,
                "ptu_flat_cost": flat_cost,
                "updated_at": now,
            },
        },
    )


async def _upsert_charge_with_retry(
    prisma_client: "PrismaClient",
    *,
    charge: _PTUCharge,
    date_str: str,
) -> bool:
    """Write one charge, retrying transient failures. Returns False once attempts are spent.

    The upsert is idempotent on the sentinel unique key, so a retry can only rewrite the
    same amount for the same day. Retrying in-run matters because the scheduled job moves
    on to the next date: a write lost here is a day of PTU cost that no later run replays.
    """
    for attempt in range(1, _UPSERT_ATTEMPTS + 1):
        try:
            await _upsert_ptu_daily_row(
                prisma_client,
                team_id=charge.team_id,
                model_id=charge.model_id,
                model_name=charge.model_name,
                date_str=date_str,
                flat_cost=charge.flat_cost,
            )
            return True
        except Exception as exc:  # noqa: BLE001  # one bad row must not stop the batch
            if attempt < _UPSERT_ATTEMPTS:
                verbose_proxy_logger.warning(
                    "PTU rollup: upsert attempt %d/%d failed for team=%s model=%s day=%s: %s",
                    attempt,
                    _UPSERT_ATTEMPTS,
                    charge.team_id,
                    charge.model_name,
                    date_str,
                    exc,
                )
                await asyncio.sleep(_UPSERT_RETRY_BACKOFF_SECONDS * attempt)
                continue
            verbose_proxy_logger.error(
                "PTU rollup: upsert failed after %d attempts for team=%s model=%s day=%s "
                "(rerun the rollup for that date to recover): %s",
                _UPSERT_ATTEMPTS,
                charge.team_id,
                charge.model_name,
                date_str,
                exc,
            )
    return False


@dataclass(frozen=True, slots=True)
class _LoadedDeployments:
    """The deployments a run will price, and every deployment id it looked at.

    The id set is deliberately wider than the priced set. A deployment whose PTU config
    was removed produces no charge and still has to be prunable, so bounding the prune on
    what priced would strand its old rows forever. It is also a guaranteed superset of the
    priced set, or a run could write a charge that falls outside its own delete filter.
    """

    models: tuple[PTUModel, ...]
    scanned_ids: frozenset[str]


def _running_router() -> object | None:
    """The proxy's router, or None outside a running proxy.

    Read out of ``sys.modules`` rather than imported, so a rollup driven from a test or a
    script does not pull the whole proxy server in behind it.
    """
    proxy_server: Final = sys.modules.get("litellm.proxy.proxy_server")
    return getattr(proxy_server, "llm_router", None) if proxy_server is not None else None


def _config_deployments(router: object | None, *, owned_by_db: frozenset[str]) -> tuple[_PTUDeployment, ...]:
    """Deployments the router holds that no ``LiteLLM_ProxyModelTable`` row owns.

    ``db_model`` is forced True on every deployment loaded from that table and defaults to
    False on ModelInfo, so the complement is what config.yaml declared. A per-request
    credential clone carries ``original_model_id`` and reuses its source's PTU config under
    a fresh id, so pricing it would bill one reservation once per distinct client key.
    """
    entries: Final = tuple(getattr(router, "model_list", None) or ())
    records: Final = tuple(_router_deployment(entry) for entry in entries)
    return tuple(
        record
        for record in records
        if record is not None
        and record.model_info.get("db_model") is not True
        and record.model_info.get("original_model_id") is None
        and record.model_id not in owned_by_db
    )


async def _load_ptu_models(prisma_client: "PrismaClient") -> _LoadedDeployments:
    """Every deployment carrying valid manual PTU config, and every id the scan saw.

    Reserved capacity is billed by the provider whichever file declared it, so a
    deployment the proxy only knows from config.yaml accrues alongside the stored ones.
    """
    rows: Final = await prisma_client.db.litellm_proxymodeltable.find_many()
    db_ids: Final = frozenset(model_id for row in rows if (model_id := str(getattr(row, "model_id", "") or "")))
    config_records: Final = _config_deployments(_running_router(), owned_by_db=db_ids)
    models: Final = tuple(
        parsed for parsed in (_parse_ptu_model(row) for row in (*rows, *config_records)) if parsed is not None
    )
    return _LoadedDeployments(
        models=models,
        scanned_ids=db_ids
        | frozenset(record.model_id for record in config_records)
        | frozenset(model.model_id for model in models),
    )


async def run_ptu_flat_cost_rollup(
    prisma_client: "PrismaClient",
    target_date: date | None = None,
    may_prune: bool = True,
) -> RollupResult:
    """Rollup one UTC day of flat PTU cost across all PTU-configured model deployments.

    Defaults to yesterday UTC. It upserts the current charges first, then deletes the
    day's sentinel rows it scanned and did not refresh, so an invalidated or
    now-out-of-window deployment leaves no stale charge. A deployment it cannot see is
    left alone, since its charge records capacity that was reserved and this run has no
    grounds to retract it.

    The prune predicate is ``updated_at < run_started`` rather than "not in the charge
    set I computed", which matters under concurrency: whether a row is garbage becomes a
    property of the row instead of one run's in-memory config snapshot, so a run can
    never delete a row a concurrent run just wrote. It is bounded to the deployments this
    run looked at, so a row it cannot account for is out of reach either way. It is still
    skipped when any charge failed to write, since a row whose replacement never landed
    would look unrefreshed.
    """
    day: Final = target_date or (datetime.now(timezone.utc).date() - timedelta(days=1))

    if prisma_client is None:
        verbose_proxy_logger.warning("PTU rollup: prisma_client is None, skipping")
        return RollupResult(day=day, models_processed=0, rows_written=0)

    date_str: Final = day.isoformat()
    run_started: Final = datetime.now(timezone.utc)

    loaded: Final = await _load_ptu_models(prisma_client)
    ptu_models: Final = loaded.models
    charges: Final = _aggregate_charges(ptu_models, day)

    landed: Final = tuple(
        [await _upsert_charge_with_retry(prisma_client, charge=charge, date_str=date_str) for charge in charges]
    )
    rows_written: Final = sum(landed)
    rows_failed: Final = len(charges) - rows_written

    if not may_prune:
        verbose_proxy_logger.info(
            "PTU rollup for %s: ran without the cross-pod lock, skipping the prune so a "
            "concurrent pod's charges cannot be swept by this run's cutoff",
            date_str,
        )
    elif rows_failed:
        # A charge that never landed leaves its row looking unrefreshed, so the prune
        # would delete the very row the failed write was meant to replace
        verbose_proxy_logger.warning(
            "PTU rollup: %d charge(s) failed for %s, skipping the prune so a row whose "
            "replacement did not land is not deleted; rerun that date to reconcile",
            rows_failed,
            date_str,
        )
    else:
        await _prune_unrefreshed_sentinel_rows(
            prisma_client,
            date_str=date_str,
            run_started=run_started,
            scanned_ids=loaded.scanned_ids,
        )

    verbose_proxy_logger.info(
        "PTU rollup for %s: %d PTU models processed, %d rows written, %d rows failed",
        date_str,
        len(ptu_models),
        rows_written,
        rows_failed,
    )
    return RollupResult(
        day=day,
        models_processed=len(ptu_models),
        rows_written=rows_written,
        rows_failed=rows_failed,
        lapsed=_lapsed_models(ptu_models, run_started),
    )


def _slack_safe(model_name: str) -> str:
    """``model_name`` with the characters Slack reads as markup escaped.

    A model name is operator-supplied and this alert is delivered to an operator channel, so an
    unescaped name could post a channel-wide mention or a disguised link.
    """
    return model_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lapsed_models(ptu_models: tuple[PTUModel, ...], now: datetime) -> tuple[str, ...]:
    """PTU deployments whose window has closed, newest bound first.

    The provider bills reserved capacity until the deployment is deleted, so a closed window
    stops this attribution without stopping the charge. The deployment is left alone: the
    window is what the operator asked to be attributed, and per-token pricing would invent a
    charge the provider does not make for reserved capacity.
    """
    return tuple(
        _slack_safe(model.model_name)
        for model in sorted(
            (m for m in ptu_models if m.effective_to is not None and m.effective_to <= now),
            key=lambda m: m.effective_to,
            reverse=True,
        )
    )


def _backfill_window(ptu_models: tuple[PTUModel, ...], end: date) -> tuple[date, ...]:
    """The UTC days the catch-up pass considers, oldest first, through ``end`` inclusive.

    Starts at the earliest declared ``ptu_effective_from``, floored at
    ``PTU_ROLLUP_MAX_BACKFILL_DAYS`` before ``end``. A start is required alongside the
    count and rate, so a deployment without one is not priced rather than being given the
    floor, which would bill it for the whole cap window. Empty when there is no PTU
    config, or when every declared window opens after ``end``.
    """
    floor: Final = end - timedelta(days=PTU_ROLLUP_MAX_BACKFILL_DAYS)
    starts: Final = tuple(model.effective_from.date() for model in ptu_models if model.effective_from)
    if not starts:
        return ()
    start: Final = max(min(starts), floor)
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


async def _existing_sentinel_keys(
    prisma_client: "PrismaClient",
    *,
    start: date,
    end: date,
) -> frozenset[tuple[str, str, str]]:
    """``(team_id, deployment id, date)`` of every PTU sentinel row within ``[start, end]``.

    The row's ``model`` column holds the deployment id, so this is an exact identity and
    survives a rename. Nothing here reads the display name.
    """
    date_range: Final = {"gte": start.isoformat(), "lte": end.isoformat()}  # mutable-ok: prisma range filter
    rows: Final = await prisma_client.db.litellm_dailyteamspend.find_many(
        where={"api_key": PTU_SENTINEL_API_KEY, "date": date_range}  # mutable-ok: prisma find filter
    )
    return frozenset(
        (
            str(getattr(row, "team_id", "") or ""),
            str(getattr(row, "model", "") or ""),
            str(getattr(row, "date", "") or ""),
        )
        for row in rows
    )


async def run_ptu_flat_cost_backfill(
    prisma_client: "PrismaClient",
    today: date | None = None,
) -> BackfillResult:
    """Price the elapsed days of every PTU window that carry no sentinel row yet.

    Writes only the charges that are missing and never rewrites or deletes an existing
    row, so a day already priced keeps the amount it was billed, whatever the config says
    now. A day counts as priced when a sentinel row exists for that deployment id, so
    renaming a deployment neither re-prices its history nor files a second charge beside
    the row already there. Zero-cost days write nothing, which leaves a day
    outside a window reconsidered on each run rather than recorded as done.

    It deletes nothing. Removing a deployment stops it accruing new charges and leaves the
    days it was billed for standing, since those days were incurred.
    """
    end: Final = (today or datetime.now(timezone.utc).date()) - timedelta(days=1)

    if not prisma_client:
        verbose_proxy_logger.warning("PTU backfill: prisma_client is None, skipping")
        return BackfillResult(start=end, end=end, days_scanned=0, rows_written=0)

    ptu_models: Final = (await _load_ptu_models(prisma_client)).models
    days: Final = _backfill_window(ptu_models, end)

    if not days:
        return BackfillResult(start=end, end=end, days_scanned=0, rows_written=0)

    priced: Final = await _existing_sentinel_keys(prisma_client, start=days[0], end=days[-1])
    missing: Final = tuple(
        (day.isoformat(), charge)
        for day in days
        for charge in _aggregate_charges(ptu_models, day)
        if (charge.team_id, charge.model_id, day.isoformat()) not in priced
    )
    if not missing:
        return BackfillResult(start=days[0], end=days[-1], days_scanned=len(days), rows_written=0)

    landed: Final = tuple(
        [
            await _upsert_charge_with_retry(prisma_client, charge=charge, date_str=date_str)
            for date_str, charge in missing
        ]
    )
    rows_written: Final = sum(landed)
    verbose_proxy_logger.info(
        "PTU backfill for %s to %s: %d unpriced charge(s) found, %d written, %d failed",
        days[0].isoformat(),
        days[-1].isoformat(),
        len(missing),
        rows_written,
        len(missing) - rows_written,
    )
    return BackfillResult(
        start=days[0],
        end=days[-1],
        days_scanned=len(days),
        rows_written=rows_written,
        rows_failed=len(missing) - rows_written,
    )


async def run_scheduled_ptu_rollup(
    prisma_client: "PrismaClient",
    pod_lock_manager: "PodLockManager | None" = None,
    target_date: date | None = None,
    alert: Callable[[str], Awaitable[None]] | None = None,
) -> RollupResult | None:
    """Run the daily rollup under a cross-pod lock so only one proxy reconciles a day.

    Every proxy process schedules this cron, and the read-charge-prune sequence is not
    atomic: two pods reading different config snapshots can have the loser's prune delete
    a row the winner just wrote. Returns None when another pod holds the lock, since that
    pod is doing the work. A deployment without a Redis-backed lock manager runs
    unguarded, as ``SpendLogCleanup`` does, and so does a run that cannot reach Redis at
    all: the lock exists to avoid duplicate work, so no lock problem may cost a day.

    The lease is a fixed TTL with no renewal, so a long scan can outlive it. That costs
    duplicate work rather than correctness: the upserts are idempotent on the sentinel
    key and the prune reads only the row's own timestamp, so a second pod arriving
    mid-run cannot corrupt the day.

    Returns None without touching the database when PTU cost attribution is off. Proxy
    startup already skips scheduling the cron, so this guards the function itself rather
    than its one caller, and a deployment that never opted in accrues nothing whatever
    reaches it.
    """
    if not is_ptu_cost_attribution_enabled():
        return None

    if pod_lock_manager is None or pod_lock_manager.redis_cache is None:
        return await _run_and_alert(prisma_client, target_date=target_date, alert=alert, may_prune=False)

    if not await pod_lock_manager.acquire_lock(cronjob_id=PTU_ROLLUP_JOB_ID, ttl=PTU_ROLLUP_LOCK_TTL_SECONDS):
        if await _lock_is_held(pod_lock_manager):
            verbose_proxy_logger.info("PTU rollup: another pod holds the rollup lock, skipping this run")
            return None
        # acquire_lock reports contention and a Redis outage the same way, so an
        # unreachable Redis would otherwise skip the day on every pod at once. The
        # reconcile is safe to run concurrently, so losing the lock costs duplicate
        # work; losing the day costs a team's charges
        verbose_proxy_logger.warning(
            "PTU rollup: could not take the rollup lock and no other pod holds it, "
            "running unguarded rather than skipping the day"
        )
        return await _run_and_alert(prisma_client, target_date=target_date, alert=alert, may_prune=False)

    try:
        return await _run_and_alert(prisma_client, target_date=target_date, alert=alert, may_prune=True)
    finally:
        await pod_lock_manager.release_lock(cronjob_id=PTU_ROLLUP_JOB_ID)


async def _lock_is_held(pod_lock_manager: "PodLockManager") -> bool:
    """True only when the rollup lock is readable and someone is holding it.

    A Redis that cannot be read is reported as "not held" so the caller runs the day
    rather than skipping it; the cost of being wrong here is a duplicate reconcile.
    """
    try:
        lock_key: Final = pod_lock_manager.get_redis_lock_key(PTU_ROLLUP_JOB_ID)
        return bool(await pod_lock_manager.redis_cache.async_get_cache(lock_key))
    except Exception as exc:  # noqa: BLE001  # an unreadable lock must not skip the day
        verbose_proxy_logger.warning("PTU rollup: could not read the rollup lock: %s", exc)
        return False


async def _run_and_alert(
    prisma_client: "PrismaClient",
    *,
    target_date: date | None,
    alert: "Callable[[str], Awaitable[None]] | None",
    may_prune: bool = True,
) -> RollupResult:
    """Reconcile the day, catch up any days left unpriced, and alert on charges that did not land.

    A charge that exhausts its retries leaves that team showing no PTU cost for the date,
    and the scheduled job moves on to the next day rather than replaying it. That is a
    silent underbill unless someone is reading proxy logs, so it is escalated to whatever
    alerting the deployment has configured.

    The catch-up pass runs only on the scheduled shape, where ``target_date`` is None. An
    explicit date means reconcile exactly that day, so it stays a single-day operation.
    Its failure is contained: the day's own result is returned either way.
    """
    result: Final = await run_ptu_flat_cost_rollup(prisma_client, target_date=target_date, may_prune=may_prune)
    if result.rows_failed:
        await _deliver_alert(
            alert,
            f"PTU flat-cost rollup for {result.day.isoformat()}: {result.rows_failed} of "
            f"{result.rows_written + result.rows_failed} team charges failed to write. Those teams show no PTU "
            f"cost for that date until the rollup is rerun for it.",
        )
    if result.lapsed:
        await _deliver_alert(
            alert,
            f"PTU flat-cost attribution has stopped for {len(result.lapsed)} deployment(s) whose effective "
            f"window has closed: {', '.join(result.lapsed[:PTU_LAPSED_ALERT_LIMIT])}. Reserved capacity is billed "
            "until the deployment is deleted, so a deployment still serving traffic is still being charged for "
            "by the provider with nothing attributing it here. Extend the window, or retire the deployment.",
        )
    if target_date is None:
        await _backfill_and_alert(prisma_client, alert=alert)
    return result


async def _backfill_and_alert(
    prisma_client: "PrismaClient",
    *,
    alert: "Callable[[str], Awaitable[None]] | None",
) -> None:
    """Catch up unpriced PTU days, alerting on charges that did not land.

    Never raises: the day's own rollup has already run and its result must reach the
    caller whatever the catch-up pass does.
    """
    try:
        backfill: Final = await run_ptu_flat_cost_backfill(prisma_client)
    except Exception as exc:  # noqa: BLE001  # the catch-up pass must not fail the day's rollup
        verbose_proxy_logger.error("PTU backfill: catch-up pass failed, the day's rollup still stands: %s", exc)
        return
    if backfill.rows_failed:
        await _deliver_alert(
            alert,
            f"PTU flat-cost backfill for {backfill.start.isoformat()} to {backfill.end.isoformat()}: "
            f"{backfill.rows_failed} of {backfill.rows_written + backfill.rows_failed} previously unpriced charges "
            f"failed to write. Those days stay unpriced until a later run picks them up.",
        )


async def _deliver_alert(alert: "Callable[[str], Awaitable[None]] | None", message: str) -> None:
    """Send an operator alert when one is configured, swallowing a broken channel."""
    if alert is None:
        return
    try:
        await alert(message)
    except Exception as exc:  # noqa: BLE001  # a broken alert channel must not fail the rollup
        verbose_proxy_logger.error("PTU rollup: could not deliver the failed-charge alert: %s", exc)


def _prune_filter(*, date_str: str, cutoff: datetime, chunk: "tuple[str, ...]") -> "Mapping[str, object]":
    """One delete statement's predicate, bounded to the deployments in ``chunk``.

    Returns a plain dict because the query builder serialises the mapping it is handed and
    rejects a read-only view of one.
    """
    return {  # mutable-ok: prisma delete filter
        "date": date_str,
        "api_key": PTU_SENTINEL_API_KEY,
        "updated_at": {"lt": cutoff},  # mutable-ok: prisma comparison filter
        "model": {"in": chunk},  # mutable-ok: prisma membership filter
    }


async def _prune_unrefreshed_sentinel_rows(
    prisma_client: "PrismaClient",
    *,
    date_str: str,
    run_started: datetime,
    scanned_ids: frozenset[str],
) -> None:
    """Delete the day's PTU sentinel rows this run looked at and did not refresh.

    Two conditions, and a row survives unless it meets both. It must be stale: every
    charge the run wrote bumps ``updated_at`` past ``run_started``, so anything left below
    that mark is a (team, model) the current config no longer prices. The mark is pulled
    back by ``PTU_PRUNE_SKEW_GRACE_SECONDS`` because the two timestamps come from
    different hosts, and the grace separates a row that is hours old from one written
    seconds ago without waiting on clocks agreeing.

    It must also be a deployment this run could see. A charge already written is a record
    of capacity that was reserved, so the only rows a run may retract are the ones it can
    reassess: a deployment it scanned and then declined to charge, because the window
    closed or the PTU config was removed. A row whose deployment is absent from every
    source the run reads is not evidence that the reservation never happened, only that
    this host cannot account for it. A deployment the router refused to register is in that
    same bucket as one that was removed, because neither reaches the scan.

    The ids go out in chunks, because each is one bind variable and the server rejects a
    statement carrying more than 32767 of them, which a proxy holding that many
    deployments would otherwise hit every night with no handler above here.
    """
    cutoff: Final = run_started - timedelta(seconds=PTU_PRUNE_SKEW_GRACE_SECONDS)
    ordered: Final = tuple(sorted(scanned_ids))
    chunks: Final = tuple(
        ordered[start : start + _PRUNE_ID_CHUNK_SIZE] for start in range(0, len(ordered), _PRUNE_ID_CHUNK_SIZE)
    )
    filters: Final = tuple(_prune_filter(date_str=date_str, cutoff=cutoff, chunk=chunk) for chunk in chunks)
    deletions: Final = tuple(
        [await prisma_client.db.litellm_dailyteamspend.delete_many(where=where) for where in filters]
    )
    deleted: Final = sum(deletions)
    if deleted:
        verbose_proxy_logger.info(
            "PTU rollup for %s: pruned %s stale sentinel row(s) of %s deployment(s) considered",
            date_str,
            deleted,
            len(ordered),
        )


__all__ = (
    "PTU_ROLLUP_JOB_ID",
    "PTU_SENTINEL_API_KEY",
    "BackfillResult",
    "PTUModel",
    "RollupResult",
    "run_ptu_flat_cost_backfill",
    "run_ptu_flat_cost_rollup",
    "run_scheduled_ptu_rollup",
)
