"""
Track guardrail and policy usage for the dashboard: upsert daily metrics and
insert into SpendLogGuardrailIndex when spend logs are written.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient


def _guardrail_status_to_action(status: Optional[str]) -> str:
    """Map StandardLogging guardrail_status to blocked/passed/flagged."""
    if not status:
        return "passed"
    s = (status or "").lower()
    if "intervened" in s or "block" in s:
        return "blocked"
    if "fail" in s or "error" in s:
        return "flagged"
    return "passed"


def _parse_guardrail_info_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract guardrail_information from spend log payload metadata."""
    meta = payload.get("metadata")
    if not meta:
        return []
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(meta, dict):
        return []
    info = meta.get("guardrail_information") or meta.get(
        "standard_logging_guardrail_information"
    )
    if not isinstance(info, list):
        return []
    return info


def _date_str(dt: datetime) -> str:
    """YYYY-MM-DD in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


async def process_spend_logs_guardrail_usage(
    prisma_client: PrismaClient,
    logs_to_process: List[Dict[str, Any]],
) -> None:
    """
    After spend logs are written: update DailyGuardrailMetrics and insert
    SpendLogGuardrailIndex rows from guardrail_information in each payload.
    """
    if not logs_to_process:
        return
    # Aggregate daily metrics by (guardrail_id, date). Latency/score metrics dropped.
    daily_guardrail: Dict[tuple, Dict[str, Any]] = defaultdict(
        lambda: {
            "requests_evaluated": 0,
            "passed_count": 0,
            "blocked_count": 0,
            "flagged_count": 0,
        }
    )
    index_rows: List[Dict[str, Any]] = []

    for payload in logs_to_process:
        request_id = payload.get("request_id")
        start_time = payload.get("startTime")
        if not request_id or not start_time:
            continue
        if isinstance(start_time, str):
            try:
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
        date_key = _date_str(start_time)

        for entry in _parse_guardrail_info_from_payload(payload):
            guardrail_id = entry.get("guardrail_id") or entry.get("guardrail_name") or ""
            if not guardrail_id:
                continue
            key = (guardrail_id, date_key)
            daily_guardrail[key]["requests_evaluated"] += 1
            action = _guardrail_status_to_action(entry.get("guardrail_status"))
            if action == "passed":
                daily_guardrail[key]["passed_count"] += 1
            elif action == "blocked":
                daily_guardrail[key]["blocked_count"] += 1
            else:
                daily_guardrail[key]["flagged_count"] += 1
            policy_id = entry.get("policy_id")
            index_rows.append({
                "request_id": request_id,
                "guardrail_id": guardrail_id,
                "policy_id": policy_id,
                "start_time": start_time,
            })

    if not daily_guardrail and not index_rows:
        return

    try:
        # Insert index rows (skip duplicates by request_id + guardrail_id)
        if index_rows:
            index_data = []
            for r in index_rows:
                st = r["start_time"]
                if isinstance(st, str):
                    try:
                        st = datetime.fromisoformat(st.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        continue
                index_data.append({
                    "request_id": r["request_id"],
                    "guardrail_id": r["guardrail_id"],
                    "policy_id": r.get("policy_id"),
                    "start_time": st,
                })
            try:
                await prisma_client.db.litellm_spendlogguardrailindex.create_many(
                    data=index_data,
                    skip_duplicates=True,
                )
            except Exception as e:
                verbose_proxy_logger.debug(
                    "Guardrail usage tracking: index create_many skipped: %s", e
                )

        # Upsert daily guardrail metrics (counts only; latency/score dropped)
        for (guardrail_id, date_key), agg in daily_guardrail.items():
            n = int(agg["requests_evaluated"])
            if n == 0:
                continue
            await prisma_client.db.litellm_dailyguardrailmetrics.upsert(
                where={
                    "guardrail_id_date": {
                        "guardrail_id": guardrail_id,
                        "date": date_key,
                    }
                },
                data={
                    "create": {
                        "guardrail_id": guardrail_id,
                        "date": date_key,
                        "requests_evaluated": n,
                        "passed_count": int(agg["passed_count"]),
                        "blocked_count": int(agg["blocked_count"]),
                        "flagged_count": int(agg["flagged_count"]),
                    },
                    "update": {
                        "requests_evaluated": {"increment": n},
                        "passed_count": {"increment": int(agg["passed_count"])},
                        "blocked_count": {"increment": int(agg["blocked_count"])},
                        "flagged_count": {"increment": int(agg["flagged_count"])},
                    },
                },
            )
    except Exception as e:
        verbose_proxy_logger.warning(
            "Guardrail usage tracking failed (non-fatal): %s", e
        )
