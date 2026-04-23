"""
DB operations for the Evals feature.
Mirrors litellm/proxy/guardrails/guardrail_registry.py
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


async def add_eval_to_db(
    eval_config: Dict[str, Any],
    prisma_client: Any,
) -> Dict[str, Any]:
    criteria = eval_config.get("criteria", [])
    created = await prisma_client.db.litellm_evalstable.create(
        data={
            "eval_name": eval_config["eval_name"],
            "criteria": criteria,
            "judge_model": eval_config["judge_model"],
            "description": eval_config.get("description"),
            "overall_threshold": eval_config.get("overall_threshold"),
            "max_iterations": eval_config.get("max_iterations", 1),
            "created_by": eval_config.get("created_by", ""),
            "updated_by": eval_config.get("updated_by", ""),
        }
    )
    row = dict(created)
    if isinstance(row.get("criteria"), str):
        row["criteria"] = json.loads(row["criteria"])
    return row


async def get_eval_by_id(eval_id: str, prisma_client: Any) -> Optional[Dict[str, Any]]:
    row = await prisma_client.db.litellm_evalstable.find_unique(
        where={"eval_id": eval_id}
    )
    if row is None:
        return None
    result = dict(row)
    if isinstance(result.get("criteria"), str):
        result["criteria"] = json.loads(result["criteria"])
    return result


async def list_evals(prisma_client: Any) -> List[Dict[str, Any]]:
    rows = await prisma_client.db.litellm_evalstable.find_many(
        order={"created_at": "desc"}
    )
    result = []
    for row in rows:
        r = dict(row)
        if isinstance(r.get("criteria"), str):
            r["criteria"] = json.loads(r["criteria"])
        result.append(r)
    return result


async def update_eval_in_db(
    eval_id: str,
    update: Dict[str, Any],
    prisma_client: Any,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"version": {"increment": 1}}
    if "criteria" in update:
        data["criteria"] = update["criteria"]
    for field in ("judge_model", "description", "overall_threshold", "max_iterations"):
        if field in update:
            data[field] = update[field]
    if "updated_by" in update:
        data["updated_by"] = update["updated_by"]
    updated = await prisma_client.db.litellm_evalstable.update(
        where={"eval_id": eval_id}, data=data
    )
    row = dict(updated)
    if isinstance(row.get("criteria"), str):
        row["criteria"] = json.loads(row["criteria"])
    return row


async def delete_eval_from_db(eval_id: str, prisma_client: Any) -> Dict[str, Any]:
    deleted = await prisma_client.db.litellm_evalstable.delete(
        where={"eval_id": eval_id}
    )
    return dict(deleted)


async def attach_eval_to_agent(
    agent_id: str,
    eval_id: str,
    params: Dict[str, Any],
    prisma_client: Any,
) -> Dict[str, Any]:
    created = await prisma_client.db.litellm_agentevalstable.create(
        data={
            "agent_id": agent_id,
            "eval_id": eval_id,
            "eval_name": params.get("eval_name", ""),
            "on_failure": params.get("on_failure", "block"),
            "overall_threshold_override": params.get("overall_threshold_override"),
            "created_by": params.get("created_by", ""),
        }
    )
    return dict(created)


async def detach_eval_from_agent(
    agent_id: str, eval_id: str, prisma_client: Any
) -> Dict[str, Any]:
    deleted = await prisma_client.db.litellm_agentevalstable.delete(
        where={"agent_id_eval_id": {"agent_id": agent_id, "eval_id": eval_id}}
    )
    return dict(deleted)


async def get_evals_for_agent(
    agent_id: str, prisma_client: Any
) -> List[Dict[str, Any]]:
    rows = await prisma_client.db.litellm_agentevalstable.find_many(
        where={"agent_id": agent_id},
        include={"eval": True},
    )
    result = []
    for row in rows:
        r = dict(row)
        eval_obj = r.get("eval")
        if eval_obj is not None:
            eval_dict = dict(eval_obj) if not isinstance(eval_obj, dict) else eval_obj
            if isinstance(eval_dict.get("criteria"), str):
                eval_dict["criteria"] = json.loads(eval_dict["criteria"])
            r["eval"] = eval_dict
        result.append(r)
    return result
