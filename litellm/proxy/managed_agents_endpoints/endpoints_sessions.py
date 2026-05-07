import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.managed_agents_endpoints import config_loader as _config_loader
from litellm.proxy.managed_agents_endpoints.endpoints import router
from litellm.proxy.managed_agents_endpoints.fargate.bootstrap import (
    bootstrap_shared_infra,
)
from litellm.proxy.managed_agents_endpoints.fargate.tasks import (
    run_task_sync,
    stop_task_sync,
    wait_http_ready,
    wait_running_get_ip_sync,
)
from litellm.proxy.managed_agents_endpoints.git_validation import decrypt_git_token
from litellm.proxy.managed_agents_endpoints.harness_client import (
    harness_create_session,
    harness_send_message,
)
from litellm.proxy.managed_agents_endpoints.lifecycle import stop_session_task
from litellm.proxy.managed_agents_endpoints.types import (
    AwsOverrides,
    SessionCreateIn,
    SessionOut,
)
from litellm.proxy.utils import jsonify_object


def _session_row_to_out(row, response: Optional[Dict[str, Any]] = None) -> SessionOut:
    return SessionOut(
        id=row.session_id,
        agent_id=row.agent_id,
        sandbox_url=row.sandbox_url,
        status=row.status,
        task_arn=row.task_arn,
        response=response,
        created_at=getattr(row, "created_at", None),
    )


def _resolve_region() -> str:
    cfg = _config_loader.MANAGED_AGENTS_CONFIG
    if cfg is not None and cfg.aws_region:
        return cfg.aws_region
    return "us-east-1"


def _resolve_aws_overrides() -> AwsOverrides:
    cfg = _config_loader.MANAGED_AGENTS_CONFIG
    if cfg is not None:
        return cfg.aws
    return AwsOverrides()


def _resolve_cluster(aws_overrides: AwsOverrides) -> str:
    return aws_overrides.cluster or "litellm-agents"


def _coerce_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def _mark_session_failed(
    prisma_client: Any, session_id: str, failure_reason: str
) -> None:
    try:
        await prisma_client.db.litellm_managedagentsessiontable.update(
            where={"session_id": session_id},
            data={
                "status": "failed",
                "failure_reason": failure_reason,
                "stopped_at": _now_utc(),
            },
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            f"managed_agents: failed to mark session {session_id} failed: {e}"
        )


@router.post("/agents/{agent_id}/session", response_model=SessionOut)
async def create_session(
    agent_id: str,
    body: Optional[SessionCreateIn] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionOut:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    body = body or SessionCreateIn()

    agent = await prisma_client.db.litellm_managedagenttable.find_unique(
        where={"agent_id": agent_id}, include={"template": True}
    )
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    template = getattr(agent, "template", None)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"template for agent '{agent_id}' not found",
        )

    if template.build_status != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f"template '{template.template_id}' is not ready "
                f"(build_status={template.build_status})"
            ),
        )

    if not template.task_def_arn:
        raise HTTPException(
            status_code=409,
            detail=f"template '{template.template_id}' has no task_def_arn",
        )

    region = _resolve_region()
    aws_overrides = _resolve_aws_overrides()
    cluster = _resolve_cluster(aws_overrides)

    metadata = _coerce_metadata(getattr(agent, "metadata", None))

    session_create_data = jsonify_object(
        {
            "agent_id": agent_id,
            "status": "creating",
            "fargate_cluster": cluster,
            "fargate_task_def_arn": template.task_def_arn,
            "created_by": user_api_key_dict.user_id,
            "team_id": user_api_key_dict.team_id,
        }
    )

    row = await prisma_client.db.litellm_managedagentsessiontable.create(
        data=session_create_data
    )
    session_id = row.session_id

    env: Dict[str, str] = {
        "LITELLM_API_KEY": metadata.get("litellm_api_key", "") or "",
        "LITELLM_API_BASE": metadata.get("litellm_api_base", "") or "",
        "LITELLM_DEFAULT_MODEL": agent.model,
        "REPO_URL": template.repo_url,
        "BRANCH": agent.branch or template.default_branch,
    }
    git_token = await decrypt_git_token(prisma_client, template.git_credential_id)
    if git_token:
        env["GIT_TOKEN"] = git_token
    if agent.prompt:
        env["AGENT_PROMPT"] = agent.prompt

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=None, write=None, pool=10)
    )

    task_arn: Optional[str] = None
    try:
        infra = await asyncio.to_thread(
            bootstrap_shared_infra, region, aws_overrides, template.container_port
        )
        if not infra.subnet_ids:
            raise RuntimeError("bootstrap_shared_infra returned no subnets")
        subnet = infra.subnet_ids[0]
        security_group = infra.security_group_id

        task_arn = await asyncio.to_thread(
            run_task_sync,
            region=region,
            cluster=cluster,
            task_def_arn=template.task_def_arn,
            container_name="harness",
            subnet=subnet,
            security_group=security_group,
            env=env,
            session_id=session_id,
            agent_id=agent_id,
        )
        await prisma_client.db.litellm_managedagentsessiontable.update(
            where={"session_id": session_id},
            data={"task_arn": task_arn},
        )

        public_ip = await asyncio.to_thread(
            wait_running_get_ip_sync, region, cluster, task_arn, 300
        )
        sandbox_url = f"http://{public_ip}:{template.container_port}"

        await wait_http_ready(sandbox_url, client, timeout=600)

        title = (body.title if body else None) or agent.agent_name or "default"
        harness_session_id = await harness_create_session(
            sandbox_url, client, title=title
        )

        await prisma_client.db.litellm_managedagentsessiontable.update(
            where={"session_id": session_id},
            data={
                "sandbox_url": sandbox_url,
                "harness_session_id": harness_session_id,
                "status": "ready",
                "last_seen_at": _now_utc(),
            },
        )

        response_body: Optional[Dict[str, Any]] = None
        if body and body.initial_prompt:
            response_body = await harness_send_message(
                sandbox_url,
                harness_session_id,
                client,
                model=agent.model,
                parts=[{"type": "text", "text": body.initial_prompt}],
            )

        return SessionOut(
            id=session_id,
            agent_id=agent_id,
            sandbox_url=sandbox_url,
            status="ready",
            task_arn=task_arn,
            response=response_body,
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "managed_agents: create_session failed for agent=%s session=%s: %s",
            agent_id,
            session_id,
            e,
        )
        await _mark_session_failed(prisma_client, session_id, str(e))
        if task_arn:
            try:
                await asyncio.to_thread(
                    stop_task_sync, region, cluster, task_arn, "session create failed"
                )
            except Exception as stop_err:
                verbose_proxy_logger.warning(
                    f"managed_agents: stop_task after failure raised: {stop_err}"
                )
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"session create failed: {e}")
    finally:
        await client.aclose()


@router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(
    agent_id: Optional[str] = Query(default=None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[SessionOut]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    where: Dict[str, Any] = {}
    if agent_id is not None:
        where["agent_id"] = agent_id

    rows = await prisma_client.db.litellm_managedagentsessiontable.find_many(
        where=where, order={"created_at": "desc"}
    )
    return [_session_row_to_out(row) for row in rows]


@router.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionOut:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsessiontable.find_unique(
        where={"session_id": session_id}
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
    return _session_row_to_out(row)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, str]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsessiontable.find_unique(
        where={"session_id": session_id}
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")

    region = _resolve_region()
    aws_overrides = _resolve_aws_overrides()
    cluster = row.fargate_cluster or _resolve_cluster(aws_overrides)

    if row.task_arn:
        await stop_session_task(
            region=region,
            cluster=cluster,
            task_arn=row.task_arn,
            session_id=session_id,
        )

    await prisma_client.db.litellm_managedagentsessiontable.update(
        where={"session_id": session_id},
        data={"status": "dead", "stopped_at": _now_utc()},
    )

    return {"id": session_id, "status": "dead"}
