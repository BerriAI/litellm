import asyncio
import json
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    delete_verification_tokens,
    generate_key_helper_fn,
)
from litellm.proxy.managed_agents_endpoints import config_loader as _config_loader
from litellm.proxy.managed_agents_endpoints.endpoints import (
    _assert_owner_or_admin,
    _is_admin,
    router,
)
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
from litellm.proxy.managed_agents_endpoints.endpoints_passthrough import (
    _SESSION_META_CACHE,
    invalidate_session_meta_cache,
)
from litellm.proxy.managed_agents_endpoints.lifecycle import stop_session_task
from litellm.proxy.managed_agents_endpoints.types import (
    AwsOverrides,
    SessionCreateIn,
    SessionOut,
)
from litellm.proxy.managed_agents_endpoints.warm_pool import (
    post_claim as _warm_pool_post_claim,
    schedule_refill as _warm_pool_schedule_refill,
    try_claim as _warm_pool_try_claim,
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


async def _mint_session_key(
    user_api_key_dict: UserAPIKeyAuth,
    agent_id: str,
    agent_model: str,
    session_id: str,
) -> tuple[str, str]:
    """Mint a session-scoped LiteLLM key for a managed-agent sandbox.

    Returns (plaintext_key, token_hash). The hash is what's stored on the
    session row; the plaintext is injected as LITELLM_API_KEY into the
    container env and never persisted.
    """
    key_data = await generate_key_helper_fn(
        request_type="key",
        duration=None,
        models=[agent_model],
        user_id=user_api_key_dict.user_id,
        team_id=user_api_key_dict.team_id,
        agent_id=agent_id,
        key_alias=f"managed-agent-session-{session_id}",
        metadata={
            "managed_agent_id": agent_id,
            "managed_agent_session_id": session_id,
        },
    )
    plaintext = key_data.get("token")
    token_hash = key_data.get("token_id")
    if not plaintext or not token_hash:
        raise RuntimeError(
            "generate_key_helper_fn returned no token/token_id for session "
            f"{session_id}"
        )
    return plaintext, token_hash


def _region_from_arn(arn: Optional[str]) -> Optional[str]:
    """Extract region from an AWS ARN.

    ARN format: arn:<partition>:<service>:<region>:<account>:<resource>
    Returns None for malformed input so callers can fall back to global config.
    """
    if not arn:
        return None
    parts = arn.split(":", 5)
    if len(parts) < 4 or not parts[3]:
        return None
    return parts[3]


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
    _t0 = time.monotonic()

    def _ckpt(label: str) -> None:
        verbose_proxy_logger.warning(
            f"PERF create_session agent={agent_id} {label} t+{time.monotonic() - _t0:.2f}s"
        )

    agent = await prisma_client.db.litellm_managedagenttable.find_unique(
        where={"agent_id": agent_id}, include={"template": True}
    )
    _ckpt("agent_fetched")
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")
    _assert_owner_or_admin(user_api_key_dict, agent.created_by, "agent", agent_id)

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

    aws_overrides = _resolve_aws_overrides()
    cluster = _resolve_cluster(aws_overrides)
    # Region must match the template's task-def ARN. Falling back to global
    # config produces "Invalid Region in ARN" when a template was built in a
    # different region than the proxy's current default.
    region = _region_from_arn(template.task_def_arn) or _resolve_region()

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
    _ckpt(f"session_row_created session={session_id}")

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=None, write=None, pool=10)
    )

    task_arn: Optional[str] = None
    session_key_hash: Optional[str] = None
    try:
        # Mint a session-scoped LiteLLM key for the sandbox container instead
        # of passing the user's own key through. The key is scoped to the
        # agent's model only, aliased by session_id, and revoked when the
        # session stops. Done inside try so a mint failure cleans up the
        # session row via _mark_session_failed.
        session_key, session_key_hash = await _mint_session_key(
            user_api_key_dict=user_api_key_dict,
            agent_id=agent_id,
            agent_model=agent.model,
            session_id=session_id,
        )
        _ckpt("session_key_minted")
        await prisma_client.db.litellm_managedagentsessiontable.update(
            where={"session_id": session_id},
            data={"virtual_key_hash": session_key_hash},
        )
        _ckpt("virtual_key_hash_persisted")

        env: Dict[str, str] = {
            "LITELLM_API_KEY": session_key,
            "LITELLM_API_BASE": (
                metadata.get("litellm_api_base")
                or os.environ.get("LITELLM_API_BASE")
                or ""
            ),
            "LITELLM_DEFAULT_MODEL": agent.model,
        }
        if template.repo_url:
            env["REPO_URL"] = template.repo_url
            branch = agent.branch or template.default_branch
            if branch:
                env["BRANCH"] = branch
        git_token = await decrypt_git_token(prisma_client, template.git_credential_id)
        if git_token:
            env["GIT_TOKEN"] = git_token
        if agent.prompt:
            env["AGENT_PROMPT"] = agent.prompt

        # Warm-pool fast path: try to claim a pre-spawned Fargate task whose
        # shim is already listening. On success we skip RunTask + ENI wait +
        # ECR pull entirely (~30-60s saved). On any failure (no slot, claim
        # POST 4xx/5xx/timeout) we fall through to the cold path below.
        public_ip: Optional[str] = None
        cfg = _config_loader.MANAGED_AGENTS_CONFIG
        pool_enabled = bool(cfg and getattr(cfg, "pool_enabled", False))
        pool_min_warm = int(getattr(cfg, "pool_min_warm", 1)) if cfg else 1
        if pool_enabled:
            slot = await _warm_pool_try_claim(template.template_id)
            _ckpt(
                f"warm_pool_claim slot={'hit' if slot else 'miss'} "
                f"template={template.template_id}"
            )
            if slot is not None:
                claimed_arn, claimed_ip, claimed_port, claimed_secret = slot
                try:
                    await _warm_pool_post_claim(
                        public_ip=claimed_ip,
                        container_port=claimed_port,
                        secret=claimed_secret,
                        env=env,
                        timeout=240.0,
                        client=client,
                    )
                    _ckpt(f"warm_pool_post_claim_done ip={claimed_ip}")
                    task_arn = claimed_arn
                    public_ip = claimed_ip
                    verbose_proxy_logger.info(
                        f"managed_agents: warm-pool hit template={template.template_id} "
                        f"session={session_id} ip={public_ip}"
                    )
                except Exception as e:
                    verbose_proxy_logger.warning(
                        "managed_agents: warm-pool claim failed, falling back "
                        f"(template={template.template_id} arn={claimed_arn}): {e}"
                    )
                    await asyncio.to_thread(
                        stop_task_sync,
                        region,
                        cluster,
                        claimed_arn,
                        "warm_pool claim failed",
                    )
                    public_ip = None
                    task_arn = None
                finally:
                    _warm_pool_schedule_refill(
                        template=template,
                        region=region,
                        aws_overrides=aws_overrides,
                        cluster=cluster,
                        min_warm=pool_min_warm,
                    )

        if public_ip is None:
            infra = await asyncio.to_thread(
                bootstrap_shared_infra, region, aws_overrides, template.container_port
            )
            _ckpt("bootstrap_done")
            if not infra.subnet_ids:
                raise RuntimeError("bootstrap_shared_infra returned no subnets")
            subnet = infra.subnet_ids[0]
            security_group = infra.security_group_id

            shim_secret = secrets.token_urlsafe(32)
            task_arn = await asyncio.to_thread(
                run_task_sync,
                region=region,
                cluster=cluster,
                task_def_arn=template.task_def_arn,
                container_name="harness",
                subnet=subnet,
                security_group=security_group,
                env={"SHIM_SECRET": shim_secret},
                session_id=session_id,
                agent_id=agent_id,
            )
            _ckpt(f"run_task_done arn={task_arn}")

            public_ip = await asyncio.to_thread(
                wait_running_get_ip_sync, region, cluster, task_arn, 300
            )
            _ckpt(f"wait_running_done ip={public_ip}")

            shim_port = template.container_port + 1
            shim_health = f"http://{public_ip}:{shim_port}/healthz"
            await wait_http_ready(shim_health, client, timeout=180)
            _ckpt("shim_ready")
            await _warm_pool_post_claim(
                public_ip=public_ip,
                container_port=template.container_port,
                secret=shim_secret,
                env=env,
                timeout=240.0,
                client=client,
            )
            _ckpt("cold_post_claim_done")
        # NOTE (v1): proxy↔sandbox traffic is plain HTTP over the task's public IP.
        # Tracked for follow-up: route through PrivateLink/VPC-internal addressing
        # or terminate TLS on the harness so prompts/responses and the env-injected
        # LITELLM_API_KEY do not transit the public internet in cleartext.
        sandbox_url = f"http://{public_ip}:{template.container_port}"

        await wait_http_ready(sandbox_url, client, timeout=600)
        _ckpt("sandbox_http_ready")

        title = (body.title if body else None) or agent.agent_name or "default"
        harness_session_id = await harness_create_session(
            sandbox_url, client, title=title
        )
        _ckpt(f"harness_session_created hsid={harness_session_id}")

        await prisma_client.db.litellm_managedagentsessiontable.update(
            where={"session_id": session_id},
            data={
                "task_arn": task_arn,
                "sandbox_url": sandbox_url,
                "harness_session_id": harness_session_id,
                "status": "ready",
                "last_seen_at": _now_utc(),
            },
        )
        _ckpt("session_marked_ready")
        _SESSION_META_CACHE[session_id] = {
            "status": "ready",
            "sandbox_url": sandbox_url,
            "harness_session_id": harness_session_id,
            "created_by": user_api_key_dict.user_id,
            "model": agent.model,
            "_cacheable": True,
        }

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
        if session_key_hash:
            await _revoke_session_key(
                session_id=session_id,
                token_hash=session_key_hash,
                user_api_key_dict=user_api_key_dict,
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
    if not _is_admin(user_api_key_dict):
        # Non-admin callers see only their own rows. If the API key has no
        # user_id, treat it as "no rows" rather than exposing every session.
        if user_api_key_dict.user_id is None:
            return []
        where["created_by"] = user_api_key_dict.user_id

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
    _assert_owner_or_admin(user_api_key_dict, row.created_by, "session", session_id)
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
    _assert_owner_or_admin(user_api_key_dict, row.created_by, "session", session_id)

    aws_overrides = _resolve_aws_overrides()
    cluster = row.fargate_cluster or _resolve_cluster(aws_overrides)
    # Stop the task in the same region it runs in — the session's task ARN
    # encodes that region. The template's task-def ARN is the next-best fallback
    # for sessions created before task_arn was persisted.
    region = (
        _region_from_arn(row.task_arn)
        or _region_from_arn(getattr(row, "fargate_task_def_arn", None))
        or _resolve_region()
    )

    if row.task_arn:
        await stop_session_task(
            region=region,
            cluster=cluster,
            task_arn=row.task_arn,
            session_id=session_id,
        )

    if row.virtual_key_hash:
        await _revoke_session_key(
            session_id=session_id,
            token_hash=row.virtual_key_hash,
            user_api_key_dict=user_api_key_dict,
        )

    await prisma_client.db.litellm_managedagentsessiontable.update(
        where={"session_id": session_id},
        data={"status": "dead", "stopped_at": _now_utc()},
    )
    invalidate_session_meta_cache(session_id)

    return {"id": session_id, "status": "dead"}


async def _revoke_session_key(
    session_id: str,
    token_hash: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """Revoke a session-scoped LiteLLM key. Best-effort: a failure here must
    not block session teardown."""
    from litellm.proxy.proxy_server import user_api_key_cache

    try:
        await delete_verification_tokens(
            tokens=[token_hash],
            user_api_key_cache=user_api_key_cache,
            user_api_key_dict=user_api_key_dict,
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            "managed_agents: failed to revoke session key for session=%s: %s",
            session_id,
            e,
        )
