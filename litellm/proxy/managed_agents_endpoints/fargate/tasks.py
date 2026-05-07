"""ECS Fargate task lifecycle for managed agent sandboxes."""

import asyncio
import time
from typing import Any, Dict, List, Optional

import boto3
import httpx
from botocore.exceptions import ClientError

from litellm._logging import verbose_proxy_logger

TAG_SESSION_ID = "litellm:managed_agent_session_id"
TAG_AGENT_ID = "litellm:managed_agent_id"


_clients: Dict[str, Any] = {}


def _ecs(region: str):
    key = f"ecs:{region}"
    if key not in _clients:
        _clients[key] = boto3.client("ecs", region_name=region)
    return _clients[key]


def _ec2(region: str):
    key = f"ec2:{region}"
    if key not in _clients:
        _clients[key] = boto3.client("ec2", region_name=region)
    return _clients[key]


def run_task_sync(
    *,
    region: str,
    cluster: str,
    task_def_arn: str,
    container_name: str,
    subnet: str,
    security_group: str,
    env: Dict[str, str],
    session_id: str,
    agent_id: str,
) -> str:
    """Launch Fargate task. Tags task w/ session_id + agent_id for orphan reconciliation."""
    overrides = {
        "containerOverrides": [
            {
                "name": container_name,
                "environment": [{"name": k, "value": v} for k, v in env.items()],
            }
        ]
    }
    r = _ecs(region).run_task(
        cluster=cluster,
        taskDefinition=task_def_arn,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": [subnet],
                "securityGroups": [security_group],
                "assignPublicIp": "ENABLED",
            }
        },
        overrides=overrides,
        tags=[
            {"key": TAG_SESSION_ID, "value": session_id},
            {"key": TAG_AGENT_ID, "value": agent_id},
        ],
        propagateTags="TASK_DEFINITION",
        count=1,
    )
    if r.get("failures"):
        raise RuntimeError(f"run_task failures: {r['failures']}")
    return r["tasks"][0]["taskArn"]


def stop_task_sync(
    region: str, cluster: str, task_arn: str, reason: str = "litellm cleanup"
) -> None:
    """Best-effort stop. Idempotent — swallows ClientError on already-stopped tasks."""
    try:
        _ecs(region).stop_task(cluster=cluster, task=task_arn, reason=reason)
    except ClientError as e:
        verbose_proxy_logger.warning(f"stop_task failed for {task_arn}: {e}")


def wait_running_get_ip_sync(
    region: str, cluster: str, task_arn: str, timeout: int = 300
) -> str:
    deadline = time.time() + timeout
    ecs_client = _ecs(region)
    ec2_client = _ec2(region)
    while time.time() < deadline:
        d = ecs_client.describe_tasks(cluster=cluster, tasks=[task_arn])["tasks"][0]
        st = d["lastStatus"]
        if st == "STOPPED":
            reasons = [c.get("reason") for c in d.get("containers", [])]
            raise RuntimeError(
                f"task stopped: {d.get('stoppedReason')} containers={reasons}"
            )
        if st == "RUNNING":
            for att in d.get("attachments", []):
                eni = next(
                    (
                        kv["value"]
                        for kv in att.get("details", [])
                        if kv["name"] == "networkInterfaceId"
                    ),
                    None,
                )
                if eni:
                    ni = ec2_client.describe_network_interfaces(
                        NetworkInterfaceIds=[eni]
                    )["NetworkInterfaces"][0]
                    ip = ni.get("Association", {}).get("PublicIp")
                    if ip:
                        return ip
        time.sleep(3)
    raise TimeoutError("task never reached RUNNING with public IP")


async def wait_http_ready(
    url: str, client: httpx.AsyncClient, timeout: int = 240
) -> None:
    deadline = time.time() + timeout
    last: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = await client.get(url, timeout=3)
            if r.status_code < 500:
                return
        except (httpx.HTTPError, OSError) as e:
            last = e
        await asyncio.sleep(2)
    raise TimeoutError(f"sandbox never ready at {url}: {last}")


def list_tagged_task_arns(region: str, cluster: str) -> List[str]:
    """All task ARNs in cluster regardless of status. Reconciler filters by tag."""
    ecs_client = _ecs(region)
    arns: List[str] = []
    for status in ("RUNNING", "PENDING"):
        paginator = ecs_client.get_paginator("list_tasks")
        for page in paginator.paginate(cluster=cluster, desiredStatus=status):
            arns.extend(page.get("taskArns", []))
    return arns


def describe_tasks_with_tags(
    region: str, cluster: str, task_arns: List[str]
) -> List[Dict[str, Any]]:
    """Returns list of {taskArn, tags: {key: value}} for arns. Batches in 100s (ECS API limit)."""
    if not task_arns:
        return []
    ecs_client = _ecs(region)
    out: List[Dict[str, Any]] = []
    for i in range(0, len(task_arns), 100):
        batch = task_arns[i : i + 100]
        r = ecs_client.describe_tasks(cluster=cluster, tasks=batch, include=["TAGS"])
        for t in r.get("tasks", []):
            tags = {tag["key"]: tag["value"] for tag in t.get("tags", [])}
            out.append(
                {
                    "taskArn": t["taskArn"],
                    "tags": tags,
                    "lastStatus": t.get("lastStatus"),
                }
            )
    return out
