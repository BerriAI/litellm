"""Top-level orchestrator: bootstrap shared infra, build/push image, register task def."""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.managed_agents_endpoints.fargate.bootstrap import (
    SharedInfra,
    bootstrap_shared_infra,
)
from litellm.proxy.managed_agents_endpoints.fargate.registry import (
    build_and_push,
    compute_dockerfile_hash,
)
from litellm.proxy.managed_agents_endpoints.fargate.tasks import _ecs as _ecs_client
from litellm.proxy.managed_agents_endpoints.types import AwsOverrides

_DOCKERFILE_ID_SANITIZE_RE = re.compile(r"[^a-z0-9_-]")


@dataclass(frozen=True)
class ProvisionedTemplate:
    image_uri: str
    task_def_arn: str
    image_hash: str
    container_port: int
    cluster_arn: str
    security_group_id: str
    subnet_ids: List[str]


_locks: Dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


async def _get_lock(dockerfile_id: str) -> asyncio.Lock:
    async with _locks_guard:
        if dockerfile_id not in _locks:
            _locks[dockerfile_id] = asyncio.Lock()
        return _locks[dockerfile_id]


def _sanitize_dockerfile_id(dockerfile_id: str) -> str:
    return _DOCKERFILE_ID_SANITIZE_RE.sub("-", dockerfile_id.lower())


def _register_task_definition(
    *,
    region: str,
    family: str,
    image_uri: str,
    container_port: int,
    shared_infra: SharedInfra,
) -> str:
    ecs = _ecs_client(region)
    r = ecs.register_task_definition(
        family=family,
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu="512",
        memory="1024",
        executionRoleArn=shared_infra.task_exec_role_arn,
        runtimePlatform={
            "cpuArchitecture": "X86_64",
            "operatingSystemFamily": "LINUX",
        },
        containerDefinitions=[
            {
                "name": "harness",
                "image": image_uri,
                "essential": True,
                "portMappings": [{"containerPort": container_port, "protocol": "tcp"}],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": shared_infra.log_group_name,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": "harness",
                    },
                },
            }
        ],
    )
    return r["taskDefinition"]["taskDefinitionArn"]


async def provision_template(
    *,
    dockerfile_id: str,
    dockerfile_path: str,
    context_dir: Optional[str],
    container_port: int,
    region: str,
    aws_overrides: AwsOverrides,
    log_callback: Optional[Callable[[str], None]] = None,
) -> ProvisionedTemplate:
    """Bootstrap shared infra → build/push image → register task def. Idempotent."""
    lock = await _get_lock(dockerfile_id)
    verbose_proxy_logger.debug(
        f"provision_template waiting for lock dockerfile_id={dockerfile_id}"
    )
    async with lock:
        verbose_proxy_logger.debug(
            f"provision_template acquired lock dockerfile_id={dockerfile_id}"
        )
        try:
            ctx_dir = (
                context_dir
                if context_dir
                else os.path.dirname(os.path.abspath(dockerfile_path))
            )

            image_hash = await asyncio.to_thread(
                compute_dockerfile_hash, dockerfile_path, ctx_dir
            )

            shared_infra = await asyncio.to_thread(
                bootstrap_shared_infra, region, aws_overrides, container_port
            )

            sanitized = _sanitize_dockerfile_id(dockerfile_id)
            repo_name = f"litellm-agents-{sanitized}"
            family = f"litellm-agents-{sanitized}"

            verbose_proxy_logger.info(
                f"provision_template build start dockerfile_id={dockerfile_id} "
                f"hash={image_hash[:12]} repo={repo_name}"
            )
            image_uri = await asyncio.to_thread(
                build_and_push,
                region=region,
                repo_name=repo_name,
                dockerfile_path=dockerfile_path,
                context_dir=ctx_dir,
                content_hash=image_hash,
                log_callback=log_callback,
            )

            task_def_arn = await asyncio.to_thread(
                _register_task_definition,
                region=region,
                family=family,
                image_uri=image_uri,
                container_port=container_port,
                shared_infra=shared_infra,
            )
            verbose_proxy_logger.info(
                f"provision_template register-task-def complete family={family} "
                f"task_def_arn={task_def_arn}"
            )

            return ProvisionedTemplate(
                image_uri=image_uri,
                task_def_arn=task_def_arn,
                image_hash=image_hash,
                container_port=container_port,
                cluster_arn=shared_infra.cluster_arn,
                security_group_id=shared_infra.security_group_id,
                subnet_ids=shared_infra.subnet_ids,
            )
        finally:
            verbose_proxy_logger.debug(
                f"provision_template released lock dockerfile_id={dockerfile_id}"
            )
