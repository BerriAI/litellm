"""Idempotent bootstrap for shared Fargate infrastructure used by managed agents."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.managed_agents_endpoints.fargate.tasks import _ec2, _ecs
from litellm.proxy.managed_agents_endpoints.types import AwsOverrides

DEFAULT_CLUSTER_NAME = "litellm-agents"
DEFAULT_TASK_EXEC_ROLE_NAME = "litellm-agents-task-exec"
DEFAULT_SECURITY_GROUP_NAME = "litellm-agents-sg"
DEFAULT_LOG_GROUP_NAME = "/ecs/litellm-agents"

TASK_EXEC_ROLE_POLICY_ARN = (
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
)

_TASK_EXEC_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


_bootstrap_clients: Dict[str, Any] = {}


def _iam():
    if "iam" not in _bootstrap_clients:
        _bootstrap_clients["iam"] = boto3.client("iam")
    return _bootstrap_clients["iam"]


def _logs(region: str):
    key = f"logs:{region}"
    if key not in _bootstrap_clients:
        _bootstrap_clients[key] = boto3.client("logs", region_name=region)
    return _bootstrap_clients[key]


@dataclass(frozen=True)
class SharedInfra:
    cluster_arn: str
    task_exec_role_arn: str
    security_group_id: str
    log_group_name: str
    vpc_id: str
    subnet_ids: List[str]


def ensure_cluster(region: str, cluster_name: str) -> str:
    ecs = _ecs(region)
    r = ecs.describe_clusters(clusters=[cluster_name])
    clusters = r.get("clusters", [])
    if clusters and clusters[0].get("status") == "ACTIVE":
        verbose_proxy_logger.debug(
            f"ECS cluster {cluster_name} already ACTIVE in {region}"
        )
        return clusters[0]["clusterArn"]

    verbose_proxy_logger.info(f"Creating ECS cluster {cluster_name} in {region}")
    created = ecs.create_cluster(clusterName=cluster_name)
    arn = created.get("cluster", {}).get("clusterArn")
    if arn:
        return arn
    r2 = ecs.describe_clusters(clusters=[cluster_name])
    return r2["clusters"][0]["clusterArn"]


def ensure_task_exec_role(role_name: str) -> str:
    iam = _iam()
    try:
        arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        verbose_proxy_logger.debug(f"IAM role {role_name} already exists")
        return arn
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise

    verbose_proxy_logger.info(f"Creating IAM role {role_name}")
    arn = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(_TASK_EXEC_TRUST_POLICY),
        Description="ECS task execution role for litellm managed agents",
    )["Role"]["Arn"]
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn=TASK_EXEC_ROLE_POLICY_ARN,
    )
    time.sleep(10)
    return arn


def discover_vpc_subnet(region: str) -> Tuple[str, str, str]:
    ec2 = _ec2(region)
    r = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    vpcs = r.get("Vpcs", [])
    if not vpcs:
        raise RuntimeError(f"no default VPC in region {region}")
    vpc_id = vpcs[0]["VpcId"]
    vpc_cidr = vpcs[0]["CidrBlock"]
    s = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "map-public-ip-on-launch", "Values": ["true"]},
        ]
    )
    subnets = s.get("Subnets", [])
    if not subnets:
        raise RuntimeError(f"no public subnet in VPC {vpc_id} ({region})")
    verbose_proxy_logger.debug(
        f"Discovered default VPC {vpc_id} cidr={vpc_cidr} in {region}"
    )
    return vpc_id, vpc_cidr, subnets[0]["SubnetId"]


def _sg_has_tcp_ingress(sg: dict, port: int) -> bool:
    for perm in sg.get("IpPermissions", []) or []:
        if perm.get("IpProtocol") != "tcp":
            continue
        if (
            perm.get("FromPort") is not None
            and perm.get("ToPort") is not None
            and perm["FromPort"] <= port <= perm["ToPort"]
        ):
            return True
    return False


def _ensure_tcp_ingress(ec2, sg_id: str, sg_name: str, port: int) -> None:
    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": port,
                    "ToPort": port,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }
            ],
        )
        verbose_proxy_logger.info(f"Added ingress for tcp/{port} to {sg_name}")
    except ClientError as e:
        if "InvalidPermission.Duplicate" not in str(e):
            raise


def ensure_security_group(
    region: str,
    sg_name: str,
    vpc_id: str,
    vpc_cidr: str,
    container_port: int,
) -> str:
    ec2 = _ec2(region)
    shim_port = container_port + 1
    r = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": [sg_name]},
        ]
    )
    existing = r.get("SecurityGroups", [])
    if existing:
        sg = existing[0]
        sg_id = sg["GroupId"]
        verbose_proxy_logger.debug(
            f"Security group {sg_name} already exists in VPC {vpc_id}"
        )
        if not _sg_has_tcp_ingress(sg, container_port):
            _ensure_tcp_ingress(ec2, sg_id, sg_name, container_port)
        if not _sg_has_tcp_ingress(sg, shim_port):
            _ensure_tcp_ingress(ec2, sg_id, sg_name, shim_port)
        return sg_id

    verbose_proxy_logger.info(f"Creating security group {sg_name} in VPC {vpc_id}")
    sg_id = ec2.create_security_group(
        GroupName=sg_name,
        Description="litellm managed agents sandbox",
        VpcId=vpc_id,
    )["GroupId"]

    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": container_port,
                "ToPort": container_port,
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": shim_port,
                "ToPort": shim_port,
                "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
            },
        ],
    )

    # Revoke default allow-all egress so we can install a restricted set.
    try:
        ec2.revoke_security_group_egress(
            GroupId=sg_id,
            IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
        )
    except ClientError as e:
        verbose_proxy_logger.debug(f"revoke default egress on {sg_id} no-op: {e}")

    ec2.authorize_security_group_egress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 443,
                "ToPort": 443,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTPS"}],
            },
            {
                "IpProtocol": "udp",
                "FromPort": 53,
                "ToPort": 53,
                "IpRanges": [
                    {"CidrIp": vpc_cidr, "Description": "DNS to VPC resolver"}
                ],
            },
            {
                "IpProtocol": "tcp",
                "FromPort": 53,
                "ToPort": 53,
                "IpRanges": [{"CidrIp": vpc_cidr, "Description": "DNS TCP fallback"}],
            },
        ],
    )
    return sg_id


def ensure_log_group(region: str, log_group_name: str) -> None:
    logs = _logs(region)
    try:
        logs.create_log_group(logGroupName=log_group_name)
        verbose_proxy_logger.info(
            f"Created CloudWatch log group {log_group_name} in {region}"
        )
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
            verbose_proxy_logger.debug(
                f"Log group {log_group_name} already exists in {region}"
            )
            return
        raise


def _validate_existing_cluster(region: str, cluster_name_or_arn: str) -> str:
    ecs = _ecs(region)
    r = ecs.describe_clusters(clusters=[cluster_name_or_arn])
    clusters = r.get("clusters", [])
    if not clusters or clusters[0].get("status") != "ACTIVE":
        raise RuntimeError(
            f"override cluster {cluster_name_or_arn} not ACTIVE in {region}"
        )
    return clusters[0]["clusterArn"]


def _validate_existing_role(role_arn_or_name: str) -> str:
    iam = _iam()
    if role_arn_or_name.startswith("arn:"):
        role_name = role_arn_or_name.rsplit("/", 1)[-1]
    else:
        role_name = role_arn_or_name
    return iam.get_role(RoleName=role_name)["Role"]["Arn"]


def _validate_existing_security_group(region: str, sg_id: str, vpc_id: str) -> str:
    ec2 = _ec2(region)
    r = ec2.describe_security_groups(GroupIds=[sg_id])
    groups = r.get("SecurityGroups", [])
    if not groups:
        raise RuntimeError(f"override security_group {sg_id} not found in {region}")
    if groups[0].get("VpcId") != vpc_id:
        raise RuntimeError(
            f"override security_group {sg_id} is in VPC {groups[0].get('VpcId')}, "
            f"expected {vpc_id}"
        )
    return groups[0]["GroupId"]


def _validate_existing_subnets(
    region: str, subnet_ids: List[str]
) -> Tuple[str, List[str]]:
    ec2 = _ec2(region)
    r = ec2.describe_subnets(SubnetIds=subnet_ids)
    subnets = r.get("Subnets", [])
    if len(subnets) != len(subnet_ids):
        found = {s["SubnetId"] for s in subnets}
        missing = [s for s in subnet_ids if s not in found]
        raise RuntimeError(f"override subnets not found: {missing}")
    vpc_ids = {s["VpcId"] for s in subnets}
    if len(vpc_ids) != 1:
        raise RuntimeError(f"override subnets span multiple VPCs: {vpc_ids}")
    return vpc_ids.pop(), [s["SubnetId"] for s in subnets]


def _validate_existing_log_group(region: str, log_group_name: str) -> None:
    logs = _logs(region)
    r = logs.describe_log_groups(logGroupNamePrefix=log_group_name)
    for g in r.get("logGroups", []):
        if g.get("logGroupName") == log_group_name:
            return
    raise RuntimeError(f"override log_group {log_group_name} not found in {region}")


def bootstrap_shared_infra(
    region: str,
    overrides: AwsOverrides,
    container_port: int = 4096,
) -> SharedInfra:
    if overrides.cluster:
        cluster_arn = _validate_existing_cluster(region, overrides.cluster)
    else:
        cluster_arn = ensure_cluster(region, DEFAULT_CLUSTER_NAME)

    if overrides.task_exec_role_arn:
        task_exec_role_arn = _validate_existing_role(overrides.task_exec_role_arn)
    else:
        task_exec_role_arn = ensure_task_exec_role(DEFAULT_TASK_EXEC_ROLE_NAME)

    if overrides.subnets:
        vpc_id, subnet_ids = _validate_existing_subnets(region, overrides.subnets)
        vpc_cidr_resp = _ec2(region).describe_vpcs(VpcIds=[vpc_id])
        vpc_cidr = vpc_cidr_resp["Vpcs"][0]["CidrBlock"]
    else:
        vpc_id, vpc_cidr, subnet = discover_vpc_subnet(region)
        subnet_ids = [subnet]

    if overrides.security_group:
        security_group_id = _validate_existing_security_group(
            region, overrides.security_group, vpc_id
        )
    else:
        security_group_id = ensure_security_group(
            region, DEFAULT_SECURITY_GROUP_NAME, vpc_id, vpc_cidr, container_port
        )

    if overrides.log_group:
        _validate_existing_log_group(region, overrides.log_group)
        log_group_name = overrides.log_group
    else:
        ensure_log_group(region, DEFAULT_LOG_GROUP_NAME)
        log_group_name = DEFAULT_LOG_GROUP_NAME

    return SharedInfra(
        cluster_arn=cluster_arn,
        task_exec_role_arn=task_exec_role_arn,
        security_group_id=security_group_id,
        log_group_name=log_group_name,
        vpc_id=vpc_id,
        subnet_ids=subnet_ids,
    )
