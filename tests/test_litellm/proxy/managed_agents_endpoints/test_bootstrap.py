"""Unit tests for fargate bootstrap module.

Mocks boto3 clients used by `bootstrap.py` (`_ecs`, `_ec2`, `_iam`, `_logs`).
No real AWS calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from litellm.proxy.managed_agents_endpoints.fargate import bootstrap
from litellm.proxy.managed_agents_endpoints.types import AwsOverrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": code}},
        operation_name="op",
    )


def _no_such_entity_exception_class():
    return type("NoSuchEntityException", (Exception,), {})


def _resource_already_exists_class():
    return type("ResourceAlreadyExistsException", (Exception,), {})


@pytest.fixture
def mock_ecs():
    return MagicMock()


@pytest.fixture
def mock_ec2():
    return MagicMock()


@pytest.fixture
def mock_iam():
    return MagicMock()


@pytest.fixture
def mock_logs():
    m = MagicMock()
    m.exceptions.ResourceAlreadyExistsException = _resource_already_exists_class()
    return m


# ---------------------------------------------------------------------------
# ensure_cluster
# ---------------------------------------------------------------------------


def test_ensure_cluster_existing_active_returns_arn(mock_ecs):
    mock_ecs.describe_clusters.return_value = {
        "clusters": [
            {"status": "ACTIVE", "clusterArn": "arn:aws:ecs:us-west-2:1:cluster/foo"}
        ]
    }
    with patch.object(bootstrap, "_ecs", return_value=mock_ecs):
        arn = bootstrap.ensure_cluster("us-west-2", "foo")

    assert arn == "arn:aws:ecs:us-west-2:1:cluster/foo"
    mock_ecs.describe_clusters.assert_called_once_with(clusters=["foo"])
    mock_ecs.create_cluster.assert_not_called()


def test_ensure_cluster_missing_creates(mock_ecs):
    mock_ecs.describe_clusters.return_value = {"clusters": []}
    mock_ecs.create_cluster.return_value = {
        "cluster": {"clusterArn": "arn:aws:ecs:us-west-2:1:cluster/new"}
    }

    with patch.object(bootstrap, "_ecs", return_value=mock_ecs):
        arn = bootstrap.ensure_cluster("us-west-2", "new")

    assert arn == "arn:aws:ecs:us-west-2:1:cluster/new"
    mock_ecs.create_cluster.assert_called_once_with(clusterName="new")


# ---------------------------------------------------------------------------
# ensure_task_exec_role
# ---------------------------------------------------------------------------


def test_ensure_task_exec_role_existing_returns_arn(mock_iam):
    mock_iam.get_role.return_value = {
        "Role": {"Arn": "arn:aws:iam::1:role/litellm-agents-task-exec"}
    }
    with patch.object(bootstrap, "_iam", return_value=mock_iam):
        arn = bootstrap.ensure_task_exec_role("litellm-agents-task-exec")

    assert arn == "arn:aws:iam::1:role/litellm-agents-task-exec"
    mock_iam.create_role.assert_not_called()
    mock_iam.attach_role_policy.assert_not_called()


def test_ensure_task_exec_role_missing_creates_attaches_and_sleeps(mock_iam):
    mock_iam.get_role.side_effect = _client_error("NoSuchEntity")
    mock_iam.create_role.return_value = {
        "Role": {"Arn": "arn:aws:iam::1:role/new-role"}
    }

    with (
        patch.object(bootstrap, "_iam", return_value=mock_iam),
        patch.object(bootstrap.time, "sleep") as sleep_mock,
    ):
        arn = bootstrap.ensure_task_exec_role("new-role")

    assert arn == "arn:aws:iam::1:role/new-role"
    mock_iam.create_role.assert_called_once()
    mock_iam.attach_role_policy.assert_called_once_with(
        RoleName="new-role",
        PolicyArn=bootstrap.TASK_EXEC_ROLE_POLICY_ARN,
    )
    sleep_mock.assert_called_once_with(10)


# ---------------------------------------------------------------------------
# discover_vpc_subnet
# ---------------------------------------------------------------------------


def test_discover_vpc_subnet_happy_path(mock_ec2):
    mock_ec2.describe_vpcs.return_value = {
        "Vpcs": [{"VpcId": "vpc-1", "CidrBlock": "10.0.0.0/16"}]
    }
    mock_ec2.describe_subnets.return_value = {"Subnets": [{"SubnetId": "subnet-1"}]}

    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        vpc_id, vpc_cidr, subnet_id = bootstrap.discover_vpc_subnet("us-west-2")

    assert vpc_id == "vpc-1"
    assert vpc_cidr == "10.0.0.0/16"
    assert subnet_id == "subnet-1"


def test_discover_vpc_subnet_no_default_vpc_raises(mock_ec2):
    mock_ec2.describe_vpcs.return_value = {"Vpcs": []}

    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        with pytest.raises(RuntimeError, match="no default VPC"):
            bootstrap.discover_vpc_subnet("us-west-2")


def test_discover_vpc_subnet_no_public_subnet_raises(mock_ec2):
    mock_ec2.describe_vpcs.return_value = {
        "Vpcs": [{"VpcId": "vpc-1", "CidrBlock": "10.0.0.0/16"}]
    }
    mock_ec2.describe_subnets.return_value = {"Subnets": []}

    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        with pytest.raises(RuntimeError, match="no public subnet"):
            bootstrap.discover_vpc_subnet("us-west-2")


# ---------------------------------------------------------------------------
# ensure_security_group
# ---------------------------------------------------------------------------


def test_ensure_security_group_existing_with_port_returns_id(mock_ec2):
    # Existing SG must already have BOTH the harness port AND the warm-pool
    # shim port (container_port + 1) for the no-op path.
    mock_ec2.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-existing",
                "IpPermissions": [
                    {"IpProtocol": "tcp", "FromPort": 4096, "ToPort": 4096},
                    {"IpProtocol": "tcp", "FromPort": 4097, "ToPort": 4097},
                ],
            }
        ]
    }
    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        sg_id = bootstrap.ensure_security_group(
            "us-west-2", "litellm-sg", "vpc-1", "10.0.0.0/16", 4096
        )

    assert sg_id == "sg-existing"
    mock_ec2.create_security_group.assert_not_called()
    mock_ec2.authorize_security_group_ingress.assert_not_called()
    mock_ec2.revoke_security_group_egress.assert_not_called()
    mock_ec2.authorize_security_group_egress.assert_not_called()


def test_ensure_security_group_existing_missing_port_authorizes(mock_ec2):
    """Existing SG was created for a different container_port; we must add
    ingress rules for both the harness port and the shim port (port + 1)."""
    mock_ec2.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-existing",
                "IpPermissions": [
                    {"IpProtocol": "tcp", "FromPort": 4096, "ToPort": 4096}
                ],
            }
        ]
    }
    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        sg_id = bootstrap.ensure_security_group(
            "us-west-2", "litellm-sg", "vpc-1", "10.0.0.0/16", 5000
        )

    assert sg_id == "sg-existing"
    # Two calls: one for the harness port (5000), one for the shim port (5001).
    assert mock_ec2.authorize_security_group_ingress.call_count == 2
    authorized_ports = sorted(
        call.kwargs["IpPermissions"][0]["FromPort"]
        for call in mock_ec2.authorize_security_group_ingress.call_args_list
    )
    assert authorized_ports == [5000, 5001]


def test_ensure_security_group_existing_missing_port_swallows_duplicate(mock_ec2):
    mock_ec2.describe_security_groups.return_value = {
        "SecurityGroups": [
            {
                "GroupId": "sg-existing",
                "IpPermissions": [
                    {"IpProtocol": "tcp", "FromPort": 4096, "ToPort": 4096}
                ],
            }
        ]
    }
    mock_ec2.authorize_security_group_ingress.side_effect = _client_error(
        "InvalidPermission.Duplicate"
    )
    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        sg_id = bootstrap.ensure_security_group(
            "us-west-2", "litellm-sg", "vpc-1", "10.0.0.0/16", 5000
        )
    assert sg_id == "sg-existing"


def test_ensure_security_group_missing_creates_ingress_revoke_and_authorize(mock_ec2):
    mock_ec2.describe_security_groups.return_value = {"SecurityGroups": []}
    mock_ec2.create_security_group.return_value = {"GroupId": "sg-new"}

    with patch.object(bootstrap, "_ec2", return_value=mock_ec2):
        sg_id = bootstrap.ensure_security_group(
            "us-west-2", "litellm-sg", "vpc-1", "10.0.0.0/16", 4096
        )

    assert sg_id == "sg-new"
    mock_ec2.create_security_group.assert_called_once()
    mock_ec2.authorize_security_group_ingress.assert_called_once()
    ingress_kwargs = mock_ec2.authorize_security_group_ingress.call_args.kwargs
    assert ingress_kwargs["GroupId"] == "sg-new"
    assert ingress_kwargs["IpPermissions"][0]["FromPort"] == 4096
    assert ingress_kwargs["IpPermissions"][0]["ToPort"] == 4096

    mock_ec2.revoke_security_group_egress.assert_called_once()

    mock_ec2.authorize_security_group_egress.assert_called_once()
    egress_kwargs = mock_ec2.authorize_security_group_egress.call_args.kwargs
    perms = egress_kwargs["IpPermissions"]
    ports = sorted({(p["IpProtocol"], p["FromPort"]) for p in perms})
    assert ("tcp", 443) in ports
    assert ("tcp", 53) in ports
    assert ("udp", 53) in ports


# ---------------------------------------------------------------------------
# ensure_log_group
# ---------------------------------------------------------------------------


def test_ensure_log_group_missing_creates(mock_logs):
    with patch.object(bootstrap, "_logs", return_value=mock_logs):
        bootstrap.ensure_log_group("us-west-2", "/ecs/foo")

    mock_logs.create_log_group.assert_called_once_with(logGroupName="/ecs/foo")


def test_ensure_log_group_already_exists_swallowed(mock_logs):
    mock_logs.create_log_group.side_effect = _client_error(
        "ResourceAlreadyExistsException"
    )

    with patch.object(bootstrap, "_logs", return_value=mock_logs):
        # Should NOT raise
        bootstrap.ensure_log_group("us-west-2", "/ecs/foo")

    mock_logs.create_log_group.assert_called_once()


def test_ensure_log_group_other_error_reraised(mock_logs):
    mock_logs.create_log_group.side_effect = _client_error("AccessDenied")

    with patch.object(bootstrap, "_logs", return_value=mock_logs):
        with pytest.raises(ClientError):
            bootstrap.ensure_log_group("us-west-2", "/ecs/foo")


# ---------------------------------------------------------------------------
# bootstrap_shared_infra
# ---------------------------------------------------------------------------


def test_bootstrap_shared_infra_no_overrides_calls_every_ensure():
    overrides = AwsOverrides()

    with (
        patch.object(
            bootstrap, "ensure_cluster", return_value="cluster-arn"
        ) as ensure_cluster,
        patch.object(
            bootstrap, "ensure_task_exec_role", return_value="role-arn"
        ) as ensure_role,
        patch.object(
            bootstrap,
            "discover_vpc_subnet",
            return_value=("vpc-1", "10.0.0.0/16", "subnet-1"),
        ) as discover,
        patch.object(
            bootstrap, "ensure_security_group", return_value="sg-1"
        ) as ensure_sg,
        patch.object(bootstrap, "ensure_log_group") as ensure_lg,
    ):
        infra = bootstrap.bootstrap_shared_infra("us-west-2", overrides)

    ensure_cluster.assert_called_once()
    ensure_role.assert_called_once()
    discover.assert_called_once()
    ensure_sg.assert_called_once()
    ensure_lg.assert_called_once()

    assert infra.cluster_arn == "cluster-arn"
    assert infra.task_exec_role_arn == "role-arn"
    assert infra.security_group_id == "sg-1"
    assert infra.log_group_name == bootstrap.DEFAULT_LOG_GROUP_NAME
    assert infra.vpc_id == "vpc-1"
    assert infra.subnet_ids == ["subnet-1"]


def test_bootstrap_shared_infra_with_overrides_uses_given_skips_ensures():
    overrides = AwsOverrides(
        cluster="my-cluster",
        security_group="sg-given",
        task_exec_role_arn="arn:aws:iam::1:role/given-role",
    )

    with (
        patch.object(
            bootstrap, "_validate_existing_cluster", return_value="cluster-arn-given"
        ) as v_cluster,
        patch.object(
            bootstrap, "_validate_existing_role", return_value="role-arn-given"
        ) as v_role,
        patch.object(
            bootstrap, "_validate_existing_security_group", return_value="sg-given"
        ) as v_sg,
        patch.object(
            bootstrap,
            "discover_vpc_subnet",
            return_value=("vpc-1", "10.0.0.0/16", "subnet-1"),
        ) as discover,
        patch.object(bootstrap, "ensure_log_group") as ensure_lg,
        patch.object(bootstrap, "ensure_cluster") as ensure_cluster,
        patch.object(bootstrap, "ensure_task_exec_role") as ensure_role,
        patch.object(bootstrap, "ensure_security_group") as ensure_sg,
    ):
        infra = bootstrap.bootstrap_shared_infra("us-west-2", overrides)

    # Validators called for overrides
    v_cluster.assert_called_once()
    v_role.assert_called_once()
    v_sg.assert_called_once()

    # ensure_* skipped for overridden resources
    ensure_cluster.assert_not_called()
    ensure_role.assert_not_called()
    ensure_sg.assert_not_called()

    # Subnet not overridden → discover_vpc_subnet called
    discover.assert_called_once()
    # log_group not overridden → ensure_log_group called
    ensure_lg.assert_called_once()

    assert infra.cluster_arn == "cluster-arn-given"
    assert infra.task_exec_role_arn == "role-arn-given"
    assert infra.security_group_id == "sg-given"
