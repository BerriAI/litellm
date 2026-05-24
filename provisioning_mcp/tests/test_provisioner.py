import pytest
import yaml
from conftest import FakeRunner, make_settings

from litellm_provisioning_mcp.commands import CommandError, CommandResult
from litellm_provisioning_mcp.helm import HelmRunner
from litellm_provisioning_mcp.kubectl import KubectlRunner
from litellm_provisioning_mcp.provisioner import (
    ProvisionError,
    ProvisionRequest,
    Provisioner,
)


def _provisioner(runner: FakeRunner) -> Provisioner:
    settings = make_settings()
    return Provisioner(
        settings,
        helm=HelmRunner(namespace="litellm", runner=runner, wait_timeout=600),
        kubectl=KubectlRunner(namespace="litellm", runner=runner),
    )


def _helm_values(runner: FakeRunner) -> dict:
    _, values_yaml = runner.find("upgrade")
    return yaml.safe_load(values_yaml)


async def test_provision_with_ephemeral_postgres_and_redis():
    runner = FakeRunner()
    provisioner = _provisioner(runner)

    result = await provisioner.provision(
        ProvisionRequest(
            repo_url="https://github.com/BerriAI/litellm",
            revision="abc123",
            enable_postgres=True,
            enable_redis=True,
        )
    )

    assert result["success"] is True
    assert result["release"] == "litellm-e2e-abc123"
    assert result["images"]["gateway"] == "ghcr.io/berriai/litellm-gateway"

    values = _helm_values(runner)
    assert values["fullnameOverride"] == "litellm-e2e-abc123"
    assert values["gateway"]["image"]["tag"] == "abc123"
    assert values["gateway"]["image"]["repository"] == "ghcr.io/berriai/litellm-gateway"
    assert values["database"]["writer"]["host"] == "litellm-e2e-abc123-postgres"
    assert values["redis"]["host"] == "litellm-e2e-abc123-redis"
    assert values["masterKey"]["secretName"] == "litellm-e2e-abc123-master-key"
    assert values["ui"]["enabled"] is False

    # ephemeral datastores were applied and waited on
    runner.find("apply")
    runner.find("wait", "deployment/litellm-e2e-abc123-postgres")
    runner.find("wait", "deployment/litellm-e2e-abc123-redis")


async def test_provision_without_db_raises():
    runner = FakeRunner()
    provisioner = _provisioner(runner)
    with pytest.raises(ProvisionError, match="database is required"):
        await provisioner.provision(
            ProvisionRequest(
                repo_url="https://github.com/BerriAI/litellm",
                revision="abc",
                enable_postgres=False,
            )
        )


async def test_provision_with_external_database():
    runner = FakeRunner()
    provisioner = _provisioner(runner)

    result = await provisioner.provision(
        ProvisionRequest(
            repo_url="https://github.com/BerriAI/litellm",
            revision="v1",
            enable_postgres=False,
            external_database={
                "host": "pg.prod.internal",
                "dbname": "litellm",
                "secret_name": "prod-db",
            },
        )
    )

    assert result["success"] is True
    values = _helm_values(runner)
    assert values["database"]["writer"]["host"] == "pg.prod.internal"
    assert values["database"]["writer"]["passwordSecret"]["name"] == "prod-db"
    # no ephemeral postgres applied
    with pytest.raises(AssertionError):
        runner.find("wait", "deployment/litellm-e2e-v1-postgres")


async def test_external_database_missing_keys_raises():
    runner = FakeRunner()
    provisioner = _provisioner(runner)
    with pytest.raises(ProvisionError, match="missing required keys"):
        await provisioner.provision(
            ProvisionRequest(
                repo_url="https://github.com/BerriAI/litellm",
                revision="v1",
                enable_postgres=False,
                external_database={"host": "pg"},
            )
        )


async def test_image_registry_override_and_extra_values():
    runner = FakeRunner()
    provisioner = _provisioner(runner)

    await provisioner.provision(
        ProvisionRequest(
            repo_url="https://github.com/BerriAI/litellm",
            revision="sha",
            image_registry="myreg.io/team",
            service_account="litellm-workload",
            extra_values={"gateway": {"numWorkers": 4}},
        )
    )

    values = _helm_values(runner)
    assert values["gateway"]["image"]["repository"] == "myreg.io/team/litellm-gateway"
    assert values["serviceAccount"] == {"create": False, "name": "litellm-workload"}
    # extra_values deep-merge preserves the derived image block
    assert values["gateway"]["numWorkers"] == 4
    assert values["gateway"]["image"]["tag"] == "sha"


async def test_provision_forces_release_prefix():
    runner = FakeRunner()
    provisioner = _provisioner(runner)

    result = await provisioner.provision(
        ProvisionRequest(
            repo_url="https://github.com/BerriAI/litellm",
            revision="abc",
            release_name="litellm",  # would collide with a real release
        )
    )

    assert result["release"] == "litellm-e2e-litellm"
    values = _helm_values(runner)
    assert values["fullnameOverride"] == "litellm-e2e-litellm"


async def test_provision_rejects_self_service_account():
    runner = FakeRunner()
    provisioner = _provisioner(runner)
    with pytest.raises(ProvisionError, match="own"):
        await provisioner.provision(
            ProvisionRequest(
                repo_url="https://github.com/BerriAI/litellm",
                revision="abc",
                service_account="litellm-provisioning-mcp",
            )
        )


async def test_provision_rejects_disallowed_service_account():
    runner = FakeRunner()
    settings = make_settings(allowed_service_accounts=("litellm-workload",))
    provisioner = Provisioner(
        settings,
        helm=HelmRunner(namespace="litellm", runner=runner, wait_timeout=600),
        kubectl=KubectlRunner(namespace="litellm", runner=runner),
    )
    with pytest.raises(ProvisionError, match="not in the allowed list"):
        await provisioner.provision(
            ProvisionRequest(
                repo_url="https://github.com/BerriAI/litellm",
                revision="abc",
                service_account="some-other-sa",
            )
        )


async def test_provision_refuses_existing_unmanaged_release():
    runner = FakeRunner()
    runner.helm_releases = ["litellm-e2e-abc"]  # already exists
    runner.owns_release = False  # but not created by us
    provisioner = _provisioner(runner)
    with pytest.raises(ProvisionError, match="refusing to overwrite"):
        await provisioner.provision(
            ProvisionRequest(
                repo_url="https://github.com/BerriAI/litellm",
                revision="abc",
            )
        )
    with pytest.raises(AssertionError):
        runner.find("upgrade")


async def test_provision_preserves_existing_master_key():
    runner = FakeRunner()
    runner.master_key_exists = True
    provisioner = _provisioner(runner)

    await provisioner.provision(
        ProvisionRequest(
            repo_url="https://github.com/BerriAI/litellm",
            revision="abc123",
        )
    )

    # The master-key Secret already exists, so it must not be re-applied
    # (re-applying would rotate the key and invalidate live API keys).
    runner.find("get", "secret")  # existence was probed
    applied = [text for args, text in runner.calls if "apply" in args and text]
    assert not any("master-key" in payload for payload in applied)


async def test_delete_refuses_release_not_owned_by_tool():
    runner = FakeRunner()
    runner.owns_release = False
    provisioner = _provisioner(runner)

    with pytest.raises(ProvisionError, match="refusing to delete"):
        await provisioner.delete("some-production-release")

    # Must not have attempted a helm uninstall on an unowned release.
    with pytest.raises(AssertionError):
        runner.find("uninstall")


async def test_delete_uninstalls_and_cleans_datastores():
    runner = FakeRunner()
    provisioner = _provisioner(runner)

    result = await provisioner.delete("litellm-e2e-abc")

    assert result["success"] is True
    runner.find("uninstall", "litellm-e2e-abc")
    args, _ = runner.find("delete", "--selector")
    assert "litellm.ai/release=litellm-e2e-abc" in args


async def test_status_reports_helm_and_pods():
    runner = FakeRunner()
    provisioner = _provisioner(runner)
    result = await provisioner.status("litellm-e2e-abc")
    assert result["status"] == "deployed"
    assert result["pods"] == []


async def test_provision_propagates_helm_failure():
    runner = FakeRunner()

    async def failing(args, *, input_text=None, timeout=None):
        result = await runner(args, input_text=input_text, timeout=timeout)
        if args[0].endswith("helm") and "upgrade" in args:
            return CommandResult(1, "", "boom: chart not found")
        return result

    settings = make_settings()
    provisioner = Provisioner(
        settings,
        helm=HelmRunner(namespace="litellm", runner=failing, wait_timeout=600),
        kubectl=KubectlRunner(namespace="litellm", runner=failing),
    )
    with pytest.raises(CommandError, match="boom"):
        await provisioner.provision(
            ProvisionRequest(
                repo_url="https://github.com/BerriAI/litellm", revision="x"
            )
        )
