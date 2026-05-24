import pytest
from conftest import FakeRunner

from litellm_provisioning_mcp.commands import CommandError, CommandResult
from litellm_provisioning_mcp.helm import HelmRunner


async def test_upgrade_install_builds_command_and_passes_values_via_stdin():
    runner = FakeRunner()
    helm = HelmRunner(namespace="litellm", runner=runner, wait_timeout=300)

    await helm.upgrade_install(
        release="rel", chart_path="/chart", values_yaml="key: value\n"
    )

    args, stdin = runner.calls[0]
    assert args[:4] == ["helm", "upgrade", "rel", "/chart"]
    assert "--install" in args
    assert args[-2:] == ["--namespace", "litellm"]
    assert "--values" in args and "-" in args
    assert "--timeout" in args and "300s" in args
    assert stdin == "key: value\n"


async def test_status_parses_json():
    runner = FakeRunner()
    helm = HelmRunner(namespace="litellm", runner=runner)
    status = await helm.status(release="rel")
    assert status["info"]["status"] == "deployed"


async def test_nonzero_exit_raises_command_error():
    async def runner(args, *, input_text=None, timeout=None):
        return CommandResult(1, "", "release not found")

    helm = HelmRunner(namespace="litellm", runner=runner)
    with pytest.raises(CommandError, match="release not found"):
        await helm.status(release="missing")
