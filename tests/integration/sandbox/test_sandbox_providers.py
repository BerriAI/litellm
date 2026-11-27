"""
Boat, Tensorlake, Blaxel and Modal sandboxes - end-to-end integration tests.

These tests make REAL calls to each provider and are skipped unless that provider's credentials are set.

Run only these tests:
    pytest tests/integration/sandbox/test_sandbox_providers.py -v
"""

import os
import subprocess
from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.sandbox.transformation import shell_command

PROVIDER_ENV: Final = {
    "boat": ("BOAT_API_KEY",),
    "tensorlake": ("TENSORLAKE_API_KEY",),
    "blaxel": ("BL_API_KEY",),
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
}
PROVIDERS: Final = [
    pytest.param(
        provider,
        marks=pytest.mark.skipif(
            any(name not in os.environ for name in names), reason=f"needs real {' and '.join(names)}"
        ),
    )
    for provider, names in PROVIDER_ENV.items()
]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_integration_lifecycle_and_cost(provider):
    container = await litellm.acreate_sandbox(provider=provider, timeout=120)
    try:
        ok = await litellm.arun_code(provider=provider, container=container, code="print(sum(range(10)))")
        failed = await litellm.arun_code(
            provider=provider, container=container, code="echo out; echo err >&2; exit 3", language="bash"
        )
    finally:
        assert await litellm.adelete_sandbox(provider=provider, container=container)
    cost = await litellm.asandbox_cost(provider=provider, container=container)

    assert ok.stdout.strip() == "45"
    assert ok.error is None
    assert (failed.stdout.strip(), failed.stderr.strip(), failed.exit_code) == ("out", "err", 3)
    assert cost.source == ("provider" if provider == "boat" else "estimate")
    assert cost.cost_usd is not None and cost.cost_usd >= 0


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_integration_ephemeral_reports_cost(provider):
    result = await litellm.acode_interpreter_tool(provider=provider, code="print(6*7)", timeout=120)

    assert result.stdout.strip() == "42"
    assert result._hidden_params["response_cost"] >= 0


def _run_locally(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30, check=False)


def test_integration_the_sandbox_command_runs_the_code_with_its_streams_and_exit_code():
    code = "import sys; print('a b'); print(\"c\"); print('bad', file=sys.stderr); sys.exit(3)"
    ran = _run_locally(shell_command(code, "python"))
    assert (ran.stdout, ran.stderr, ran.returncode) == ("a b\nc\n", "bad\n", 3)


@pytest.mark.parametrize("stream", ["", " >&2"])
def test_integration_a_command_that_never_stops_printing_is_ended_one_byte_past_the_cap(stream):
    ran = _run_locally(shell_command(f"yes{stream}", "bash", max_output_bytes=1000))
    assert len(ran.stdout + ran.stderr) == 1001
    assert ran.returncode != 0


def test_integration_a_command_that_reads_stdin_gets_end_of_file():
    ran = _run_locally(shell_command("cat; echo done", "bash"))
    assert (ran.stdout, ran.returncode) == ("done\n", 0)
