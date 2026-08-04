"""Unit tests for `docker/component_entrypoint.sh` and its wiring into the
componentized `gateway` / `backend` images and Terraform deployments."""

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_ENTRYPOINT = REPO_ROOT / "docker" / "component_entrypoint.sh"
PROD_ENTRYPOINT = REPO_ROOT / "docker" / "prod_entrypoint.sh"
GATEWAY_DOCKERFILE = REPO_ROOT / "gateway" / "Dockerfile"
BACKEND_DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
TERRAFORM_ECS = REPO_ROOT / "terraform" / "litellm" / "aws" / "ecs.tf"
TERRAFORM_CLOUDRUN = REPO_ROOT / "terraform" / "litellm" / "gcp" / "cloudrun.tf"

IMAGE_ENTRYPOINT_PATH = "/app/docker/component_entrypoint.sh"

PYTHONPATH_SENTINEL = "/lit-entrypoint-sentinel:/app"

_STUB_TEMPLATE = """#!/bin/sh
{{
  echo "exec={name}"
  echo "args=$*"
  echo "DD_TRACE_OPENAI_ENABLED=${{DD_TRACE_OPENAI_ENABLED-<unset>}}"
  echo "PYTHONPATH=${{PYTHONPATH-<unset>}}"
}} >> "$RECORD"
"""

_ENTRYPOINT_RE = re.compile(r"^ENTRYPOINT\s+(\[.*\])\s*$", re.MULTILINE)
_CMD_RE = re.compile(r"^CMD\s+(\[.*\])\s*$", re.MULTILINE)
_APP_TARGET_RE = re.compile(r"(?:gateway|backend)\.main:app")
_TF_STRING_LOCAL_RE = re.compile(r'^\s*(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"\s*$', re.MULTILINE)
_TF_INTERPOLATION_RE = re.compile(r"\$\{(local|var)\.(\w+)\}")

TERRAFORM_LAUNCH_SITES = {TERRAFORM_ECS: 2, TERRAFORM_CLOUDRUN: 2}
TERRAFORM_VAR_STUBS = {"gateway_num_workers": "2"}
_MAX_INTERPOLATION_PASSES = 5


def _write_stubs(bin_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        stub = bin_dir / name
        stub.write_text(_STUB_TEMPLATE.format(name=name))
        stub.chmod(0o755)


def _run_entrypoint(
    script: Path,
    argv: tuple[str, ...],
    use_ddtrace: Optional[str],
    tmp_path: Path,
) -> tuple[str, ...]:
    """Run `script` with stubbed executables on PATH and return the recorded lines."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stubs(bin_dir, ("ddtrace-run", "uvicorn", "litellm"))
    record = tmp_path / "record.txt"

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RECORD": str(record),
        "PYTHONPATH": PYTHONPATH_SENTINEL,
    }
    env.pop("USE_DDTRACE", None)
    env.pop("DD_TRACE_OPENAI_ENABLED", None)
    if use_ddtrace is not None:
        env["USE_DDTRACE"] = use_ddtrace

    result = subprocess.run(
        ["sh", str(script), *argv],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    return tuple(record.read_text().splitlines()) if record.exists() else ()


def _run_shell_command(command: str, bin_dir: Path, record: Path, use_ddtrace: Optional[str]) -> tuple[str, ...]:
    """Run a resolved Terraform launch command through `sh -c` and return the recorded lines."""
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RECORD": str(record),
        "PYTHONPATH": PYTHONPATH_SENTINEL,
    }
    env.pop("USE_DDTRACE", None)
    env.pop("DD_TRACE_OPENAI_ENABLED", None)
    if use_ddtrace is not None:
        env["USE_DDTRACE"] = use_ddtrace

    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    return tuple(record.read_text().splitlines()) if record.exists() else ()


def _resolve_tf_local(terraform_file: Path, name: str) -> str:
    """Read a string local out of a .tf file and expand its `local.` / `var.` interpolations."""
    values = {
        m.group(1): m.group(2).replace('\\"', '"') for m in _TF_STRING_LOCAL_RE.finditer(terraform_file.read_text())
    }

    assert name in values, f"{terraform_file} defines no {name} local"
    resolved = values[name]
    for _ in range(_MAX_INTERPOLATION_PASSES):
        if "${" not in resolved:
            return resolved
        resolved = _TF_INTERPOLATION_RE.sub(
            lambda m: values[m.group(2)] if m.group(1) == "local" else TERRAFORM_VAR_STUBS[m.group(2)],
            resolved,
        )
    raise AssertionError(f"{terraform_file}:{name} still has unresolved interpolations: {resolved}")


def _entrypoint_argv(dockerfile: Path) -> tuple[str, ...]:
    matches = _ENTRYPOINT_RE.findall(dockerfile.read_text())
    assert matches, f"no exec-form ENTRYPOINT found in {dockerfile}"
    parsed = json.loads(matches[-1])
    return tuple(str(part) for part in parsed)


def _cmd_argv(dockerfile: Path) -> tuple[str, ...]:
    matches = _CMD_RE.findall(dockerfile.read_text())
    assert matches, f"no exec-form CMD found in {dockerfile}"
    return tuple(str(part) for part in json.loads(matches[-1]))


def test_ddtrace_enabled_wraps_the_command_and_disables_the_openai_integration(
    tmp_path: Path,
) -> None:
    """`USE_DDTRACE=true` must prefix the command with `ddtrace-run`, turn the openai
    integration off, and leave PYTHONPATH alone.

    `ddtrace-run` installs its instrumentation by PREPENDING a bootstrap directory to
    PYTHONPATH, and the images set PYTHONPATH=/app so the app package is importable. A
    wrapper that assigned PYTHONPATH instead of inheriting it would either drop the
    bootstrap (silently disabling tracing) or drop /app (breaking the import), so the
    recorded value is asserted verbatim.

    PYTHONPATH_SENTINEL deliberately differs from the images' own /app: with /app as the
    fixture value, a wrapper that overwrote PYTHONPATH with /app would still satisfy this
    assertion and the check would prove nothing.

    The openai integration is disabled because under `ddtrace-run` the bootstrap patches
    it before any litellm code runs, so litellm's in-process `patch_all(..., openai=False)`
    can no longer suppress it; leaving it on double-reports every LLM call.
    """
    recorded = _run_entrypoint(
        COMPONENT_ENTRYPOINT,
        ("uvicorn", "gateway.main:app", "--workers", "2", "--port", "4000"),
        use_ddtrace="true",
        tmp_path=tmp_path,
    )

    assert recorded == (
        "exec=ddtrace-run",
        "args=uvicorn gateway.main:app --workers 2 --port 4000",
        "DD_TRACE_OPENAI_ENABLED=False",
        f"PYTHONPATH={PYTHONPATH_SENTINEL}",
    )


def test_ddtrace_disabled_execs_the_command_directly(tmp_path: Path) -> None:
    recorded = _run_entrypoint(
        COMPONENT_ENTRYPOINT,
        ("uvicorn", "backend.main:app", "--port", "4001"),
        use_ddtrace=None,
        tmp_path=tmp_path,
    )

    assert recorded == (
        "exec=uvicorn",
        "args=backend.main:app --port 4001",
        "DD_TRACE_OPENAI_ENABLED=<unset>",
        f"PYTHONPATH={PYTHONPATH_SENTINEL}",
    )


@pytest.mark.parametrize("use_ddtrace", [None, "", "false", "True", "TRUE", "1", "yes"])
def test_gating_matches_the_monolithic_entrypoint(use_ddtrace: Optional[str], tmp_path: Path) -> None:
    """The componentized images must honor `USE_DDTRACE` exactly as the monolith does."""
    component = _run_entrypoint(
        COMPONENT_ENTRYPOINT,
        ("uvicorn", "gateway.main:app"),
        use_ddtrace=use_ddtrace,
        tmp_path=tmp_path / "component",
    )
    monolith = _run_entrypoint(
        PROD_ENTRYPOINT,
        ("--port", "4000"),
        use_ddtrace=use_ddtrace,
        tmp_path=tmp_path / "monolith",
    )

    assert component[0].startswith("exec=") and monolith[0].startswith("exec=")
    assert (component[0] == "exec=ddtrace-run") == (monolith[0] == "exec=ddtrace-run")


def test_entrypoint_script_is_executable() -> None:
    mode = COMPONENT_ENTRYPOINT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be committed executable to run as the image ENTRYPOINT"
    assert mode & stat.S_IXOTH, "entrypoint must be executable by the unprivileged `nonroot` user"


def test_entrypoint_script_has_no_carriage_returns() -> None:
    assert b"\r" not in COMPONENT_ENTRYPOINT.read_bytes()


@pytest.mark.parametrize(
    "dockerfile, app_target",
    [
        (GATEWAY_DOCKERFILE, "gateway.main:app"),
        (BACKEND_DOCKERFILE, "backend.main:app"),
    ],
)
def test_component_images_launch_uvicorn_through_the_entrypoint(dockerfile: Path, app_target: str) -> None:
    entrypoint = " ".join(_entrypoint_argv(dockerfile))

    assert IMAGE_ENTRYPOINT_PATH in entrypoint, f"{dockerfile} bypasses the ddtrace-aware entrypoint"
    assert app_target in entrypoint
    assert entrypoint.index(IMAGE_ENTRYPOINT_PATH) < entrypoint.index("uvicorn"), (
        f"{dockerfile} must invoke uvicorn through the entrypoint, not the other way around"
    )


@pytest.mark.parametrize("dockerfile", [GATEWAY_DOCKERFILE, BACKEND_DOCKERFILE])
def test_component_images_make_the_entrypoint_executable(dockerfile: Path) -> None:
    body = dockerfile.read_text()
    assert "chmod +x docker/component_entrypoint.sh" in body


@pytest.mark.parametrize("terraform_file", TERRAFORM_LAUNCH_SITES, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("component", ["gateway", "backend"])
@pytest.mark.parametrize("use_ddtrace", [None, "true", "false", "True"])
def test_terraform_launch_command_matches_the_script_contract(
    terraform_file: Path, component: str, use_ddtrace: Optional[str], tmp_path: Path
) -> None:
    """The Terraform command and `docker/component_entrypoint.sh` must decide identically.

    The decision deliberately lives in two places. The script is what the image ENTRYPOINT runs;
    the Terraform strings are what runs when a deployment overrides that ENTRYPOINT, and they
    cannot call the script because the caller supplies the image tag and it may predate the file.
    Both modules default to a tag that does. So instead of asserting a shared path, this runs both
    implementations under the same environment and asserts they agree on which binary is exec'd
    and on whether the openai integration is disabled.
    """
    app_target = f"{component}.main:app"
    command = _resolve_tf_local(terraform_file, f"{component}_launch_cmd")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stubs(bin_dir, ("ddtrace-run", "uvicorn"))
    from_terraform = _run_shell_command(command, bin_dir, tmp_path / "terraform.txt", use_ddtrace)

    from_script = _run_entrypoint(
        COMPONENT_ENTRYPOINT,
        ("uvicorn", app_target),
        use_ddtrace=use_ddtrace,
        tmp_path=tmp_path / "script",
    )

    assert from_terraform[0] == from_script[0], (
        f"{terraform_file} disagrees with the script on USE_DDTRACE={use_ddtrace}"
    )
    assert from_terraform[2] == from_script[2], f"{terraform_file} disagrees with the script on the openai integration"
    assert app_target in from_terraform[1]

    if use_ddtrace == "true":
        assert from_terraform[0] == "exec=ddtrace-run"
        assert from_terraform[2] == "DD_TRACE_OPENAI_ENABLED=False"
    else:
        assert from_terraform[0] == "exec=uvicorn"
        assert from_terraform[2] == "DD_TRACE_OPENAI_ENABLED=<unset>"


@pytest.mark.parametrize("terraform_file", TERRAFORM_LAUNCH_SITES, ids=lambda p: p.parent.name)
def test_terraform_routes_every_launch_site_through_a_traced_command(terraform_file: Path) -> None:
    """Every place Terraform names a component ASGI target has to honor `USE_DDTRACE`.

    The count is pinned as well: a launch site that is deleted or renamed would otherwise drop
    out of the scan and let this pass while covering less than it claims.
    """
    launch_sites = tuple(line for line in terraform_file.read_text().splitlines() if _APP_TARGET_RE.search(line))

    assert len(launch_sites) == TERRAFORM_LAUNCH_SITES[terraform_file], (
        f"{terraform_file} launch-site count changed; re-check each one honors USE_DDTRACE"
    )
    for line in launch_sites:
        assert "ddtrace-run" in line, f"{terraform_file} launches uvicorn without honoring USE_DDTRACE: {line.strip()}"

    body = terraform_file.read_text()
    for component in ("gateway", "backend"):
        assert f"local.{component}_launch_cmd" in body, (
            f"{terraform_file} defines a {component} launch command but never uses it"
        )


@pytest.mark.parametrize("terraform_file", TERRAFORM_LAUNCH_SITES, ids=lambda p: p.parent.name)
def test_terraform_does_not_depend_on_the_entrypoint_script(terraform_file: Path) -> None:
    """Terraform must not reference a file the caller-supplied image may not contain.

    Both modules default to an image tag published before the script existed, so exec'ing that
    path would fail at container start rather than degrade to an untraced process.
    """
    assert IMAGE_ENTRYPOINT_PATH not in terraform_file.read_text(), (
        f"{terraform_file} depends on a script that images predating it do not ship"
    )


def test_gateway_keeps_its_worker_count_and_backend_keeps_a_single_process() -> None:
    gateway = " ".join(_entrypoint_argv(GATEWAY_DOCKERFILE)) + " " + " ".join(_cmd_argv(GATEWAY_DOCKERFILE))
    backend = " ".join(_entrypoint_argv(BACKEND_DOCKERFILE)) + " " + " ".join(_cmd_argv(BACKEND_DOCKERFILE))

    assert "--workers" in gateway and "NUM_WORKERS" in gateway
    assert "--workers" not in backend and "NUM_WORKERS" not in backend
