"""Unit tests for `docker-entrypoint.sh`, the component dispatcher every shipped
container runs, and for the Terraform launch commands that must agree with it."""

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_ENTRYPOINT = REPO_ROOT / "docker-entrypoint.sh"
DOCKERFILE = REPO_ROOT / "Dockerfile"
TERRAFORM_ECS = REPO_ROOT / "terraform" / "litellm" / "aws" / "ecs.tf"
TERRAFORM_CLOUDRUN = REPO_ROOT / "terraform" / "litellm" / "gcp" / "cloudrun.tf"

IMAGE_ENTRYPOINT_PATH = "/app/docker-entrypoint.sh"
STUBBED_EXECUTABLES = ("ddtrace-run", "uvicorn", "python", "litellm", "nginx")

TRUTHY_USE_DDTRACE = ("true", "True", "TRUE", "tRuE")
FALSY_USE_DDTRACE = (None, "", "false", "False", "1", "yes", "on", "truex")

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
_APP_TARGET_RE = re.compile(r"(?:gateway|backend)\.main:app|gateway\.launch")
_TF_STRING_LOCAL_RE = re.compile(r'^\s*(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"\s*$', re.MULTILINE)
_TF_INTERPOLATION_RE = re.compile(r"\$\{(local|var)\.(\w+)\}")

TERRAFORM_LAUNCH_SITES = {TERRAFORM_ECS: 2, TERRAFORM_CLOUDRUN: 2}
TERRAFORM_VAR_STUBS = {"gateway_num_workers": "2"}
COMPONENT_LAUNCHERS = {
    "gateway": ("python", "-m", "gateway.launch"),
    "backend": ("uvicorn", "backend.main:app"),
}
_MAX_INTERPOLATION_PASSES = 5


def _write_stubs(bin_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        stub = bin_dir / name
        stub.write_text(_STUB_TEMPLATE.format(name=name))
        stub.chmod(0o755)


def _entrypoint_env(bin_dir: Path, record: Path, overrides: dict[str, str | None]) -> dict[str, str]:
    """The container-like environment the entrypoint runs under, with `overrides` applied (None unsets)."""
    cleared = ("USE_DDTRACE", "DD_TRACE_OPENAI_ENABLED", "LITELLM_COMPONENT", "NUM_WORKERS", *overrides)
    return {
        **{k: v for k, v in os.environ.items() if k not in cleared},
        **{k: v for k, v in overrides.items() if v is not None},
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RECORD": str(record),
        "PYTHONPATH": PYTHONPATH_SENTINEL,
    }


def _run_entrypoint(
    argv: tuple[str, ...],
    tmp_path: Path,
    use_ddtrace: str | None = None,
    **overrides: str | None,
) -> tuple[str, ...]:
    """Run the entrypoint with stubbed executables on PATH and return the recorded lines."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stubs(bin_dir, STUBBED_EXECUTABLES)
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"USE_DDTRACE": use_ddtrace, **overrides})

    result = subprocess.run(["sh", str(DOCKER_ENTRYPOINT), *argv], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    return tuple(record.read_text().splitlines()) if record.exists() else ()


def _run_shell_command(command: str, bin_dir: Path, record: Path, use_ddtrace: str | None) -> tuple[str, ...]:
    """Run a resolved Terraform launch command through `sh -c` and return the recorded lines."""
    env = _entrypoint_env(bin_dir, record, {"USE_DDTRACE": use_ddtrace})
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
    return tuple(str(part) for part in json.loads(matches[-1]))


@pytest.mark.parametrize(
    "argv, expected_exec, expected_args",
    [
        (("proxy", "--config", "/app/config.yaml"), "exec=litellm", "args=--config /app/config.yaml"),
        (("gateway",), "exec=python", "args=-m gateway.launch --workers 1 --host 0.0.0.0 --port 4000"),
        (
            ("gateway", "--port", "8080"),
            "exec=python",
            "args=-m gateway.launch --workers 1 --host 0.0.0.0 --port 4000 --port 8080",
        ),
        (("backend",), "exec=uvicorn", "args=backend.main:app --host 0.0.0.0 --port 4001"),
        (("backend", "--port", "9001"), "exec=uvicorn", "args=backend.main:app --host 0.0.0.0 --port 4001 --port 9001"),
        (("ui",), "exec=nginx", "args=-g daemon off;"),
        (("migrations",), "exec=python", "args=/app/migrations/run.py"),
        (("metrics", "--port", "9090"), "exec=python", "args=-m litellm.proxy.prometheus_metrics_server --port 9090"),
        (("collector",), "exec=python", "args=-m litellm.proxy.collector"),
    ],
)
def test_the_first_argument_selects_the_component(
    argv: tuple[str, ...], expected_exec: str, expected_args: str, tmp_path: Path
) -> None:
    recorded = _run_entrypoint(argv, tmp_path)

    assert recorded[:2] == (expected_exec, expected_args)


@pytest.mark.parametrize(
    "argv, expected",
    [
        ((), ("exec=litellm", "args=")),
        (("--port", "4000"), ("exec=litellm", "args=--port 4000")),
        (
            ("--config", "/app/config.yaml", "--detailed_debug"),
            ("exec=litellm", "args=--config /app/config.yaml --detailed_debug"),
        ),
    ],
)
def test_flags_alone_still_run_the_monolithic_proxy(
    argv: tuple[str, ...], expected: tuple[str, str], tmp_path: Path
) -> None:
    """Every `docker run litellm --config ...` written before components existed keeps working."""
    assert _run_entrypoint(argv, tmp_path)[:2] == expected


def test_the_dockerfile_leaves_the_command_empty_so_the_env_var_can_pick_the_component(tmp_path: Path) -> None:
    """A CMD naming a component would beat LITELLM_COMPONENT, and the proxy already listens on 4000 with no flags."""
    dockerfile_text = DOCKERFILE.read_text()
    assert _entrypoint_argv(DOCKERFILE) == (IMAGE_ENTRYPOINT_PATH,), "the image ENTRYPOINT must be the bare dispatcher"
    assert not _CMD_RE.search(dockerfile_text), "the Dockerfile must not set a CMD"

    assert _run_entrypoint((), tmp_path)[:2] == ("exec=litellm", "args=")


@pytest.mark.parametrize(
    "component, argv, expected_exec, expected_args",
    [
        ("backend", (), "exec=uvicorn", "args=backend.main:app --host 0.0.0.0 --port 4001"),
        (
            "gateway",
            ("--port", "4100"),
            "exec=python",
            "args=-m gateway.launch --workers 1 --host 0.0.0.0 --port 4000 --port 4100",
        ),
        ("ui", (), "exec=nginx", "args=-g daemon off;"),
    ],
)
def test_litellm_component_env_selects_the_component_when_the_command_has_none(
    component: str, argv: tuple[str, ...], expected_exec: str, expected_args: str, tmp_path: Path
) -> None:
    """Helm and Compose can pick the component with an env var and keep `args` for the process flags."""
    recorded = _run_entrypoint(argv, tmp_path, LITELLM_COMPONENT=component)

    assert recorded[:2] == (expected_exec, expected_args)


def test_an_explicit_component_argument_beats_the_env_var(tmp_path: Path) -> None:
    recorded = _run_entrypoint(("backend",), tmp_path, LITELLM_COMPONENT="gateway")

    assert recorded[0] == "exec=uvicorn"


def test_the_gateway_honours_num_workers(tmp_path: Path) -> None:
    recorded = _run_entrypoint(("gateway",), tmp_path, NUM_WORKERS="4")

    assert recorded[1] == "args=-m gateway.launch --workers 4 --host 0.0.0.0 --port 4000"


def test_an_unknown_first_word_is_run_as_the_command(tmp_path: Path) -> None:
    """`docker run <image> sh -c ...` and `kubectl exec`-style overrides bypass the dispatcher."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, ("some-tool",))
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), "some-tool", "--flag"], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    assert record.read_text().splitlines()[:2] == ["exec=some-tool", "args=--flag"]


def test_an_unknown_litellm_component_fails_fast_instead_of_guessing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, STUBBED_EXECUTABLES)
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"LITELLM_COMPONENT": "gatway"})

    result = subprocess.run(["sh", str(DOCKER_ENTRYPOINT)], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 64
    assert "gatway" in result.stderr
    assert not record.exists(), "nothing may be exec'd when the component is unknown"


def test_ddtrace_enabled_wraps_the_command_and_disables_the_openai_integration(tmp_path: Path) -> None:
    """`USE_DDTRACE=true` must prefix the command with `ddtrace-run`, turn the openai
    integration off, and leave PYTHONPATH alone.

    `ddtrace-run` installs its instrumentation by PREPENDING a bootstrap directory to
    PYTHONPATH, and the image sets PYTHONPATH=/app so the component packages are
    importable. A wrapper that assigned PYTHONPATH instead of inheriting it would either
    drop the bootstrap (silently disabling tracing) or drop /app (breaking the import), so
    the recorded value is asserted verbatim.

    PYTHONPATH_SENTINEL deliberately differs from the image's own /app: with /app as the
    fixture value, a wrapper that overwrote PYTHONPATH with /app would still satisfy this
    assertion and the check would prove nothing.

    The openai integration is disabled because under `ddtrace-run` the bootstrap patches
    it before any litellm code runs, so litellm's in-process `patch_all(..., openai=False)`
    can no longer suppress it; leaving it on double-reports every LLM call.
    """
    recorded = _run_entrypoint(("gateway",), tmp_path, use_ddtrace="true", NUM_WORKERS="2")

    assert recorded == (
        "exec=ddtrace-run",
        "args=python -m gateway.launch --workers 2 --host 0.0.0.0 --port 4000",
        "DD_TRACE_OPENAI_ENABLED=False",
        f"PYTHONPATH={PYTHONPATH_SENTINEL}",
    )


def test_ddtrace_disabled_execs_the_command_directly(tmp_path: Path) -> None:
    recorded = _run_entrypoint(("backend", "--workers", "2"), tmp_path)

    assert recorded == (
        "exec=uvicorn",
        "args=backend.main:app --host 0.0.0.0 --port 4001 --workers 2",
        "DD_TRACE_OPENAI_ENABLED=<unset>",
        f"PYTHONPATH={PYTHONPATH_SENTINEL}",
    )


@pytest.mark.parametrize(
    "use_ddtrace, traced",
    [*((v, True) for v in TRUTHY_USE_DDTRACE), *((v, False) for v in FALSY_USE_DDTRACE)],
)
def test_ddtrace_gating_matches_get_secret_bool_for_every_component(
    use_ddtrace: str | None, traced: bool, tmp_path: Path
) -> None:
    """The shell gate must accept exactly the spellings `get_secret_bool` accepts.

    `ProxyStartupEvent._init_dd_tracer` reads `USE_DDTRACE` through `get_secret_bool`, which
    matches `true` case-insensitively. If the shell gate were stricter, `USE_DDTRACE=True` would
    give in-process LLM spans without `ddtrace-run` HTTP spans, a half-enabled state.
    """
    component = _run_entrypoint(("gateway",), tmp_path / "component", use_ddtrace=use_ddtrace)
    monolith = _run_entrypoint(("--port", "4000"), tmp_path / "monolith", use_ddtrace=use_ddtrace)

    expected_openai = "DD_TRACE_OPENAI_ENABLED=False" if traced else "DD_TRACE_OPENAI_ENABLED=<unset>"
    assert component[0] == ("exec=ddtrace-run" if traced else "exec=python")
    assert component[2] == expected_openai
    assert monolith[0] == ("exec=ddtrace-run" if traced else "exec=litellm")
    assert monolith[1] == ("args=litellm --port 4000" if traced else "args=--port 4000")
    assert monolith[2] == expected_openai


def test_wipes_the_prometheus_multiproc_dir_before_uvicorn_forks(tmp_path: Path) -> None:
    """A restarted container inherits the emptyDir of its predecessor, whose worker pids it may reuse, so the
    stale .db files must be gone before any worker opens the one carrying its own pid."""
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "gauge_livesum_7.db").write_bytes(b"stale")
    (multiproc_dir / "counter_7.db").write_bytes(b"stale")
    (multiproc_dir / "keep.txt").write_text("not a sample")

    recorded = _run_entrypoint(("gateway",), tmp_path, PROMETHEUS_MULTIPROC_DIR=str(multiproc_dir))

    assert sorted(p.name for p in multiproc_dir.iterdir()) == ["keep.txt"]
    assert recorded[0] == "exec=python"


@pytest.mark.parametrize(
    "argv",
    [
        ("python", "-m", "litellm.proxy.prometheus_metrics_server"),
        ("python", "-m", "litellm.proxy.collector"),
        ("sh", "-c", "exec python -m litellm.proxy.collector --port 4010"),
    ],
)
def test_a_raw_metrics_reader_keeps_the_workers_samples(argv: tuple[str, ...], tmp_path: Path) -> None:
    """A raw command that only reads PROMETHEUS_MULTIPROC_DIR, like a separately launched metrics
    server or collector, must not delete the samples the gateway workers are still writing."""
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "counter_7.db").write_bytes(b"live")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, ("python",))
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"PROMETHEUS_MULTIPROC_DIR": str(multiproc_dir)})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), *argv],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    assert [p.name for p in multiproc_dir.iterdir()] == ["counter_7.db"]
    assert record.read_text().splitlines()[0] == "exec=python"


@pytest.mark.parametrize(
    "argv",
    [
        ("python", "-m", "gateway.launch"),
        ("litellm", "--config", "/app/config.yaml"),
        ("uvicorn", "backend.main:app"),
    ],
)
def test_a_raw_writer_starts_with_an_empty_prometheus_multiproc_dir(argv: tuple[str, ...], tmp_path: Path) -> None:
    """A raw command that writes samples, restarted on a persistent multiproc dir, must not keep
    stale .db files from the previous run; files that are not samples stay."""
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "counter_7.db").write_bytes(b"stale")
    (multiproc_dir / "gauge_livesum_7.db").write_bytes(b"stale")
    (multiproc_dir / "keep.txt").write_text("not a sample")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, STUBBED_EXECUTABLES)
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"PROMETHEUS_MULTIPROC_DIR": str(multiproc_dir)})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), *argv],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    assert sorted(p.name for p in multiproc_dir.iterdir()) == ["keep.txt"]
    assert record.read_text().splitlines()[0] == f"exec={argv[0]}"


def test_a_raw_command_still_gets_the_prometheus_multiproc_dir_created(tmp_path: Path) -> None:
    """`docker run <image> python -m gateway.launch` cannot write samples into a directory that is not there."""
    multiproc_dir = tmp_path / "not-yet" / "multiproc"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, ("python",))
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"PROMETHEUS_MULTIPROC_DIR": str(multiproc_dir)})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), "python", "-m", "gateway.launch"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    assert multiproc_dir.is_dir()
    assert record.read_text().splitlines()[0] == "exec=python"


@pytest.mark.parametrize("component", ["metrics", "collector"])
def test_readers_of_the_prometheus_multiproc_dir_keep_the_workers_samples(component: str, tmp_path: Path) -> None:
    """The metrics and collector sidecars share the volume with gateway workers that are already
    serving traffic, so their (re)start must not erase the samples those workers have written."""
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "counter_12.db").write_bytes(b"live")
    (multiproc_dir / "histogram_13.db").write_bytes(b"live")

    recorded = _run_entrypoint((component,), tmp_path, PROMETHEUS_MULTIPROC_DIR=str(multiproc_dir))

    assert sorted(p.name for p in multiproc_dir.iterdir()) == ["counter_12.db", "histogram_13.db"]
    assert recorded[0] == "exec=python"


def test_creates_a_missing_prometheus_multiproc_dir(tmp_path: Path) -> None:
    missing = tmp_path / "multiproc"

    _run_entrypoint(("backend",), tmp_path, PROMETHEUS_MULTIPROC_DIR=str(missing))

    assert missing.is_dir()


@pytest.mark.parametrize("terraform_file", TERRAFORM_LAUNCH_SITES, ids=lambda p: p.parent.name)
@pytest.mark.parametrize("component", ["gateway", "backend"])
@pytest.mark.parametrize("use_ddtrace", [*TRUTHY_USE_DDTRACE, *FALSY_USE_DDTRACE])
def test_terraform_launch_command_matches_the_script_contract(
    terraform_file: Path, component: str, use_ddtrace: str | None, tmp_path: Path
) -> None:
    """The Terraform command and `docker-entrypoint.sh <component>` must decide identically.

    The decision deliberately lives in two places. The script is what the image ENTRYPOINT runs;
    the Terraform strings are what runs when a deployment overrides that ENTRYPOINT, and they
    cannot call the script because the caller supplies the image tag and it may predate the file.
    So instead of asserting a shared path, this runs both implementations under the same
    environment and asserts they agree on which binary is exec'd and on whether the openai
    integration is disabled.
    """
    launcher = COMPONENT_LAUNCHERS[component]
    app_target = " ".join(launcher[1:])
    command = _resolve_tf_local(terraform_file, f"{component}_launch_cmd")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stubs(bin_dir, ("ddtrace-run", "uvicorn", "python"))
    from_terraform = _run_shell_command(command, bin_dir, tmp_path / "terraform.txt", use_ddtrace)

    from_script = _run_entrypoint((component,), tmp_path / "script", use_ddtrace=use_ddtrace)

    assert from_terraform[0] == from_script[0], (
        f"{terraform_file} disagrees with the script on USE_DDTRACE={use_ddtrace}"
    )
    assert from_terraform[2] == from_script[2], f"{terraform_file} disagrees with the script on the openai integration"
    assert app_target in from_terraform[1]
    assert app_target in from_script[1]
    assert "gateway.main:app" not in from_terraform[1], f"{terraform_file} bypasses the gateway.launch supervisor"

    if use_ddtrace in TRUTHY_USE_DDTRACE:
        assert from_terraform[0] == "exec=ddtrace-run"
        assert from_terraform[1].startswith(f"args={launcher[0]} ")
        assert from_terraform[2] == "DD_TRACE_OPENAI_ENABLED=False"
    else:
        assert from_terraform[0] == f"exec={launcher[0]}"
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


def test_entrypoint_script_is_executable() -> None:
    mode = DOCKER_ENTRYPOINT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be committed executable to run as the image ENTRYPOINT"
    assert mode & stat.S_IXOTH, "entrypoint must be executable by the unprivileged `nonroot` user"


def test_entrypoint_script_has_no_carriage_returns() -> None:
    assert b"\r" not in DOCKER_ENTRYPOINT.read_bytes()
