import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Final

import pytest

from litellm.constants import PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DOCKER_ENTRYPOINT: Final = REPO_ROOT / "docker-entrypoint.sh"
DOCKERFILE: Final = REPO_ROOT / "Dockerfile"
TERRAFORM_ECS: Final = REPO_ROOT / "terraform" / "litellm" / "aws" / "ecs.tf"
TERRAFORM_CLOUDRUN: Final = REPO_ROOT / "terraform" / "litellm" / "gcp" / "cloudrun.tf"

IMAGE_ENTRYPOINT_PATH: Final = "/app/docker-entrypoint.sh"
STUBBED_EXECUTABLES: Final = ("ddtrace-run", "uvicorn", "python", "litellm", "nginx")

TRUTHY_USE_DDTRACE: Final = ("true", "True", "TRUE", "tRuE")
FALSY_USE_DDTRACE: Final = (None, "", "false", "False", "1", "yes", "on", "truex")

PYTHONPATH_SENTINEL: Final = "/lit-entrypoint-sentinel:/app"

_STUB_TEMPLATE: Final = """#!/bin/sh
{{
  echo "exec={name}"
  echo "args=$*"
  echo "DD_TRACE_OPENAI_ENABLED=${{DD_TRACE_OPENAI_ENABLED-<unset>}}"
  echo "PYTHONPATH=${{PYTHONPATH-<unset>}}"
}} >> "$RECORD"
"""

_ENTRYPOINT_RE: Final = re.compile(r"^ENTRYPOINT\s+(\[.*\])\s*$", re.MULTILINE)
_CMD_RE: Final = re.compile(r"^CMD\s+(\[.*\])\s*$", re.MULTILINE)
_APP_TARGET_RE: Final = re.compile(r"(?:gateway|backend)\.main:app|gateway\.launch")
_TF_STRING_LOCAL_RE: Final = re.compile(r'^\s*(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"\s*$', re.MULTILINE)
_TF_INTERPOLATION_RE: Final = re.compile(r"\$\{(local|var)\.(\w+)\}")

TERRAFORM_LAUNCH_SITES: Final = {TERRAFORM_ECS: 2, TERRAFORM_CLOUDRUN: 2}
TERRAFORM_VAR_STUBS: Final = {"gateway_num_workers": "2"}
COMPONENT_LAUNCHERS: Final = {
    "gateway": ("python", "-m", "gateway.launch"),
    "backend": ("uvicorn", "backend.main:app"),
}
_MAX_INTERPOLATION_PASSES: Final = 5


def _write_stubs(bin_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        stub = bin_dir / name
        stub.write_text(_STUB_TEMPLATE.format(name=name))
        stub.chmod(0o755)


def _entrypoint_env(bin_dir: Path, record: Path, overrides: dict[str, str | None]) -> dict[str, str]:
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
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stubs(bin_dir, STUBBED_EXECUTABLES)
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"USE_DDTRACE": use_ddtrace, **overrides})

    result = subprocess.run(["sh", str(DOCKER_ENTRYPOINT), *argv], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    return tuple(record.read_text().splitlines()) if record.exists() else ()


def _run_shell_command(command: str, bin_dir: Path, record: Path, use_ddtrace: str | None) -> tuple[str, ...]:
    env = _entrypoint_env(bin_dir, record, {"USE_DDTRACE": use_ddtrace})
    result = subprocess.run(["sh", "-c", command], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    return tuple(record.read_text().splitlines()) if record.exists() else ()


def _resolve_tf_local(terraform_file: Path, name: str) -> str:
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
    assert _run_entrypoint(argv, tmp_path)[:2] == expected


def test_the_dockerfile_leaves_the_command_empty_so_the_env_var_can_pick_the_component(tmp_path: Path) -> None:
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
    recorded = _run_entrypoint(argv, tmp_path, LITELLM_COMPONENT=component)

    assert recorded[:2] == (expected_exec, expected_args)


def test_an_explicit_component_argument_beats_the_env_var(tmp_path: Path) -> None:
    recorded = _run_entrypoint(("backend",), tmp_path, LITELLM_COMPONENT="gateway")

    assert recorded[0] == "exec=uvicorn"


def test_the_gateway_honours_num_workers(tmp_path: Path) -> None:
    recorded = _run_entrypoint(("gateway",), tmp_path, NUM_WORKERS="4")

    assert recorded[1] == "args=-m gateway.launch --workers 4 --host 0.0.0.0 --port 4000"


def test_an_unknown_first_word_is_run_as_the_command(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    "use_ddtrace, traced",
    [*((v, True) for v in TRUTHY_USE_DDTRACE), *((v, False) for v in FALSY_USE_DDTRACE)],
)
def test_ddtrace_wraps_a_raw_command_like_a_component(use_ddtrace: str | None, traced: bool, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, ("some-tool", "ddtrace-run"))
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {"USE_DDTRACE": use_ddtrace})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), "some-tool", "--flag"], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    assert record.read_text().splitlines()[:2] == (
        ["exec=ddtrace-run", "args=some-tool --flag"] if traced else ["exec=some-tool", "args=--flag"]
    )


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


def test_admin_agent_flag_runs_the_bundled_liteadmin_agent_not_the_proxy(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stubs(bin_dir, STUBBED_EXECUTABLES)
    record = tmp_path / "record.txt"
    env = _entrypoint_env(bin_dir, record, {})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), "--admin-agent", "--flag"], env=env, capture_output=True, text=True, check=False
    )

    assert "/opt/liteadmin/bin/litellm-admin-agent" in result.stderr
    assert result.returncode == 127
    assert not record.exists(), "--admin-agent must not fall through to the proxy"


def test_ddtrace_enabled_wraps_the_command_and_disables_the_openai_integration(tmp_path: Path) -> None:
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
    component = _run_entrypoint(("gateway",), tmp_path / "component", use_ddtrace=use_ddtrace)
    monolith = _run_entrypoint(("--port", "4000"), tmp_path / "monolith", use_ddtrace=use_ddtrace)

    expected_openai = "DD_TRACE_OPENAI_ENABLED=False" if traced else "DD_TRACE_OPENAI_ENABLED=<unset>"
    assert component[0] == ("exec=ddtrace-run" if traced else "exec=python")
    assert component[2] == expected_openai
    assert monolith[0] == ("exec=ddtrace-run" if traced else "exec=litellm")
    assert monolith[1] == ("args=litellm --port 4000" if traced else "args=--port 4000")
    assert monolith[2] == expected_openai


def test_wipes_the_prometheus_multiproc_dir_before_uvicorn_forks(tmp_path: Path) -> None:
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "gauge_livesum_7.db").write_bytes(b"stale")
    (multiproc_dir / "counter_7.db").write_bytes(b"stale")
    (multiproc_dir / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric").write_bytes(b"stale")
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
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "counter_7.db").write_bytes(b"live")
    (multiproc_dir / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric").write_bytes(b"live")
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
    assert sorted(p.name for p in multiproc_dir.iterdir()) == [
        "counter_7.db",
        f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric",
    ]
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
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "counter_7.db").write_bytes(b"stale")
    (multiproc_dir / "gauge_livesum_7.db").write_bytes(b"stale")
    (multiproc_dir / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric").write_bytes(b"stale")
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
    multiproc_dir = tmp_path / "multiproc"
    multiproc_dir.mkdir()
    (multiproc_dir / "counter_12.db").write_bytes(b"live")
    (multiproc_dir / "histogram_13.db").write_bytes(b"live")
    (multiproc_dir / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric").write_bytes(b"live")

    recorded = _run_entrypoint((component,), tmp_path, PROMETHEUS_MULTIPROC_DIR=str(multiproc_dir))

    assert sorted(p.name for p in multiproc_dir.iterdir()) == [
        "counter_12.db",
        "histogram_13.db",
        f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric",
    ]
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
    assert IMAGE_ENTRYPOINT_PATH not in terraform_file.read_text(), (
        f"{terraform_file} depends on a script that images predating it do not ship"
    )


def test_help_prints_the_header_comment_block_without_shell_source(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    _write_stubs(bin_dir, STUBBED_EXECUTABLES)
    env = _entrypoint_env(bin_dir, tmp_path / "record.txt", {})

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), "help"], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    assert "--admin-agent" in result.stderr
    assert "collector" in result.stderr
    assert "set -eu" not in result.stderr
    assert result.stderr.rstrip().endswith("other raw commands clear stale samples.")


def test_entrypoint_script_is_executable() -> None:
    mode = DOCKER_ENTRYPOINT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be committed executable to run as the image ENTRYPOINT"
    assert mode & stat.S_IXOTH, "entrypoint must be executable by the unprivileged `nonroot` user"


def test_entrypoint_script_has_no_carriage_returns() -> None:
    assert b"\r" not in DOCKER_ENTRYPOINT.read_bytes()
