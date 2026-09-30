import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from litellm.constants import PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DOCKER_ENTRYPOINT: Final = REPO_ROOT / "docker-entrypoint.sh"
DOCKERFILE: Final = REPO_ROOT / "Dockerfile"
TERRAFORM_MODULES: Final = {
    "aws": REPO_ROOT / "terraform" / "litellm" / "aws",
    "gcp": REPO_ROOT / "terraform" / "litellm" / "gcp",
}

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
_TF_ENTRYPOINT_LOCAL_RE: Final = re.compile(r'^\s*image_entrypoint\s*=\s*"([^"]*)"\s*$', re.MULTILINE)
_TF_WORD: Final = r'(?:"[^"\s]*"|tostring\(var\.\w+\)|\$\{var\.\w+\}|[\w.-]+)'
_TF_ENTRYPOINT_CALL_RE: Final = re.compile(rf"exec \$\{{local\.image_entrypoint\}}((?: {_TF_WORD})+)")
_TF_COMPONENT_LIST_RE: Final = re.compile(
    rf"(?:command|args|gateway|backend|ui|migrations|metrics|collector)\s*=\s*"
    rf'\[\s*("(?:proxy|gateway|backend|ui|migrations|metrics|collector)"(?:\s*,\s*{_TF_WORD})*)\s*\]'
)
_TF_VAR_RE: Final = re.compile(r"tostring\(var\.(\w+)\)|\$\{var\.(\w+)\}")

TERRAFORM_VAR_STUBS: Final = {"gateway_num_workers": "2", "gateway_metrics_port": "9464"}
TERRAFORM_WORKLOADS: Final = frozenset({"gateway", "backend", "ui", "migrations", "metrics", "collector"})
COMPONENT_BINARIES: Final = {
    "gateway": "python",
    "backend": "uvicorn",
    "ui": "nginx",
    "migrations": "python",
    "metrics": "python",
    "collector": "python",
}


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


def _tf_word(word: str) -> str:
    var = _TF_VAR_RE.fullmatch(word)
    if var is not None:
        return TERRAFORM_VAR_STUBS[var.group(1) or var.group(2)]
    return word.strip('"')


def _terraform_component_argvs(module_dir: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (tf.name, tuple(_tf_word(word) for word in words))
        for tf in sorted(module_dir.glob("*.tf"))
        for words in (
            *(m.group(1).split() for m in _TF_ENTRYPOINT_CALL_RE.finditer(tf.read_text())),
            *(re.split(r"\s*,\s*", m.group(1)) for m in _TF_COMPONENT_LIST_RE.finditer(tf.read_text())),
        )
    )  # comprehension-ok: the two regex scans over one file read are clearer inline than as a generator helper


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


@pytest.mark.parametrize("module", TERRAFORM_MODULES, ids=str)
def test_terraform_hands_every_workload_to_the_image_entrypoint(module: str) -> None:
    argvs = _terraform_component_argvs(TERRAFORM_MODULES[module])

    assert {argv[0] for _, argv in argvs} == TERRAFORM_WORKLOADS, f"{module} workloads: {argvs}"
    for tf in TERRAFORM_MODULES[module].glob("*.tf"):
        for entrypoint_path in _TF_ENTRYPOINT_LOCAL_RE.findall(tf.read_text()):
            assert entrypoint_path == _entrypoint_argv(DOCKERFILE)[0], f"{tf} execs a path the image does not ship"


@pytest.mark.parametrize("module", TERRAFORM_MODULES, ids=str)
@pytest.mark.parametrize("use_ddtrace", ["true", None])
def test_the_entrypoint_accepts_every_terraform_argv(module: str, use_ddtrace: str | None, tmp_path: Path) -> None:
    for index, (tf, argv) in enumerate(_terraform_component_argvs(TERRAFORM_MODULES[module])):
        component, *extra = argv
        recorded = _run_entrypoint(argv, tmp_path / str(index), use_ddtrace=use_ddtrace)

        expected_exec = "ddtrace-run" if use_ddtrace and component != "ui" else COMPONENT_BINARIES[component]
        assert recorded[0] == f"exec={expected_exec}", f"{module}/{tf} {argv}: {recorded}"
        assert recorded[1].endswith(" ".join(extra)), f"{module}/{tf} {argv} drops its flags: {recorded[1]}"
        if component == "gateway":
            assert extra == ["--workers", TERRAFORM_VAR_STUBS["gateway_num_workers"]], f"{module}/{tf}: {argv}"
        if component == "metrics":
            assert extra == ["--port", TERRAFORM_VAR_STUBS["gateway_metrics_port"]], f"{module}/{tf}: {argv}"


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


def _run_proxy_with_discrete_database(
    tmp_path: Path, extra_env: dict[str, str | None], extra_pythonpath: Path | None = None
) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "litellm").write_text(
        '#!/bin/sh\n{ echo "args=$*"; echo "DATABASE_URL=$DATABASE_URL"; '
        'echo "DATABASE_URL_READ_REPLICA=${DATABASE_URL_READ_REPLICA-<unset>}"; } >> "$RECORD"\n'
    )
    (bin_dir / "litellm").chmod(0o755)
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    record = tmp_path / "record.txt"
    env = _entrypoint_env(
        bin_dir,
        record,
        {
            "DATABASE_URL": None,
            "DATABASE_URL_READ_REPLICA": None,
            "DIRECT_URL": None,
            "DATABASE_MAX_IDLE_CONNECTION_LIFETIME": None,
            "USE_AWS_KMS": None,
            "DATABASE_HOST": "writer.db",
            "DATABASE_PORT": "6543",
            "DATABASE_USER": "app",
            "DATABASE_PASSWORD": "p@ss",
            "DATABASE_NAME": "litellm",
            "DATABASE_HOST_READ_REPLICA": "reader.db",
            "DATABASE_PORT_READ_REPLICA": "6544",
            **extra_env,
        },
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(extra_pythonpath), str(REPO_ROOT)] if extra_pythonpath else [str(REPO_ROOT)]
    )

    result = subprocess.run(
        ["sh", str(DOCKER_ENTRYPOINT), "proxy", "--port", "4000"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout} stderr={result.stderr}"
    return record.read_text().splitlines()


def test_proxy_assembles_database_urls_from_discrete_variables(tmp_path: Path) -> None:
    lines = _run_proxy_with_discrete_database(tmp_path, {})

    assert lines[0] == "args=--port 4000"
    assert lines[1].startswith("DATABASE_URL=postgresql://app:p%40ss@writer.db:6543/litellm")
    assert lines[2].startswith("DATABASE_URL_READ_REPLICA=postgresql://app:p%40ss@reader.db:6544/litellm")
    assert "max_idle_connection_lifetime" not in lines[1]
    assert "max_idle_connection_lifetime" not in lines[2]


def test_proxy_decrypts_a_kms_database_password_before_assembling(tmp_path: Path) -> None:
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(
        "import os\n"
        "import litellm.secret_managers.aws_secret_manager as kms\n"
        "kms.decrypt_env_var = lambda: {'DATABASE_PASSWORD': 'p@ss'} "
        "if os.environ.get('DATABASE_PASSWORD') == 'aws_kms/ciphertext' else {}\n"
    )
    lines = _run_proxy_with_discrete_database(
        tmp_path,
        {"USE_AWS_KMS": "True", "DATABASE_PASSWORD": "aws_kms/ciphertext"},
        extra_pythonpath=site_dir,
    )

    assert lines[1].startswith("DATABASE_URL=postgresql://app:p%40ss@writer.db:6543/litellm")
    assert lines[2].startswith("DATABASE_URL_READ_REPLICA=postgresql://app:p%40ss@reader.db:6544/litellm")


def test_proxy_pins_an_env_configured_idle_lifetime(tmp_path: Path) -> None:
    lines = _run_proxy_with_discrete_database(tmp_path, {"DATABASE_MAX_IDLE_CONNECTION_LIFETIME": "120"})

    assert "max_idle_connection_lifetime=120" in lines[1]
    assert "max_idle_connection_lifetime=120" in lines[2]


def test_proxy_keeps_an_operator_database_url(tmp_path: Path) -> None:
    record = _run_entrypoint(
        ("proxy",),
        tmp_path,
        DATABASE_URL="postgresql://pinned/db",
        DATABASE_HOST="writer.db",
    )

    assert record[0] == "exec=litellm"
