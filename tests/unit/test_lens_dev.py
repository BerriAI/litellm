import os
import subprocess
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "lens_dev.sh"

# Fake curl: answers the worker-token check with $CLAIM_STATUS, and key/generate and
# workers/register with a JSON "token". Every call is appended to $CURL_LOG.
FAKE_CURL = """#!/bin/sh
echo "$@" >> "$CURL_LOG"
case "$*" in
  *worker/claim*) printf '%s' "$CLAIM_STATUS" ;;
  */key/generate*) printf '{"token": "%064d"}' 0 ;;
  */lens/workers/register*) printf '{"token": "lens-fresh"}' ;;
esac
"""


def _run(tmp_path: Path, snippet: str, **env: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"\n{snippet}'],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(tmp_path),
            "LENS_DEV_STATE_DIR": str(state),
            "LENS_DEV_PYTHON": sys.executable,
            "CURL_LOG": str(tmp_path / "curl.log"),
            "CLAIM_STATUS": "409",
            **env,
        },
    )


def _curl_calls(tmp_path: Path) -> str:
    log = tmp_path / "curl.log"
    return log.read_text() if log.exists() else ""


def test_missing_worker_token_registers_a_worker(tmp_path):
    proc = _run(tmp_path, "ensure_worker_token")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "state" / "worker_token").read_text().strip() == "lens-fresh"
    assert oct((tmp_path / "state" / "worker_token").stat().st_mode & 0o777) == "0o600"
    assert f'"analysis_key_id": "{0:064d}"' in _curl_calls(tmp_path)


def test_accepted_worker_token_is_reused(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "worker_token").write_text("lens-saved\n")
    proc = _run(tmp_path, "ensure_worker_token", CLAIM_STATUS="409")
    assert proc.returncode == 0, proc.stderr
    assert "reusing worker token" in proc.stdout
    assert (tmp_path / "state" / "worker_token").read_text().strip() == "lens-saved"
    assert "/lens/workers/register" not in _curl_calls(tmp_path)


def test_rejected_worker_token_is_replaced(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "worker_token").write_text("lens-revoked\n")
    proc = _run(tmp_path, "ensure_worker_token", CLAIM_STATUS="401")
    assert proc.returncode == 0, proc.stderr
    assert "was rejected" in proc.stdout
    assert (tmp_path / "state" / "worker_token").read_text().strip() == "lens-fresh"


def test_unexpected_token_check_status_fails(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "worker_token").write_text("lens-saved\n")
    proc = _run(tmp_path, "ensure_worker_token", CLAIM_STATUS="500")
    assert proc.returncode == 1
    assert "unexpected HTTP 500" in proc.stderr


def test_default_master_key_is_random_and_stable(tmp_path):
    first = _run(tmp_path, 'load_master_key; echo "$master_key"')
    second = _run(tmp_path, 'load_master_key; echo "$master_key"')
    assert first.returncode == 0, first.stderr
    key = first.stdout.strip()
    assert key.startswith("sk-") and len(key) == 51 and key != "sk-1234"
    assert second.stdout.strip() == key
    assert oct((tmp_path / "state" / "master_key").stat().st_mode & 0o777) == "0o600"


def test_master_key_override_wins(tmp_path):
    proc = _run(tmp_path, 'load_master_key; echo "$master_key"', LENS_DEV_MASTER_KEY="sk-mine")
    assert proc.stdout.strip() == "sk-mine"
    assert not (tmp_path / "state" / "master_key").exists()


def test_proxy_env_drops_inherited_redis_and_base_urls(tmp_path):
    proc = _run(
        tmp_path,
        'master_key=sk-strong; proxy_env "export OPENAI_API_KEY=from-dotenv"; env',
        REDIS_HOST="redis.example",
        REDIS_PORT="6379",
        REDIS_PASSWORD="secret",
        ANTHROPIC_BASE_URL="http://elsewhere",
        OPENAI_BASE_URL="http://elsewhere",
    )
    assert proc.returncode == 0, proc.stderr
    names = {line.split("=", 1)[0] for line in proc.stdout.splitlines()}
    assert not {n for n in names if n.startswith("REDIS_")}
    assert not names & {"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"}
    assert "OPENAI_API_KEY=from-dotenv" in proc.stdout
    assert "LITELLM_MODE=PRODUCTION" in proc.stdout
    assert "UI_PASSWORD=sk-strong" in proc.stdout
    assert "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY" not in names


def test_proxy_env_permits_the_weak_key_only_when_chosen(tmp_path):
    proc = _run(tmp_path, 'master_key=sk-1234; proxy_env ""; env')
    assert "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true" in proc.stdout


def test_source_development_overrides_an_inherited_release_with_its_own_commit(tmp_path: Path) -> None:
    proc: Final = _run(
        tmp_path,
        'proxy_env "export LITELLM_RELEASE_TAG=v0.0.0-old"; '
        'test "$LITELLM_RELEASE_TAG" = "sha-$(git -C "$repo_root" rev-parse HEAD)"; '
        'printf "%s" "$LENS_WORKER_IMAGE"',
        LITELLM_RELEASE_TAG="v0.0.0-old",
        LENS_WORKER_IMAGE="registry.example/lens-worker:old",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "litellm-lens-worker:local"


def test_external_database_url_never_starts_compose_postgres(tmp_path):
    docker = tmp_path / "bin" / "docker"
    proc = _run(
        tmp_path,
        f"listening() {{ return 1; }}\n"
        f"printf '#!/bin/sh\\necho \"$@\" > {tmp_path}/docker.log\\n' > {docker}; chmod +x {docker}\n"
        "ensure_services",
        LENS_DEV_DATABASE_URL="postgresql://elsewhere/db?schema=public",
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "docker.log").read_text().split()[-2:] == ["--wait", "clickhouse"]


def test_cleanup_kills_child_process_trees(tmp_path):
    proc = _run(
        tmp_path,
        "set -m\n"
        "(sleep 300 & wait) & pids+=($!)\n"
        "child=$!; sleep 0.3\n"
        "cleanup\n"
        "sleep 0.3\n"
        'pgrep -g "$child" >/dev/null && echo LEFTOVER || echo CLEAN',
    )
    assert proc.returncode == 0, proc.stderr
    assert "lens-dev: stopping" in proc.stdout
    assert proc.stdout.strip().endswith("CLEAN")


def test_seed_only_uses_local_credentials_and_profile(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        "parse_args --seed-only --seed large --copies 7; master_key=sk-local; "
        'py() { env; printf "%s\\n" "$@"; }; py=py; seed_data',
    )
    assert proc.returncode == 0, proc.stderr
    assert "LITELLM_MASTER_KEY=sk-local" in proc.stdout
    assert "PROXY_BASE_URL=http://localhost:4000" in proc.stdout
    assert "CLICKHOUSE_DATABASE=litellm" in proc.stdout
    assert "--profile\nlarge\n--copies\n7" in proc.stdout


def test_seed_arguments_reject_invalid_counts_before_startup(tmp_path: Path) -> None:
    proc = _run(tmp_path, "parse_args --seed large --copies 0")
    assert proc.returncode == 1
    assert "positive integer" in proc.stderr


def test_seed_only_defaults_to_small_profile(tmp_path: Path) -> None:
    proc = _run(tmp_path, 'parse_args --seed-only; echo "$seed_profile $seed_only"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "default 1"


def test_seed_only_with_no_cli_count_preserves_env_controls(tmp_path: Path) -> None:
    proc = _run(
        tmp_path,
        "parse_args --seed-only; master_key=sk-local; py() { "
        'printf "%s %s\\n" "$LENS_DEV_SEED_COPIES" "$@"; }; py=py; seed_data',
        LENS_DEV_SEED_COPIES="3",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("3 -m")


def test_proxy_uses_this_checkouts_ui_build(tmp_path: Path) -> None:
    proc = _run(tmp_path, 'proxy_env "export LITELLM_UI_PATH=/old/build"; echo "$LITELLM_UI_PATH"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(ROOT / "ui/litellm-dashboard/out")


def test_dashboard_build_uses_same_origin_and_captures_failures(tmp_path: Path) -> None:
    dashboard = tmp_path / "ui/litellm-dashboard"
    dashboard.mkdir(parents=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "with_dashboard_node.sh"
    runner.write_text('#!/bin/sh\nprintf "base=%s args=%s\\n" "$NEXT_PUBLIC_BASE_URL" "$*"\nexit "$BUILD_STATUS"\n')
    runner.chmod(0o755)
    snippet = f'repo_root="{tmp_path}"; mkdir -p "$log_dir"; build_dashboard'
    success = _run(tmp_path, snippet, BUILD_STATUS="0", LENS_DEV_BUILD_UI="1", NEXT_PUBLIC_BASE_URL="http://old-proxy")
    assert success.returncode == 0, success.stderr
    log = tmp_path / "state/logs/ui-build.log"
    assert log.read_text().strip() == "base= args=npm run build"
    failure = _run(tmp_path, snippet, BUILD_STATUS="1", LENS_DEV_BUILD_UI="1")
    assert failure.returncode == 1
    assert "UI build failed" in failure.stderr


def test_skipping_ui_build_needs_no_static_export(tmp_path: Path) -> None:
    proc = _run(tmp_path, f'repo_root="{tmp_path}"; build_dashboard', LENS_DEV_BUILD_UI="0")
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "ui/litellm-dashboard/out").exists()


def test_ui_readiness_uses_live_login_route(tmp_path: Path) -> None:
    proc = _run(tmp_path, 'wait_for_ui "$$"', LENS_DEV_UI_PORT="3017")
    assert proc.returncode == 0, proc.stderr
    assert "http://localhost:3017/ui/login/" in _curl_calls(tmp_path)


def test_ui_exit_fails_before_readiness_request(tmp_path: Path) -> None:
    proc = _run(tmp_path, 'true & child=$!; wait "$child"; wait_for_ui "$child"')
    assert proc.returncode == 1
    assert "UI exited; see" in proc.stderr
    assert "ui.log" in proc.stderr
    assert _curl_calls(tmp_path) == ""
