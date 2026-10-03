import os
import subprocess
import sys
from pathlib import Path

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
