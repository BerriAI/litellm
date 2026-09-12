import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
START_IDP = ROOT / ".github/e2e-stack/start-idp.sh"


def run_start(tmp_path: Path, *, platform: str = "Linux", failure: str = ""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "docker.jsonl"
    programs = {
        "docker": """import json, os, sys
with open(os.environ['DOCKER_LOG'], 'a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ['FAILURE'] == 'schema' and 'psql' in sys.argv:
    sys.exit(17)
if os.environ['FAILURE'] == 'launch' and '--name' in sys.argv:
    sys.exit(18)
""",
        "curl": "import os, sys; sys.exit(1 if os.environ['FAILURE'] == 'readiness' else 0)\n",
        "uname": "import os; print(os.environ['PLATFORM'])\n",
    }
    for name, source in programs.items():
        program = bin_dir / name
        program.write_text(f"#!{sys.executable}\n{source}")
        program.chmod(0o755)
    result = subprocess.run(
        ["bash", str(START_IDP)],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DOCKER_LOG": str(calls),
            "PLATFORM": platform,
            "FAILURE": failure,
            "DATABASE_HOST": "127.0.0.1",
            "DATABASE_PORT": "5544",
            "DATABASE_USER": "fixture_user",
            "DATABASE_PASSWORD": "fixture_password",
            "DATABASE_NAME": "fixture_db",
            "E2E_KEYCLOAK_PORT": "8181",
            "E2E_KEYCLOAK_STARTUP_TIMEOUT": "0",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, [json.loads(line) for line in calls.read_text().splitlines()]


@pytest.mark.parametrize("platform", ("Linux", "Darwin"))
def test_idp_uses_existing_database_and_imports_runner_realm(tmp_path: Path, platform: str) -> None:
    result, calls = run_start(tmp_path, platform=platform)

    assert result.returncode == 0, result.stderr
    schema, _, launch = calls
    host = "127.0.0.1" if platform == "Linux" else "host.docker.internal"
    assert schema[schema.index("-h") + 1] == host
    assert schema[schema.index("-p") + 1] == "5544"
    assert "ON_ERROR_STOP=1" in schema
    assert "CREATE SCHEMA IF NOT EXISTS keycloak" in schema
    assert f"KC_DB_URL_HOST={host}" in launch
    assert "KC_DB_URL_PORT=5544" in launch
    assert "KC_DB_SCHEMA=keycloak" in launch
    assert "KC_DB_POOL_MAX_SIZE=10" in launch
    assert f"{ROOT}/tests/e2e/idp_realm.json:/opt/keycloak/data/import/realm.json:ro" in launch
    assert "KC_HTTP_PORT=8181" in launch
    if platform == "Linux":
        assert launch[launch.index("--network") + 1] == "host"
    else:
        assert launch[launch.index("-p") + 1] == "127.0.0.1:8181:8181"
    assert "Keycloak realm is up" in result.stdout


@pytest.mark.parametrize(("failure", "code"), (("schema", 17), ("launch", 18), ("readiness", 1)))
def test_idp_failure_stops_stack_startup(tmp_path: Path, failure: str, code: int) -> None:
    result, calls = run_start(tmp_path, failure=failure)

    assert result.returncode == code
    assert "Keycloak realm is up" not in result.stdout
    if failure == "schema":
        assert len(calls) == 1, "do not replace an IdP when its database is unavailable"


def test_proxy_and_test_runner_share_the_same_jwt_configuration() -> None:
    config = yaml.safe_load((ROOT / "tests/e2e/gateway/stage_mirror_ci_config.yml").read_text())
    general = config["general_settings"]
    assert general["enable_jwt_auth"] is True
    assert general["litellm_jwtauth"] == {
        "user_id_jwt_field": "sub",
        "user_email_jwt_field": "email",
        "team_ids_jwt_field": "groups",
        "user_id_upsert": True,
    }
    up = (START_IDP.parent / "up.sh").read_text()
    assert '"JWT_ISSUER=http://127.0.0.1:${KEYCLOAK_PORT}/realms/litellm-e2e"' in up
    assert '"JWT_AUDIENCE=litellm-e2e"' in up
    assert "E2E_KEYCLOAK_URL=http://127.0.0.1:${KEYCLOAK_PORT}" in up
    assert up.index("bash .github/e2e-stack/start-idp.sh") < up.index("start_server backend")
    assert "e2e-keycloak" in (START_IDP.parent / "down.sh").read_text()
