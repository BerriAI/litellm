import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process

MODEL_IDS: Final = ("gpt-5.4-mini", "Claude-sonnet", "azure-gpt", "bedrock-llama", "Anthropic-haiku")
EDGE_IDS: Final = ("z-last", "alpha", "ALPHA", "éclair", "Éclair", "2-model", "10-model", "ß", "ss")
EDGE_SORTED_IDS: Final = ("10-model", "2-model", "alpha", "ALPHA", "ß", "ss", "z-last", "éclair", "Éclair")


def _config(directory: Path, gateway: Gateway, model_ids: tuple[str, ...]) -> Path:
    config: Final = directory / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {
                            "model": "openai/integration-model",
                            "api_base": f"{gateway.upstream_url}/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                    for name in model_ids
                ],
                "router_settings": {"redis_host": "os.environ/REDIS_HOST", "redis_port": "os.environ/REDIS_PORT"},
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "coordination_redis": {"host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                },
            }
        )
    )
    return config


def _run_cli(
    gateway: Gateway,
    home: Path,
    arguments: tuple[str, ...],
    *,
    shell_input: str | None = None,
    api_key: str | None = None,
) -> subprocess.CompletedProcess[str]:
    root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    return subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "from litellm.cli.main import cli; cli()",
            "--base-url",
            str(gateway.client.base_url),
            "--api-key",
            gateway.key if api_key is None else api_key,
            *arguments,
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "HOME": str(home), "COLUMNS": "140", "NO_COLOR": "1"},
        input=shell_input,
        text=True,
        capture_output=True,
        timeout=60,
    )


def _table_ids(output: str) -> tuple[str, ...]:
    return tuple(
        line.split("│")[1].strip()
        for line in output.splitlines()
        if line.startswith("│") and line.split("│")[1].strip() != "ID"
    )


def _assert_no_upstream_calls(gateway: Gateway) -> None:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=10, trust_env=False)
    assert response.status_code == 200, response.text
    assert response.json()["requests"] == [], response.text


@pytest.fixture(scope="module")
def cli_gateway(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("cli-models")
    config: Final = _config(directory, session_gateway, MODEL_IDS)
    with owned_proxy(session_gateway, directory, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as proxy:
        yield proxy


@pytest.mark.parametrize("no_sort", (False, True), ids=("alphabetical", "proxy-order"))
def test_cli_models_list_order(cli_gateway: Gateway, no_sort: bool, tmp_path: Path) -> None:
    response: Final = cli_gateway.request("GET", "/v1/models")
    assert response.status_code == 200, response.text
    proxy_ids: Final = tuple(model["id"] for model in response.json()["data"])
    assert set(proxy_ids) == set(MODEL_IDS), response.text
    assert proxy_ids != tuple(sorted(MODEL_IDS, key=str.casefold)), response.text
    result: Final = _run_cli(
        cli_gateway, tmp_path, ("models", "list", "--format", "json", *(("--no-sort",) if no_sort else ()))
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected: Final = proxy_ids if no_sort else tuple(sorted(MODEL_IDS, key=str.casefold))
    assert tuple(model["id"] for model in json.loads(result.stdout)) == expected, result.stdout
    _assert_no_upstream_calls(cli_gateway)


@pytest.mark.parametrize("flag", ((), ("--sort",), ("--no-sort",)), ids=("default", "sort", "no-sort"))
def test_cli_models_list_table(cli_gateway: Gateway, tmp_path: Path, flag: tuple[str, ...]) -> None:
    result: Final = _run_cli(cli_gateway, tmp_path, ("models", "list", *flag))
    assert result.returncode == 0, result.stdout + result.stderr
    expected: Final = MODEL_IDS if flag == ("--no-sort",) else tuple(sorted(MODEL_IDS, key=str.casefold))
    assert _table_ids(result.stdout) == expected, result.stdout
    assert "Available Models" in result.stdout
    assert "Object" in result.stdout and "Created" in result.stdout and "Owned By" in result.stdout
    _assert_no_upstream_calls(cli_gateway)


def test_cli_models_list_explicit_sort_json(cli_gateway: Gateway, tmp_path: Path) -> None:
    response: Final = cli_gateway.request("GET", "/v1/models")
    assert response.status_code == 200, response.text
    result: Final = _run_cli(cli_gateway, tmp_path, ("models", "list", "--format", "json", "--sort"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == sorted(response.json()["data"], key=lambda model: model["id"].casefold())
    _assert_no_upstream_calls(cli_gateway)


@pytest.mark.parametrize("output_format", ("json", "table"))
@pytest.mark.parametrize("no_sort", (False, True), ids=("alphabetical", "proxy-order"))
def test_cli_models_list_shell(cli_gateway: Gateway, tmp_path: Path, output_format: str, no_sort: bool) -> None:
    flag: Final = " --no-sort" if no_sort else ""
    result: Final = _run_cli(
        cli_gateway, tmp_path, (), shell_input=f"models\nlist --format {output_format}{flag}\nquit\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected: Final = MODEL_IDS if no_sort else tuple(sorted(MODEL_IDS, key=str.casefold))
    assert "Error" not in result.stdout + result.stderr, result.stdout + result.stderr
    if output_format == "table":
        assert _table_ids(result.stdout) == expected, result.stdout
    else:
        start: Final = result.stdout.index("[\n")
        listing, _end = json.JSONDecoder().raw_decode(result.stdout[start:])
        assert tuple(model["id"] for model in listing) == expected, result.stdout
    _assert_no_upstream_calls(cli_gateway)


@pytest.mark.parametrize("output_format", ("json", "table"))
def test_cli_models_list_invalid_key(cli_gateway: Gateway, tmp_path: Path, output_format: str) -> None:
    result: Final = _run_cli(
        cli_gateway,
        tmp_path,
        ("models", "list", "--format", output_format, "--sort"),
        api_key="sk-invalid-cli-audit",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "401" in result.stderr, result.stderr
    assert not any(name in result.stdout for name in MODEL_IDS), result.stdout
    _assert_no_upstream_calls(cli_gateway)


@pytest.fixture(scope="module", params=((), EDGE_IDS), ids=("empty", "ties-unicode-digits"))
def edge_cli_gateway(
    session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Iterator[tuple[Gateway, tuple[str, ...]]]:
    directory: Final = tmp_path_factory.mktemp("cli-edge-models")
    model_ids: Final = tuple(request.param)
    config: Final = _config(directory, session_gateway, model_ids)
    with owned_proxy(session_gateway, directory, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as proxy:
        yield proxy, model_ids


@pytest.mark.parametrize("output_format", ("json", "table"))
def test_cli_models_list_edges(
    edge_cli_gateway: tuple[Gateway, tuple[str, ...]], tmp_path: Path, output_format: str
) -> None:
    gateway, model_ids = edge_cli_gateway
    response: Final = gateway.request("GET", "/v1/models")
    assert response.status_code == 200, response.text
    assert tuple(model["id"] for model in response.json()["data"]) == model_ids, response.text
    expected: Final = EDGE_SORTED_IDS if model_ids else ()
    result: Final = _run_cli(gateway, tmp_path, ("models", "list", "--format", output_format, "--sort"))
    assert result.returncode == 0, result.stdout + result.stderr
    if output_format == "json":
        assert tuple(model["id"] for model in json.loads(result.stdout)) == expected, result.stdout
    else:
        assert "Available Models" in result.stdout
        assert _table_ids(result.stdout) == expected, result.stdout
    _assert_no_upstream_calls(gateway)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_cli_models_list_after_worker_restart(session_gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _config(tmp_path, session_gateway, MODEL_IDS)
    with owned_proxy_process(
        session_gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2
    ) as owned:
        before: Final = owned.worker_pids()
        assert len(before) == 2, before
        victim: Final = min(before)
        replacement: Final = owned.kill_worker(victim)
        assert replacement not in before
        assert victim not in owned.worker_pids()
        assert len(owned.worker_pids()) == 2
        result: Final = _run_cli(owned.gateway, tmp_path, ("models", "list", "--format", "json", "--sort"))
        assert result.returncode == 0, result.stdout + result.stderr
        assert tuple(model["id"] for model in json.loads(result.stdout)) == tuple(sorted(MODEL_IDS, key=str.casefold))
        original: Final = _run_cli(owned.gateway, tmp_path, ("models", "list", "--format", "json", "--no-sort"))
        assert original.returncode == 0, original.stdout + original.stderr
        assert tuple(model["id"] for model in json.loads(original.stdout)) == MODEL_IDS
        _assert_no_upstream_calls(owned.gateway)
