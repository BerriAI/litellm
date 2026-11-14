import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy

MODEL_IDS: Final = ("gpt-5.4-mini", "Claude-sonnet", "azure-gpt", "bedrock-llama", "Anthropic-haiku")


@pytest.fixture(scope="module")
def cli_gateway(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("cli-models")
    config: Final = directory / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {
                            "model": "openai/integration-model",
                            "api_base": f"{session_gateway.upstream_url}/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                    for name in MODEL_IDS
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    with owned_proxy(session_gateway, directory, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as proxy:
        yield proxy


@pytest.mark.parametrize("no_sort", (False, True), ids=("alphabetical", "proxy-order"))
def test_cli_models_list_order(cli_gateway: Gateway, no_sort: bool, tmp_path: Path) -> None:
    response: Final = cli_gateway.request("GET", "/v1/models")
    assert response.status_code == 200, response.text
    proxy_ids: Final = tuple(model["id"] for model in response.json()["data"])
    assert set(proxy_ids) == set(MODEL_IDS), response.text
    assert proxy_ids != tuple(sorted(MODEL_IDS, key=str.casefold)), response.text
    root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    result: Final = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "from litellm.proxy.client.cli import cli; cli()",
            "--base-url",
            str(cli_gateway.client.base_url),
            "--api-key",
            cli_gateway.key,
            "models",
            "list",
            "--format",
            "json",
            *(("--no-sort",) if no_sort else ()),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "HOME": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected: Final = proxy_ids if no_sort else tuple(sorted(MODEL_IDS, key=str.casefold))
    assert tuple(model["id"] for model in json.loads(result.stdout)) == expected, result.stdout
