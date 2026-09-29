"""Regression: wildcard expansion of a partial provider prefix like
`databricks/system.ai.*` must produce `databricks/system.ai.<endpoint>` ids,
not `databricks/system.ai.databricks/databricks-<name>`.

On main, get_known_models_from_wildcard prepends the literal prefix before the
`*` ("system.ai.") to cost-map keys that already carry the `databricks/`
provider prefix, so /v1/models and /model/info list names that Databricks
rejects with "Invalid Unity Catalog name". See LIT-8910.

The test boots its own proxy (the shared stack's model_list is fixed), with two
wildcard deployments: `databricks/system.ai.*` under test and `databricks/*` as
a control whose expansion must stay `databricks/databricks-*`.
"""

import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Final, cast

import pytest
from e2e_http import URL, AuthHeaders, NoBody, get, probe, unwrap
from models import ModelsListParams, ModelsListResponse
from pydantic import BaseModel, ConfigDict

REPO_ROOT: Final = Path(__file__).resolve().parents[2]

MASTER_KEY: Final = "sk-1234"
AUTH: Final = AuthHeaders(authorization=f"Bearer {MASTER_KEY}")

CONFIG_YAML: Final = """\
model_list:
  - model_name: databricks/system.ai.*
    litellm_params:
      model: databricks/system.ai.*
      api_key: fake-databricks-token
      api_base: https://example.invalid
  - model_name: databricks/*
    litellm_params:
      model: databricks/*
      api_key: fake-databricks-token
      api_base: https://example.invalid
general_settings:
  master_key: sk-1234
"""


class _ModelInfoParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")
    model: str = ""


class _ModelInfoRow(BaseModel):
    model_name: str
    litellm_params: _ModelInfoParams = _ModelInfoParams()


class _ModelInfoResponse(BaseModel):
    data: list[_ModelInfoRow] = []


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast("tuple[str, int]", sock.getsockname())[1]


@pytest.fixture(scope="module")
def wildcard_proxy(tmp_path_factory: pytest.TempPathFactory) -> Generator[str]:
    config_path = tmp_path_factory.mktemp("wildcard") / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-P",
            str(REPO_ROOT / "litellm/proxy/proxy_cli.py"),
            "--config",
            str(config_path),
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(Path.home()),
            "PYTHONPATH": str(REPO_ROOT),
            "LITELLM_LOCAL_MODEL_COST_MAP": "true",
            "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY": "true",
            "LITELLM_MASTER_KEY": MASTER_KEY,
        },
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 180
        while True:
            alive = probe(URL(f"{base}/health/liveliness"), headers=NoBody(), params=NoBody(), timeout=5.0)
            if alive.healthy:
                break
            if proc.poll() is not None:
                pytest.fail(f"proxy exited during boot with {proc.returncode}")
            assert time.time() < deadline, "proxy did not answer /health/liveliness within 180s"
            time.sleep(2)
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def _list_model_ids(base: str) -> list[str]:
    listed = get(
        URL(f"{base}/v1/models"),
        headers=AUTH,
        params=ModelsListParams(return_wildcard_routes=False),
        response_type=ModelsListResponse,
    )
    return [entry.id for entry in unwrap(listed).data]


def test_partial_prefix_wildcard_expands_to_unity_catalog_names(wildcard_proxy: str) -> None:
    ids = _list_model_ids(wildcard_proxy)

    system_ai_ids = [model_id for model_id in ids if model_id.startswith("databricks/system.ai.")]
    assert system_ai_ids, f"no databricks/system.ai.* expansion in {ids}"

    malformed = [model_id for model_id in system_ai_ids if "/" in model_id.removeprefix("databricks/system.ai.")]
    assert not malformed, f"expanded ids embed the provider-prefixed cost-map key after 'system.ai.': {malformed}"


def test_provider_wildcard_still_expands_to_cost_map_names(wildcard_proxy: str) -> None:
    ids = _list_model_ids(wildcard_proxy)

    databricks_ids = [model_id for model_id in ids if model_id.startswith("databricks/databricks-")]
    assert databricks_ids, f"databricks/* did not expand to cost-map names: {ids}"


def test_model_info_keeps_provider_model_for_expanded_deployments(wildcard_proxy: str) -> None:
    info = unwrap(
        get(
            URL(f"{wildcard_proxy}/model/info"),
            headers=AUTH,
            params=NoBody(),
            response_type=_ModelInfoResponse,
        )
    )

    expanded = [row for row in info.data if row.model_name.startswith("databricks/system.ai.")]
    assert expanded, "model/info returned no expanded rows for databricks/system.ai.*"
    bad = [row.model_name for row in expanded if "system.ai.databricks/" in row.litellm_params.model]
    assert not bad, f"litellm_params.model carries the corrupted expanded name: {bad}"
