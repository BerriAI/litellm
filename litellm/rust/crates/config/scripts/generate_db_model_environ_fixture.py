"""Regenerate fixtures/db_model_environ_python.json: which `os.environ/` references in a stored model row Python reads.

    uv run python litellm/rust/crates/config/scripts/generate_db_model_environ_fixture.py \
        > litellm/rust/crates/config/fixtures/db_model_environ_python.json

Run it from the repository root. It upserts one LiteLLM_ProxyModelTable row through the real
ProxyConfig._add_deployment into a fresh Router and records the deployment the router then holds.
The row's values are plain text, which decrypt_value_helper passes through, since the Rust merge
takes rows its caller already decrypted.
"""

import json
import os
import sys
from types import SimpleNamespace
from typing import Final

os.environ.setdefault("LITELLM_SALT_KEY", "sk-fixture-salt")

import litellm  # noqa: E402
from litellm.proxy import proxy_server  # noqa: E402

ENVIRONMENT: Final = {"LITELLM_MASTER_KEY": "sk-master-from-env", "FIXTURE_API_KEY": "key-from-env"}

ROW: Final = {
    "model_id": "db-environ",
    "model_name": "os.environ/LITELLM_MASTER_KEY",
    "litellm_params": {
        "model": "openai/gpt-environ",
        "api_key": "os.environ/FIXTURE_API_KEY",
        "api_base": "os.environ/FIXTURE_UNSET",
        "anthropic_issuer_signing_key_ref": "os.environ/LITELLM_MASTER_KEY",
        "litellm_metadata": {"nested": "os.environ/LITELLM_MASTER_KEY"},
        "tags": ["os.environ/LITELLM_MASTER_KEY"],
    },
    "model_info": {
        "team_public_model_name": "os.environ/LITELLM_MASTER_KEY",
        "nested": {"value": "os.environ/LITELLM_MASTER_KEY"},
    },
    "blocked": False,
}


def main() -> None:
    os.environ.update(ENVIRONMENT)
    proxy_server.llm_router = litellm.Router(model_list=[])
    row = json.loads(json.dumps(ROW))
    proxy_server.ProxyConfig()._add_deployment([SimpleNamespace(**row)])
    deployment = proxy_server.llm_router.get_deployment(model_id=ROW["model_id"])
    params = deployment.litellm_params.model_dump(exclude_none=True)
    recorded = {
        "model_name": deployment.model_name,
        "litellm_params": {key: params.get(key) for key in ROW["litellm_params"]},
        "model_info": {key: deployment.model_info.get(key) for key in ROW["model_info"]},
    }
    fixture = {"environment": ENVIRONMENT, "row": ROW, "router_holds": recorded}
    json.dump(fixture, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


main()
