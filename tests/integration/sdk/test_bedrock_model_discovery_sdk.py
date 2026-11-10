from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
from integration._support.aws_control_plane import FOUNDATION_MODELS, INFERENCE_PROFILES, ControlPlane, control_plane
from integration._support.bedrock_discovery import (
    assert_listing_shape,
    assert_sigv4,
    catalog_for,
    control_plane_host,
    credential,
    listings,
    mine,
    stem,
)
from pydantic import JsonValue, TypeAdapter

REGION: Final = "us-east-1"
IDS: Final = TypeAdapter(list[str])
SCRIPT: Final = """
import json, sys
import litellm
from litellm.types.router import LiteLLM_Params
arguments = json.loads(sys.argv[1])
params = LiteLLM_Params(**arguments["litellm_params"]) if arguments["litellm_params"] is not None else None
models = litellm.get_valid_models(
    check_provider_endpoint=True, custom_llm_provider=arguments["provider"], litellm_params=params
)
print(json.dumps(sorted(models)))
"""


@pytest.fixture(scope="module")
def plane(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ControlPlane]:
    with control_plane(tmp_path_factory.mktemp("bedrock_control_plane_sdk"), (control_plane_host(REGION),)) as value:
        yield value


def _sdk_listing(
    plane: ControlPlane,
    cwd: Path,
    *,
    provider: str | None,
    litellm_params: Mapping[str, JsonValue] | None,
    process_env: Mapping[str, str] | None = None,
) -> Sequence[str]:
    environment: Final = {
        **{name: value for name, value in os.environ.items() if not name.startswith("AWS_")},
        **plane.environment(),
        "AWS_CONFIG_FILE": str(cwd / "empty-aws-config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(cwd / "empty-aws-config"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        **(process_env or {}),
    }
    (cwd / "empty-aws-config").write_text("")
    arguments: Final = json.dumps({"provider": provider, "litellm_params": litellm_params})
    completed: Final = subprocess.run(
        [sys.executable, "-P", "-c", SCRIPT, arguments], cwd=cwd, env=environment, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return IDS.validate_json(completed.stdout.strip().splitlines()[-1])


def test_sdk_get_valid_models_lists_the_account_through_the_deployment_credentials(
    plane: ControlPlane, tmp_path: Path
) -> None:
    key, secret = credential()
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with plane.answering(key, catalog.respond):
        listed: Final = _sdk_listing(
            plane,
            tmp_path,
            provider="bedrock",
            litellm_params={
                "model": "bedrock/*",
                "aws_access_key_id": key,
                "aws_secret_access_key": secret,
                "aws_region_name": REGION,
            },
        )
    assert frozenset(name for name in listed if marker in name) == catalog.vendor_ids(), listed
    requests: Final = mine(plane, key)
    assert assert_listing_shape(requests) == 1
    assert_sigv4(requests, key=key, secret=secret, region=REGION)


def test_sdk_get_valid_models_with_only_aws_environment_never_infers_bedrock(
    plane: ControlPlane, tmp_path: Path
) -> None:
    key, secret = credential()
    marker: Final = stem()
    with plane.answering(key, catalog_for(marker).respond):
        listed: Final = _sdk_listing(
            plane,
            tmp_path,
            provider=None,
            litellm_params=None,
            process_env={"AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret, "AWS_REGION_NAME": REGION},
        )
    assert not any(name.startswith("bedrock/") for name in listed), listed
    assert mine(plane, key) == ()


def test_sdk_get_valid_models_infers_bedrock_from_its_api_key_and_lists_with_the_bearer_token(
    plane: ControlPlane, tmp_path: Path
) -> None:
    token: Final = f"bearer-{stem()}"
    marker: Final = stem()
    catalog: Final = catalog_for(marker)
    with plane.answering(token, catalog.respond):
        listed: Final = _sdk_listing(
            plane,
            tmp_path,
            provider=None,
            litellm_params=None,
            process_env={"BEDROCK_API_KEY": token, "AWS_BEARER_TOKEN_BEDROCK": token, "AWS_REGION_NAME": REGION},
        )
    assert frozenset(name for name in listed if marker in name) == catalog.vendor_ids(), listed
    requests: Final = mine(plane, token)
    assert set(listings(requests)) == {FOUNDATION_MODELS, INFERENCE_PROFILES}, listings(requests)
    assert all(request.headers["authorization"] == f"Bearer {token}" for request in requests), requests
