"""Vertex credentials, project and location taken from the proxy's environment and config file.

When a deployment names none of `vertex_project`, `vertex_location` or `vertex_credentials`, the Python
getters (`VertexBase.get_vertex_ai_project` / `_location` / `_credentials`) fall back to the
`VERTEXAI_PROJECT`, `VERTEXAI_LOCATION` and `VERTEXAI_CREDENTIALS` environment variables, and a config
value written as `os.environ/<NAME>` is resolved from the environment when the config loads. These
are process-level inputs, so this module boots its own proxy with those variables pointing at key files
whose `token_uri` is a module-scoped owned peer. `credential-gaps.md` rows 5 and 6 say Rust reads
`VERTEXAI_CREDENTIALS` too and neither side reads `VERTEX_AI_CREDENTIALS`; these tests pin that.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration.providers.dec1_vertex_support import (
    Peer,
    gemini_echo,
    marker,
    models_path,
    path_of,
    peer,
    service_account_text,
)

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 180)]

_ENV_PROJECT: Final = "dec1-env-project"
_ENV_LOCATION: Final = "europe-west1"
_REF_PROJECT: Final = "dec1-ref-project"


@pytest.fixture(scope="module")
def env_peer() -> Iterator[Peer]:
    with peer(gemini_echo) as owned:
        yield owned


@pytest.fixture(scope="module")
def env_proxy(env_peer: Peer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dec1-vertex-env")
    env_key: Final = directory / "env-key.json"
    env_key.write_text(service_account_text(env_peer.url, project="dec1-env-key-project"))
    ref_key: Final = directory / "ref-key.json"
    ref_key.write_text(service_account_text(env_peer.url, project="dec1-ref-key-project"))
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(base["litellm_settings"])
    config: Final = directory / f"dec1-vertex-env-{uuid.uuid4().hex}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [
                    {
                        "model_name": "dec1-vertex-env-only",
                        "litellm_params": {"model": "vertex_ai/gemini-2.5-flash", "api_base": env_peer.url},
                    },
                    {
                        "model_name": "dec1-vertex-env-reference",
                        "litellm_params": {
                            "model": "vertex_ai/gemini-2.5-flash",
                            "api_base": env_peer.url,
                            "vertex_project": _REF_PROJECT,
                            "vertex_location": "us-east4",
                            "vertex_credentials": "os.environ/DEC1_VERTEX_KEY_PATH",
                        },
                    },
                    {
                        "model_name": "dec1-vertex-env-explicit",
                        "litellm_params": {
                            "model": "vertex_ai/gemini-2.5-flash",
                            "api_base": env_peer.url,
                            "vertex_project": "dec1-explicit-project",
                            "vertex_location": "asia-south1",
                        },
                    },
                ],
                "litellm_settings": {**settings, "cache": False},
                "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
            }
        )
    )
    overrides: Final = {
        "VERTEXAI_PROJECT": _ENV_PROJECT,
        "VERTEXAI_LOCATION": _ENV_LOCATION,
        "VERTEXAI_CREDENTIALS": str(env_key),
        "DEC1_VERTEX_KEY_PATH": str(ref_key),
    }
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, directory, overrides, config=config, workers=1) as proxy,
    ):
        yield proxy


def _ask(proxy: OwnedProxy, model: str) -> int:
    return proxy.gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": f"env {marker()}"}]}
    ).status_code


def _last_model_path(owned: Peer) -> str:
    return path_of(owned.model_calls()[-1])


def test_a_deployment_without_vertex_settings_uses_the_vertexai_environment(
    env_proxy: OwnedProxy, env_peer: Peer
) -> None:
    """A config deployment with no Vertex settings posts to `projects/dec1-env-project/locations/
    europe-west1` (from `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`) with a token minted from the key file
    `VERTEXAI_CREDENTIALS` names. The env project wins over the key's own `project_id`."""
    assert _ask(env_proxy, "dec1-vertex-env-only") == 200
    call: Final = env_peer.model_calls()[-1]
    assert path_of(call) == models_path(project=_ENV_PROJECT, location=_ENV_LOCATION) + ":generateContent"
    assert call.headers["authorization"].removeprefix("Bearer ") in env_peer.plan.issued


def test_an_os_environ_reference_in_config_names_the_key_file(env_proxy: OwnedProxy, env_peer: Peer) -> None:
    """`vertex_credentials: os.environ/DEC1_VERTEX_KEY_PATH` resolves to the path in that variable and the
    key file is loaded from disk; the call goes to the deployment's own project and location
    (`dec1-ref-project`, `us-east4`) with a minted bearer token."""
    assert _ask(env_proxy, "dec1-vertex-env-reference") == 200
    call: Final = env_peer.model_calls()[-1]
    assert path_of(call) == models_path(project=_REF_PROJECT, location="us-east4") + ":generateContent"
    assert call.headers["authorization"].removeprefix("Bearer ") in env_peer.plan.issued


def test_deployment_project_and_location_win_over_the_environment(env_proxy: OwnedProxy, env_peer: Peer) -> None:
    """A deployment that sets `vertex_project` and `vertex_location` uses them even though
    `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION` are set; only the credentials come from the environment."""
    assert _ask(env_proxy, "dec1-vertex-env-explicit") == 200
    assert (
        _last_model_path(env_peer)
        == models_path(project="dec1-explicit-project", location="asia-south1") + ":generateContent"
    )
