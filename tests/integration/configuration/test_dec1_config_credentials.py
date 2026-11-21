"""Named credentials declared in the config file (`credential_list`) and deployments that use them.

The contract: each `credential_list` entry has a `credential_name`, a `credential_values` mapping (for
example `api_key` and `api_base`) and free-form `credential_info`. A `model_list` entry may say
`litellm_credential_name: <name>` instead of carrying its own key and base; calls to that deployment then
use the named credential's values. `GET /credentials` lists file credentials with `source: config` and
masked secrets, and the credential API refuses to create, change or delete a name the file defines.

The Rust loader must resolve the name to the values at load time and keep `os.environ/` references
inside `credential_values` working, since the whole point of the feature is one place for a secret.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment, object_value
from tests.integration._support.process import OwnedProxy
from tests.integration._support.wire import Wire
from tests.integration.configuration.dec1_config_support import (
    ITEMS,
    booted,
    chat,
    model_entries,
    scripted_upstream,
    sent,
)

_NAME: Final = "dec1-shared-credential"
_ENV_NAME: Final = "dec1-env-credential"
_SECRET: Final = "sk-dec1-credential-secret-wxyz"
_ENV_SECRET: Final = "sk-dec1-credential-from-env-qrst"
_MODEL: Final = "dec1-credential-model"
_ENV_MODEL: Final = "dec1-env-credential-model"


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    upstream: Wire

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway

    def credentials(self) -> list[dict[str, JsonValue]]:
        return ITEMS.validate_python(self.gateway.get("/credentials")["credentials"])


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-credentials")
    with scripted_upstream() as upstream, gateway_from_environment() as shared:
        base: Final = f"{upstream.url}/v1"
        config: Final = {
            "credential_list": [
                {
                    "credential_name": _NAME,
                    "credential_values": {"api_key": _SECRET, "api_base": base},
                    "credential_info": {"description": "dec1 shared provider account", "custom_llm_provider": "openai"},
                },
                {
                    "credential_name": _ENV_NAME,
                    "credential_values": {"api_key": "os.environ/DEC1_CREDENTIAL_KEY", "api_base": base},
                    "credential_info": {},
                },
            ],
            "model_list": [
                {
                    "model_name": _MODEL,
                    "litellm_params": {"model": "openai/gpt-4o-mini", "litellm_credential_name": _NAME},
                },
                {
                    "model_name": _ENV_MODEL,
                    "litellm_params": {"model": "openai/gpt-4o-mini", "litellm_credential_name": _ENV_NAME},
                },
            ],
        }
        with booted(shared, directory, config, environment={"DEC1_CREDENTIAL_KEY": _ENV_SECRET}) as proxy:
            yield _Rig(proxy, upstream)


def test_a_deployment_naming_a_credential_calls_the_provider_with_that_credentials_key_and_base(rig: _Rig) -> None:
    """The deployment has no `api_key` or `api_base`, only `litellm_credential_name`. A chat completion
    must reach the credential's `api_base` as `POST /v1/chat/completions` with `Authorization: Bearer
    <the credential's api_key>`."""
    sent(rig.upstream)
    response: Final = chat(rig.gateway, _MODEL)
    assert response.status_code == 200, response.text
    requests: Final = sent(rig.upstream)
    assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]
    assert requests[0].headers["authorization"] == f"Bearer {_SECRET}"


def test_an_environment_reference_inside_credential_values_is_resolved(rig: _Rig) -> None:
    """The credential's `api_key` is `os.environ/DEC1_CREDENTIAL_KEY`. A chat completion through a
    deployment naming that credential must reach the provider with the variable's value as the bearer
    token, not the reference text."""
    sent(rig.upstream)
    response: Final = chat(rig.gateway, _ENV_MODEL)
    assert response.status_code == 200, response.text
    assert [request.headers["authorization"] for request in sent(rig.upstream)] == [f"Bearer {_ENV_SECRET}"]


def test_credentials_endpoint_lists_file_credentials_with_source_config(rig: _Rig) -> None:
    """`GET /credentials` must return `success: true` and one entry per `credential_list` item, each with
    `source: config` and a null `display_name`."""
    listed: Final = {str(item["credential_name"]): item for item in rig.credentials()}
    assert {_NAME, _ENV_NAME} <= set(listed)
    assert (listed[_NAME]["source"], listed[_NAME]["display_name"]) == ("config", None)
    assert rig.gateway.get("/credentials")["success"] is True


def test_credentials_endpoint_masks_the_secret_and_never_returns_it(rig: _Rig) -> None:
    """`GET /credentials` must return the file credential's `api_key` masked to its first two and last
    two characters around four asterisks (`sk****yz`), and the full secret must appear nowhere in the
    response body."""
    response: Final = rig.gateway.request("GET", "/credentials")
    assert response.status_code == 200, response.text
    entry: Final = next(item for item in rig.credentials() if item["credential_name"] == _NAME)
    assert object_value(entry["credential_values"])["api_key"] == "sk****yz"
    assert _SECRET not in response.text and _ENV_SECRET not in response.text


def test_credentials_endpoint_returns_credential_info_unchanged(rig: _Rig) -> None:
    """`GET /credentials` must return the entry's `credential_info` mapping exactly as written in the file."""
    entry: Final = next(item for item in rig.credentials() if item["credential_name"] == _NAME)
    assert entry["credential_info"] == {
        "description": "dec1 shared provider account",
        "custom_llm_provider": "openai",
    }


def test_a_file_credential_can_be_fetched_by_name(rig: _Rig) -> None:
    """`GET /credentials/by_name/<name>` for a file credential must return 200 with that
    `credential_name`, `source: config` and the masked key."""
    body: Final = rig.gateway.get(f"/credentials/by_name/{_NAME}")
    assert (body["credential_name"], body["source"]) == (_NAME, "config")
    assert object_value(body["credential_values"])["api_key"] == "sk****yz"


def test_fetching_an_unknown_credential_name_is_a_404(rig: _Rig) -> None:
    """`GET /credentials/by_name/dec1-no-such-credential` must return 404 with `Credential not found. Got
    credential name: dec1-no-such-credential`."""
    response: Final = rig.gateway.request("GET", "/credentials/by_name/dec1-no-such-credential")
    assert response.status_code == 404, response.text
    assert "Credential not found. Got credential name: dec1-no-such-credential" in response.text


def test_creating_a_credential_with_a_file_defined_name_is_a_409(rig: _Rig) -> None:
    """`POST /credentials` reusing the name of a file credential must return 409 with `Credential
    '<name>' is defined in config. Edit config.yaml to change it.`, and the file's values keep serving."""
    response: Final = rig.gateway.request(
        "POST",
        "/credentials",
        {"credential_name": _NAME, "credential_values": {"api_key": "sk-dec1-replacement"}, "credential_info": {}},
    )
    assert response.status_code == 409, response.text
    assert f"Credential '{_NAME}' is defined in config. Edit config.yaml to change it." in response.text
    sent(rig.upstream)
    assert chat(rig.gateway, _MODEL).status_code == 200
    assert sent(rig.upstream)[0].headers["authorization"] == f"Bearer {_SECRET}"


def test_deleting_a_file_defined_credential_is_a_405(rig: _Rig) -> None:
    """`DELETE /credentials/<name>` for a file credential must return 405 with `Credential '<name>' is
    defined in config and cannot be edited from the API or UI.` and an `Allow: GET` header, and the
    credential must still be listed."""
    response: Final = rig.gateway.request("DELETE", f"/credentials/{_NAME}")
    assert response.status_code == 405, response.text
    assert f"Credential '{_NAME}' is defined in config and cannot be edited from the API or UI." in response.text
    assert response.headers["allow"] == "GET"
    assert _NAME in {item["credential_name"] for item in rig.credentials()}


def test_patching_a_file_defined_credential_is_a_405(rig: _Rig) -> None:
    """`PATCH /credentials/<name>` for a file credential must return 405 with the same `is defined in
    config and cannot be edited from the API or UI.` message, and later calls still use the file's key."""
    response: Final = rig.gateway.request(
        "PATCH",
        f"/credentials/{_NAME}",
        {"credential_values": {"api_key": "sk-dec1-replacement"}, "credential_info": {}},
    )
    assert response.status_code == 405, response.text
    assert f"Credential '{_NAME}' is defined in config and cannot be edited from the API or UI." in response.text
    sent(rig.upstream)
    assert chat(rig.gateway, _MODEL).status_code == 200
    assert sent(rig.upstream)[0].headers["authorization"] == f"Bearer {_SECRET}"


def test_model_info_shows_the_credential_name_and_not_the_credentials_secret(rig: _Rig) -> None:
    """`/model/info` for a deployment that names a credential must show `litellm_credential_name` in
    clear (it is a reference, not a secret) and must not contain the credential's `api_key` anywhere."""
    response: Final = rig.gateway.request("GET", "/model/info")
    params: Final = object_value(model_entries(rig.gateway, _MODEL)[0]["litellm_params"])
    assert params["litellm_credential_name"] == _NAME
    assert _SECRET not in response.text


def test_a_database_credential_can_be_added_next_to_file_credentials(rig: _Rig) -> None:
    """`POST /credentials` with a new name must succeed, and `GET /credentials` must then list it with
    `source: db` next to the file credentials, which keep `source: config`.
    UNVERIFIED: the literal `db` for API-created credentials (the `CredentialItem.source` default)."""
    name: Final = "dec1-database-credential"
    created: Final = rig.gateway.request(
        "POST",
        "/credentials",
        {"credential_name": name, "credential_values": {"api_key": "sk-dec1-database"}, "credential_info": {}},
    )
    assert created.status_code == 200, created.text
    try:
        sources: Final = {str(item["credential_name"]): item["source"] for item in rig.credentials()}
    finally:
        rig.gateway.request("DELETE", f"/credentials/{name}")
    assert (sources[name], sources[_NAME]) == ("db", "config")


def test_a_deployment_naming_an_undefined_credential_does_not_borrow_another_credential(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A deployment names `litellm_credential_name: dec1-undefined` and no such credential exists, while
    another credential is defined. A call to that deployment must not reach the defined credential's
    `api_base`: the scripted provider receives nothing for it.
    UNVERIFIED: assumes the proxy boots and fails the call against the deployment's own unreachable
    `api_base`; if it refuses to boot instead, this test should become a `refused` case naming the
    credential."""
    with scripted_upstream() as upstream:
        config: Final = {
            "credential_list": [
                {
                    "credential_name": _NAME,
                    "credential_values": {"api_key": _SECRET, "api_base": f"{upstream.url}/v1"},
                    "credential_info": {},
                }
            ],
            "model_list": [
                {
                    "model_name": _MODEL,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": "http://127.0.0.1:9/v1",
                        "litellm_credential_name": "dec1-undefined",
                    },
                }
            ],
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            requests: Final = sent(upstream)
    assert response.status_code != 200, response.text
    assert requests == ()
