"""Where the proxy's master key comes from and when a boot is refused over it.

The contract: `general_settings.master_key` in the config file wins over the `LITELLM_MASTER_KEY`
environment variable; the file value may itself be an `os.environ/NAME` reference. The key that results
is the only admin bearer token. A boot with no key, a blank key, or the publicly documented example key
is refused before the server listens, with a message that names the source of the bad key and the fix,
unless the operator sets the explicit local-development override.

Assumption: the harness always sets `LITELLM_MASTER_KEY` (to its own key) in the environment of an owned
proxy and offers no way to remove it, so "no key anywhere" is reached through a file value that resolves
to null, which shadows the environment value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value
from tests.integration.configuration.dec1_config_support import booted, refused, with_key, write_config

_FILE_KEY: Final = "sk-dec1-master-from-file-7f3a"
_OTHER_KEY: Final = "sk-dec1-master-from-other-variable-91c2"
# The key every public example uses; the proxy recognises it by digest and refuses to run with it.
_PUBLIC_EXAMPLE_KEY: Final = "sk-" + "1234"
_OVERRIDE_ENV: Final = "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY"


def test_the_environment_master_key_is_the_admin_key_when_the_file_sets_none(gateway: Gateway, tmp_path: Path) -> None:
    """The config has no `general_settings.master_key` and `LITELLM_MASTER_KEY` is set. A call to the
    admin-only `POST /key/generate` with that key must succeed with 200 and return a new `sk-` key."""
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        created: Final = owned.gateway.post("/key/generate", {})
    assert str(created["key"]).startswith("sk-")


def test_a_bearer_token_that_is_not_the_master_key_is_rejected_with_401(gateway: Gateway, tmp_path: Path) -> None:
    """A caller presents an `sk-` token that is neither the master key nor a stored virtual key. The proxy
    must answer 401 with an OpenAI-shaped error whose `type` is `token_not_found_in_db`, `param` is `key`,
    `code` is `401` and message is the generic "Authentication Error, Invalid API key provided."
    (litellm/proxy/auth/auth_checks.py raises `ProxyErrorTypes.token_not_found_in_db` for an unknown key)."""
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        response: Final = owned.gateway.request("GET", "/v1/models", key="sk-dec1-not-a-real-key")
    assert response.status_code == 401, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert (error["type"], error["param"], str(error["code"])) == ("token_not_found_in_db", "key", "401"), error
    assert error["message"] == "Authentication Error, Invalid API key provided.", error


def test_a_master_key_in_the_file_wins_over_the_environment_variable(gateway: Gateway, tmp_path: Path) -> None:
    """The file sets `general_settings.master_key` to one value while `LITELLM_MASTER_KEY` holds another.
    `GET /v1/models` must return 200 for the file's key and 401 for the environment's key: the file is
    the single source once it names a key."""
    with booted(gateway, tmp_path, {"general_settings": {"master_key": _FILE_KEY}}) as owned:
        with_file_key: Final = with_key(owned.gateway, _FILE_KEY).request("GET", "/v1/models")
        with_env_key: Final = owned.gateway.request("GET", "/v1/models")
    assert with_file_key.status_code == 200, with_file_key.text
    assert with_env_key.status_code == 401, with_env_key.text


def test_a_file_master_key_that_references_the_standard_variable_accepts_that_variables_value(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file says `master_key: os.environ/LITELLM_MASTER_KEY`, the form the docs recommend. The key in
    that variable must be accepted on `GET /v1/models` with 200."""
    config: Final = {"general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"}}
    with booted(gateway, tmp_path, config) as owned:
        assert owned.gateway.request("GET", "/v1/models").status_code == 200


def test_a_file_master_key_that_references_another_variable_uses_that_variable_instead(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file says `master_key: os.environ/DEC1_MASTER_KEY` and the process has both that variable and
    `LITELLM_MASTER_KEY`. The referenced variable's value must be accepted (200) and the
    `LITELLM_MASTER_KEY` value rejected (401)."""
    config: Final = {"general_settings": {"master_key": "os.environ/DEC1_MASTER_KEY"}}
    with booted(gateway, tmp_path, config, environment={"DEC1_MASTER_KEY": _OTHER_KEY}) as owned:
        referenced: Final = with_key(owned.gateway, _OTHER_KEY).request("GET", "/v1/models")
        standard: Final = owned.gateway.request("GET", "/v1/models")
    assert referenced.status_code == 200, referenced.text
    assert standard.status_code == 401, standard.text


def test_a_null_master_key_in_the_file_refuses_to_boot_even_though_the_environment_has_one(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file has `master_key:` with nothing after it while `LITELLM_MASTER_KEY` is set. The blank file
    entry shadows the environment, so the proxy must exit non-zero and log `LiteLLM proxy refused to
    start: no master key is set, so every request would be accepted without authentication.` followed by
    a line saying `general_settings.master_key in <config path> is blank, or points at an environment
    variable that is not set.`"""
    path: Final = write_config(tmp_path, {"general_settings": {"master_key": None}})
    log: Final = refused(gateway, tmp_path, path)
    assert (
        "LiteLLM proxy refused to start: no master key is set, so every request would be accepted without "
        "authentication."
    ) in log
    assert (
        f"general_settings.master_key in {path} is blank, or points at an environment variable that is not set." in log
    )


def test_the_public_example_key_in_the_file_refuses_to_boot_and_names_the_file(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file sets the master key to the example value used throughout public docs. The proxy must exit
    non-zero and log `LiteLLM proxy refused to start: the master key is a publicly known default.`, then
    `It comes from general_settings.master_key in <config path>.`, then the fix telling the operator to
    read the key from the environment (`master_key: os.environ/LITELLM_MASTER_KEY`)."""
    path: Final = write_config(tmp_path, {"general_settings": {"master_key": _PUBLIC_EXAMPLE_KEY}})
    log: Final = refused(gateway, tmp_path, path)
    assert "LiteLLM proxy refused to start: the master key is a publicly known default." in log
    assert f"It comes from general_settings.master_key in {path}." in log
    assert "master_key: os.environ/LITELLM_MASTER_KEY" in log


def test_the_public_example_key_in_the_environment_refuses_to_boot_and_names_the_variable(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`LITELLM_MASTER_KEY` holds the public example key and the file sets no key. The proxy must exit
    non-zero and log that the key is a publicly known default and that `It comes from the
    LITELLM_MASTER_KEY environment variable.`"""
    log: Final = refused(gateway, tmp_path, {"model_list": []}, environment={"LITELLM_MASTER_KEY": _PUBLIC_EXAMPLE_KEY})
    assert "LiteLLM proxy refused to start: the master key is a publicly known default." in log
    assert "It comes from the LITELLM_MASTER_KEY environment variable." in log


def test_a_file_that_only_relays_a_bad_environment_key_blames_the_environment(gateway: Gateway, tmp_path: Path) -> None:
    """The file says `master_key: os.environ/LITELLM_MASTER_KEY` and that variable holds the public
    example key. The refusal must say the key `comes from the LITELLM_MASTER_KEY environment variable`,
    not from the config file, because the file only passes the environment value along."""
    log: Final = refused(
        gateway,
        tmp_path,
        {"general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"}},
        environment={"LITELLM_MASTER_KEY": _PUBLIC_EXAMPLE_KEY},
    )
    assert "It comes from the LITELLM_MASTER_KEY environment variable." in log


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_master_key_in_the_environment_refuses_to_boot(gateway: Gateway, tmp_path: Path, blank: str) -> None:
    """`LITELLM_MASTER_KEY` is set to an empty or whitespace-only string. The proxy must exit non-zero and
    log `LiteLLM proxy refused to start: the master key is empty.`
    UNVERIFIED for the empty string: whether an empty variable reads as empty or as unset; if it reads as
    unset the headline is the `no master key is set` one instead (see `master_key_boot_check.py`)."""
    log: Final = refused(gateway, tmp_path, {"model_list": []}, environment={"LITELLM_MASTER_KEY": blank})
    assert "LiteLLM proxy refused to start: the master key is empty." in log


def test_the_refusal_tells_the_operator_how_to_override_it_for_local_development(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Any master key refusal must end with the local-development hint naming both override switches:
    the `LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY=true` variable and
    `general_settings.dangerously_permit_weak_or_unset_master_key: true`."""
    log: Final = refused(gateway, tmp_path, {"general_settings": {"master_key": _PUBLIC_EXAMPLE_KEY}})
    assert f"Local development only: set {_OVERRIDE_ENV}=true, or" in log
    assert "general_settings.dangerously_permit_weak_or_unset_master_key: true, to start anyway." in log


def test_the_override_variable_lets_the_public_example_key_boot_with_a_warning(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file sets the public example key and the process has the override variable set to `true`. The
    proxy must boot, accept that key on `GET /v1/models`, and log `dangerously_permit_weak_or_unset_master_key
    is on, so the proxy is starting with a publicly known master key. Never run this outside local
    development.`"""
    config: Final = {"general_settings": {"master_key": _PUBLIC_EXAMPLE_KEY}}
    with booted(gateway, tmp_path, config, environment={_OVERRIDE_ENV: "true"}) as owned:
        response: Final = with_key(owned.gateway, _PUBLIC_EXAMPLE_KEY).request("GET", "/v1/models")
        log: Final = owned.log.read_text()
    assert response.status_code == 200, response.text
    assert (
        "dangerously_permit_weak_or_unset_master_key is on, so the proxy is starting with a publicly known "
        "master key. Never run this outside local development."
    ) in log


def test_the_override_setting_in_the_file_lets_the_public_example_key_boot(gateway: Gateway, tmp_path: Path) -> None:
    """The same override expressed as `general_settings.dangerously_permit_weak_or_unset_master_key: true`
    in the file, with no override variable. The proxy must boot and accept the public example key."""
    config: Final = {
        "general_settings": {"master_key": _PUBLIC_EXAMPLE_KEY, "dangerously_permit_weak_or_unset_master_key": True}
    }
    with booted(gateway, tmp_path, config, remove_environment=(_OVERRIDE_ENV,)) as owned:
        response: Final = with_key(owned.gateway, _PUBLIC_EXAMPLE_KEY).request("GET", "/v1/models")
    assert response.status_code == 200, response.text


def test_the_override_setting_must_be_the_boolean_true(gateway: Gateway, tmp_path: Path) -> None:
    """`dangerously_permit_weak_or_unset_master_key: "yes"` is a string, not the boolean true. The proxy
    must still refuse to boot with the public example key: only an exact `true` unlocks the override."""
    config: Final = {
        "general_settings": {"master_key": _PUBLIC_EXAMPLE_KEY, "dangerously_permit_weak_or_unset_master_key": "yes"}
    }
    log: Final = refused(gateway, tmp_path, config)
    assert "LiteLLM proxy refused to start: the master key is a publicly known default." in log


def test_the_master_key_is_never_stored_as_a_virtual_key_row_by_booting(gateway: Gateway, tmp_path: Path) -> None:
    """Booting with a file master key and making an authenticated call must not leave the plaintext key in
    `LiteLLM_VerificationToken`: the table may hold the key's hash, never the key text."""
    with booted(gateway, tmp_path, {"general_settings": {"master_key": _FILE_KEY}}) as owned:
        assert with_key(owned.gateway, _FILE_KEY).request("GET", "/v1/models").status_code == 200
        rows: Final = owned.gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (_FILE_KEY,))
    assert rows == []
