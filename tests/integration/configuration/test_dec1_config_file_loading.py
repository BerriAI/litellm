"""How the proxy finds, reads and merges its config file.

The contract: the file named by `--config` (or by `CONFIG_FILE_PATH`, which wins inside the server
process) is parsed as YAML. An `include:` list pulls in other files relative to the file that declares
them: list values such as `model_list` are appended, every other top-level value from the included file
replaces the declaring file's value whole, a file is read once however many times it is named, and a
cycle is harmless. A file that is missing, empty, not YAML, or whose include is malformed stops the boot
with a message naming the problem. Keys the proxy does not know are ignored, some with a warning.

A Rust loader must keep the merge rules, the precedence between the two path sources and the exact
refusal messages, because operators grep their boot logs for them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import OwnedProxy
from tests.integration.configuration.dec1_config_support import (
    booted,
    deployment,
    model_entries,
    refused,
    served_model_names,
    write_config,
)

_UNREACHABLE: Final = "http://127.0.0.1:9/v1"


def _models(*names: str) -> list[dict[str, object]]:
    return [dict(deployment(name, _UNREACHABLE)) for name in names]


@pytest.fixture(scope="module")
def included(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    """One proxy whose root config includes a sibling model file twice and a nested settings file.

    Layout: `root.yaml` includes `models.yaml`, `nested/settings.yaml` and `models.yaml` again.
    `nested/settings.yaml` includes `deeper.yaml` (which sits in `nested/`), and `deeper.yaml` includes
    `settings.yaml` back (a cycle). The cycle avoids the root file because the harness may boot the proxy
    on a renamed private copy of it.
    """
    directory: Final = tmp_path_factory.mktemp("dec1-config-include")
    nested: Final = directory / "nested"
    nested.mkdir()
    (directory / "models.yaml").write_text(yaml.safe_dump({"model_list": _models("dec1-included-model")}))
    (nested / "deeper.yaml").write_text(
        yaml.safe_dump({"include": ["settings.yaml"], "model_list": _models("dec1-deeper-model")})
    )
    (nested / "settings.yaml").write_text(
        yaml.safe_dump(
            {
                "include": ["deeper.yaml"],
                "general_settings": {"enable_drain_endpoint": False},
                "litellm_settings": {"request_timeout": 222},
            }
        )
    )
    root: Final = write_config(
        directory,
        {
            "include": ["models.yaml", "nested/settings.yaml", "models.yaml"],
            "model_list": _models("dec1-root-model"),
            "general_settings": {"allow_public_health_readiness_details": True},
            "litellm_settings": {"request_timeout": 111},
        },
        "root.yaml",
    )
    with gateway_from_environment() as shared, booted(shared, directory, root) as owned:
        yield owned


def test_include_appends_the_included_model_list_to_the_declaring_files_models(included: OwnedProxy) -> None:
    """The root config declares one model and includes a file declaring another. `GET /v1/models` must
    list both names: an included `model_list` extends the declaring list, it does not replace it."""
    names: Final = served_model_names(included.gateway)
    assert {"dec1-root-model", "dec1-included-model"} <= names, sorted(names)


def test_include_replaces_a_scalar_setting_with_the_included_files_value(included: OwnedProxy) -> None:
    """The root sets `litellm_settings.request_timeout: 111` and the included file sets it to 222.
    `GET /active/callbacks` must report `litellm.request_timeout` as 222: for non-list values the
    included file wins over the file that includes it."""
    assert included.gateway.get("/active/callbacks")["litellm.request_timeout"] == 222


def test_include_replaces_a_whole_settings_section_rather_than_merging_its_keys(included: OwnedProxy) -> None:
    """The root sets `general_settings.allow_public_health_readiness_details: true`; the included file
    has its own `general_settings` without that key. The unauthenticated `GET /health/readiness` must
    return only `status` and `db`, because the included section replaced the root's section whole and
    the root's key is gone. A loader that deep-merges sections would leak the detailed payload here."""
    response: Final = included.gateway.client.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "healthy", "db": "connected"}


def test_a_nested_include_is_resolved_next_to_the_file_that_declares_it(included: OwnedProxy) -> None:
    """`nested/settings.yaml` includes `deeper.yaml`, which exists only inside `nested/`. The model that
    file declares must be listed by `GET /v1/models`, so the path was resolved relative to the declaring
    file and not relative to the root config."""
    assert "dec1-deeper-model" in served_model_names(included.gateway)


def test_an_include_cycle_boots_and_loads_each_file_once(included: OwnedProxy) -> None:
    """`nested/settings.yaml` includes `deeper.yaml`, which includes `settings.yaml` back. The proxy must
    still boot, and `GET /model/info` must hold exactly one deployment named `dec1-deeper-model`: a config
    already loaded is neither read nor merged a second time."""
    assert len(model_entries(included.gateway, "dec1-deeper-model")) == 1


def test_a_file_included_twice_contributes_its_models_once(included: OwnedProxy) -> None:
    """The root names `models.yaml` twice in `include`. `GET /model/info` must hold exactly one
    deployment named `dec1-included-model`, not two load-balanced copies."""
    assert len(model_entries(included.gateway, "dec1-included-model")) == 1


def test_an_include_naming_a_missing_file_refuses_to_boot_and_names_the_path(gateway: Gateway, tmp_path: Path) -> None:
    """The config includes `absent.yaml`, which does not exist. The proxy must exit non-zero before
    serving, and its log must contain `Included file not found: <absolute path of absent.yaml>`."""
    log: Final = refused(gateway, tmp_path, {"include": ["absent.yaml"], "model_list": []})
    assert f"Included file not found: {tmp_path / 'absent.yaml'}" in log


def test_an_include_that_is_not_a_list_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`include: models.yaml` (a string, not a list). The proxy must exit non-zero and log
    `'include' must be a list of file paths` instead of guessing what was meant."""
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"model_list": _models("dec1-never-loaded")}))
    log: Final = refused(gateway, tmp_path, {"include": "models.yaml"})
    assert "'include' must be a list of file paths" in log


def test_an_include_list_holding_a_non_string_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`include: [{file: models.yaml}]`. Every entry must be a path string, so the proxy must exit
    non-zero and log `'include' must be a list of file paths`."""
    log: Final = refused(gateway, tmp_path, {"include": [{"file": "models.yaml"}]})
    assert "'include' must be a list of file paths" in log


def test_an_included_file_that_is_a_yaml_list_refuses_to_boot_and_names_the_file(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The included file parses as a YAML list instead of a mapping. The proxy must exit non-zero and
    log `Included config file is not a YAML mapping: <absolute path>`."""
    (tmp_path / "list.yaml").write_text("- one\n- two\n")
    log: Final = refused(gateway, tmp_path, {"include": ["list.yaml"]})
    assert f"Included config file is not a YAML mapping: {tmp_path / 'list.yaml'}" in log


def test_an_empty_config_file_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """The config file exists but holds nothing (YAML parses it to null). The proxy must exit non-zero
    and log `Config cannot be None or Empty.` rather than start with no settings."""
    empty: Final = tmp_path / "empty.yaml"
    empty.write_text("")
    assert "Config cannot be None or Empty." in refused(gateway, tmp_path, empty)


def test_a_config_path_that_does_not_exist_refuses_to_boot_and_names_the_path(gateway: Gateway, tmp_path: Path) -> None:
    """`--config` names a file that is not on disk. The proxy must exit non-zero and log
    `Config file not found: <path>`."""
    missing: Final = tmp_path / "not-there.yaml"
    assert f"Config file not found: {missing}" in refused(gateway, tmp_path, missing)


def test_a_config_file_that_is_not_valid_yaml_refuses_to_boot_with_a_yaml_error(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file has an unclosed flow sequence. The proxy must exit non-zero and the log must carry the
    YAML parser's error naming the config file, so the operator sees where the syntax broke.
    UNVERIFIED: the exact exception class text (`ParserError` or `ScannerError`) comes from PyYAML and
    is not wrapped by the proxy on the `--config` path (`_get_config_from_file`); a Rust loader will
    word it differently, so only the file name and the word `yaml` are pinned."""
    broken: Final = tmp_path / "broken.yaml"
    broken.write_text("model_list: [\n  - model_name: dec1\n general_settings: {\n")
    log: Final = refused(gateway, tmp_path, broken)
    assert "broken.yaml" in log
    assert "yaml" in log.lower()


def test_config_file_path_env_replaces_the_file_named_on_the_command_line(gateway: Gateway, tmp_path: Path) -> None:
    """`--config` names a file declaring `dec1-cli-model` and `CONFIG_FILE_PATH` names another declaring
    `dec1-env-model`. `GET /v1/models` must list the env file's model and not the command-line one: inside
    the server process `CONFIG_FILE_PATH` is consulted first and the `--config` file is never loaded."""
    from_env: Final = write_config(tmp_path, {"model_list": _models("dec1-env-model")}, "from-env.yaml")
    with booted(
        gateway,
        tmp_path,
        {"model_list": _models("dec1-cli-model")},
        environment={"CONFIG_FILE_PATH": str(from_env)},
    ) as owned:
        names: Final = served_model_names(owned.gateway)
    assert "dec1-env-model" in names
    assert "dec1-cli-model" not in names


def test_config_file_path_env_accepts_the_yml_extension(gateway: Gateway, tmp_path: Path) -> None:
    """`CONFIG_FILE_PATH` names a file ending in `.yml`. Its model must be served: both `.yaml` and
    `.yml` count as YAML configs."""
    from_env: Final = write_config(tmp_path, {"model_list": _models("dec1-yml-model")}, "from-env.yml")
    with booted(gateway, tmp_path, {"model_list": []}, environment={"CONFIG_FILE_PATH": str(from_env)}) as owned:
        assert "dec1-yml-model" in served_model_names(owned.gateway)


def test_config_file_path_env_naming_a_missing_file_leaves_the_proxy_without_any_config_models(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`CONFIG_FILE_PATH` names a file that does not exist while `--config` names a good one. The proxy
    still boots (the master key comes from the environment), but `GET /v1/models` lists neither file's
    models: a set-but-unusable `CONFIG_FILE_PATH` is silently skipped and the `--config` file is not used
    as a fallback. Surprising, and pinned so a rewrite does not change it unnoticed in either direction."""
    with booted(
        gateway,
        tmp_path,
        {"model_list": _models("dec1-cli-model")},
        environment={"CONFIG_FILE_PATH": str(tmp_path / "not-there.yaml")},
    ) as owned:
        assert "dec1-cli-model" not in served_model_names(owned.gateway)


def test_config_file_path_env_with_a_non_yaml_extension_is_ignored(gateway: Gateway, tmp_path: Path) -> None:
    """`CONFIG_FILE_PATH` names an existing file with valid YAML content but a `.txt` extension. Its model
    must not be served: only `.yaml` and `.yml` files are accepted through this variable, and anything
    else is skipped without an error."""
    from_env: Final = write_config(tmp_path, {"model_list": _models("dec1-txt-model")}, "from-env.txt")
    with booted(gateway, tmp_path, {"model_list": []}, environment={"CONFIG_FILE_PATH": str(from_env)}) as owned:
        assert "dec1-txt-model" not in served_model_names(owned.gateway)


def test_an_unknown_top_level_key_is_ignored_and_the_rest_of_the_file_applies(gateway: Gateway, tmp_path: Path) -> None:
    """The config carries a top-level key the proxy has never heard of next to a model. The proxy must
    boot and serve the model: unknown top-level keys are not an error."""
    with booted(
        gateway, tmp_path, {"dec1_unknown_section": {"anything": 1}, "model_list": _models("dec1-known-model")}
    ) as owned:
        assert "dec1-known-model" in served_model_names(owned.gateway)


def test_an_unknown_litellm_settings_key_is_ignored(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_settings` holds a key that is not a litellm setting, next to `request_timeout: 77`. The
    proxy must boot and `GET /active/callbacks` must report the timeout as 77: the unknown key is set as
    a plain attribute and has no effect."""
    with booted(gateway, tmp_path, {"litellm_settings": {"dec1_not_a_setting": "x", "request_timeout": 77}}) as owned:
        assert owned.gateway.get("/active/callbacks")["litellm.request_timeout"] == 77


def test_an_unknown_general_settings_key_is_ignored(gateway: Gateway, tmp_path: Path) -> None:
    """`general_settings` holds a key the proxy does not define. The proxy must boot, answer readiness
    with 200, and keep serving the configured model."""
    with booted(
        gateway,
        tmp_path,
        {"general_settings": {"dec1_not_a_setting": True}, "model_list": _models("dec1-known-model")},
    ) as owned:
        assert owned.gateway.client.get("/health/readiness").status_code == 200
        assert "dec1-known-model" in served_model_names(owned.gateway)


def test_an_unknown_router_settings_key_is_ignored_with_a_warning_naming_it(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings` holds `dec1_not_a_router_arg`. The proxy must boot and its log must contain
    `Key 'dec1_not_a_router_arg' is not a valid argument for Router.__init__(). Ignoring this key.`"""
    with booted(
        gateway,
        tmp_path,
        {"router_settings": {"dec1_not_a_router_arg": 5}, "model_list": _models("dec1-known-model")},
    ) as owned:
        assert "dec1-known-model" in served_model_names(owned.gateway)
        log: Final = owned.log.read_text()
    assert "Key 'dec1_not_a_router_arg' is not a valid argument for Router.__init__(). Ignoring this key." in log


def test_health_check_interval_under_router_settings_refuses_to_boot_and_says_where_it_belongs(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`router_settings.health_check_interval` is a common misplacement. The proxy must exit non-zero and
    log `'health_check_interval' is NOT a valid router_settings parameter. Please move it to
    'general_settings'.` instead of silently ignoring it."""
    log: Final = refused(gateway, tmp_path, {"router_settings": {"health_check_interval": 30}})
    assert (
        "'health_check_interval' is NOT a valid router_settings parameter. Please move it to 'general_settings'." in log
    )


def test_jwt_settings_at_the_top_level_are_ignored_with_a_warning(gateway: Gateway, tmp_path: Path) -> None:
    """`enable_jwt_auth: true` sits at the top level instead of under `general_settings`. The proxy must
    boot, log a warning that starts `Ignoring top-level config key(s) enable_jwt_auth.`, and keep
    accepting the master key as a bearer token (JWT auth did not engage)."""
    with booted(gateway, tmp_path, {"enable_jwt_auth": True, "model_list": _models("dec1-known-model")}) as owned:
        assert owned.gateway.request("GET", "/v1/models").status_code == 200
        log: Final = owned.log.read_text()
    assert "Ignoring top-level config key(s) enable_jwt_auth." in log


def test_null_settings_sections_boot_like_absent_ones(gateway: Gateway, tmp_path: Path) -> None:
    """`general_settings:`, `litellm_settings:` and `router_settings:` are present but empty (YAML null),
    as left behind by a commented-out block. The proxy must boot and serve the configured model."""
    path: Final = tmp_path / "nulls.yaml"
    path.write_text(
        "general_settings:\nlitellm_settings:\nrouter_settings:\n"
        + yaml.safe_dump({"model_list": _models("dec1-known-model")})
    )
    with booted(gateway, tmp_path, path) as owned:
        assert "dec1-known-model" in served_model_names(owned.gateway)
