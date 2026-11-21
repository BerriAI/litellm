"""`include:` file lookup rules the existing include tests leave open.

`resolve_include_file_path` (`litellm/proxy/common_utils/config_includes.py:11`) looks for an entry next to
the file that declares it, and, for configs written before nested includes resolved that way, falls back
to the directory of the root config with a warning; when both files exist the declaring file's neighbour
wins, also with a warning. The Rust loader copies the same two-step lookup
(`litellm/rust/crates/config-load/src/includes.rs`, `declared` then `fallback`) but logs nothing, so the
warnings are pinned here as the operator-facing part of the contract. An absolute entry ignores both
directories, an empty included file adds nothing, an included file may add a whole section the root lacks,
and an included file that is not valid YAML stops the boot naming that file.

The root config always declares `router_settings.num_retries` so the harness boots the file in place
(`private_configuration` copies a file without it to another directory, which would change what a
relative include means).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration.configuration.dec1_cfgload_support import refused_with_exit, write_yaml_text
from tests.integration.configuration.dec1_config_support import (
    booted,
    chat,
    chat_reply,
    deployment,
    scripted_upstream,
    sent,
    served_model_names,
    write_config,
)

_RETRIES: Final = {"num_retries": 0}


def _models(*names: str) -> dict[str, object]:
    return {"model_list": [deployment(name, "http://127.0.0.1:9/v1") for name in names]}


def test_a_nested_include_missing_beside_its_file_is_read_from_the_root_directory_with_a_warning(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The root includes `nested/middle.yaml`, which includes `leaf.yaml`; `leaf.yaml` exists only next
    to the root, not in `nested/`. The proxy must boot serving the leaf's model, and the log must contain
    `Config include 'leaf.yaml' declared in` and `was not found next to it, so` and `Move the included file
    next to the config that declares it.`: the old root-relative lookup still works but is announced."""
    write_config(tmp_path / "nested", {"include": ["leaf.yaml"]}, "middle.yaml")
    write_config(tmp_path, _models("dec1-leaf-from-root"), "leaf.yaml")
    root: Final = write_config(tmp_path, {"include": ["nested/middle.yaml"], "router_settings": _RETRIES})
    with booted(gateway, tmp_path, root) as owned:
        names: Final = served_model_names(owned.gateway)
        log: Final = owned.log.read_text()
    assert "dec1-leaf-from-root" in names
    assert "Config include 'leaf.yaml' declared in" in log
    assert "was not found next to it, so" in log
    assert "Move the included file next to the config that declares it." in log


def test_when_both_directories_hold_the_include_the_declaring_files_neighbour_wins(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`leaf.yaml` exists both in `nested/` (declaring `dec1-leaf-nested`) and next to the root
    (declaring `dec1-leaf-root`), and `nested/middle.yaml` includes `leaf.yaml`. The proxy must serve
    `dec1-leaf-nested` and not `dec1-leaf-root`, and the log must contain `matches two files` and
    `Rename one of the two to say which one you meant.`"""
    write_config(tmp_path / "nested", {"include": ["leaf.yaml"]}, "middle.yaml")
    write_config(tmp_path / "nested", _models("dec1-leaf-nested"), "leaf.yaml")
    write_config(tmp_path, _models("dec1-leaf-root"), "leaf.yaml")
    root: Final = write_config(tmp_path, {"include": ["nested/middle.yaml"], "router_settings": _RETRIES})
    with booted(gateway, tmp_path, root) as owned:
        names: Final = served_model_names(owned.gateway)
        log: Final = owned.log.read_text()
    assert "dec1-leaf-nested" in names
    assert "dec1-leaf-root" not in names
    assert "matches two files" in log
    assert "Rename one of the two to say which one you meant." in log


def test_an_absolute_include_path_is_read_wherever_it_points(gateway: Gateway, tmp_path: Path) -> None:
    """The root sits in `root/` and includes the absolute path of `elsewhere/models.yaml`. The proxy must
    serve the included model: an absolute entry ignores the declaring and root directories."""
    included: Final = write_config(tmp_path / "elsewhere", _models("dec1-absolute"), "models.yaml")
    root: Final = write_config(tmp_path / "root", {"include": [str(included.resolve())], "router_settings": _RETRIES})
    with booted(gateway, tmp_path / "root", root) as owned:
        assert "dec1-absolute" in served_model_names(owned.gateway)


def test_an_empty_included_file_adds_nothing_and_does_not_stop_the_boot(gateway: Gateway, tmp_path: Path) -> None:
    """The root declares `dec1-root-model` and includes `empty.yaml`, a zero-byte file. The proxy must
    boot and serve `dec1-root-model`: an included file is read with `yaml.safe_load(...) or {}`
    (`_load_yaml_file`), so emptiness is allowed for includes even though an empty root is refused."""
    write_yaml_text(tmp_path, "", "empty.yaml")
    root: Final = write_config(
        tmp_path, {**_models("dec1-root-model"), "include": ["empty.yaml"], "router_settings": _RETRIES}
    )
    with booted(gateway, tmp_path, root) as owned:
        assert "dec1-root-model" in served_model_names(owned.gateway)


def test_an_included_file_can_supply_a_section_the_root_lacks(gateway: Gateway, tmp_path: Path) -> None:
    """The root has the model list and `router_settings.num_retries` only and no `general_settings`;
    `aliases.yaml` supplies `general_settings.litellm_key_header_name: x-dec1-included-header`. `GET
    /v1/models` with the master
    key in `x-dec1-included-header` (as `Bearer <key>`, the form the custom header is parsed in; a bare key is
    refused with `Malformed API Key passed in. Ensure Key has `Bearer ` prefix.`, see
    `test_dec1_config_general_settings_matrix.py`) must return 200: a section that only an included file
    declares is applied like one in the root."""
    write_config(tmp_path, {"general_settings": {"litellm_key_header_name": "x-dec1-included-header"}}, "aliases.yaml")
    root: Final = write_config(
        tmp_path, {**_models("dec1-root-model"), "include": ["aliases.yaml"], "router_settings": _RETRIES}
    )
    with booted(gateway, tmp_path, root) as owned:
        response: Final = owned.gateway.client.get(
            "/v1/models", headers={"x-dec1-included-header": f"Bearer {gateway.key}"}
        )
    assert response.status_code == 200, response.text


def test_an_included_router_settings_replaces_the_roots_router_settings(gateway: Gateway, tmp_path: Path) -> None:
    """The root sets `router_settings: {num_retries: 0}` and `retries.yaml` sets `router_settings:
    {num_retries: 1}`; the upstream answers 500. One chat completion must reach the upstream twice: the
    included section replaces the root's section (no deep merge), which is how a team overrides retries
    from a shared file."""
    with scripted_upstream(lambda request: chat_reply("dec1 down", status=500)) as upstream:
        write_config(tmp_path, {"router_settings": {"num_retries": 1}}, "retries.yaml")
        root: Final = write_config(
            tmp_path,
            {
                "model_list": [deployment("dec1-retry", f"{upstream.url}/v1")],
                "include": ["retries.yaml"],
                "router_settings": _RETRIES,
            },
        )
        with booted(gateway, tmp_path, root) as owned:
            response: Final = chat(owned.gateway, "dec1-retry")
            requests: Final = sent(upstream)
    assert response.status_code == 500, response.text
    assert len(requests) == 2


def test_an_included_file_with_invalid_yaml_stops_the_boot_naming_that_file(gateway: Gateway, tmp_path: Path) -> None:
    """The root includes `broken.yaml`, whose text is an unclosed flow sequence. The proxy must exit
    non-zero and the log must contain `Error loading yaml file` followed by the path of `broken.yaml`
    (`_load_yaml_file`), so the operator is sent to the included file, not the root."""
    broken: Final = write_yaml_text(tmp_path, "model_list: [\n", "broken.yaml")
    root: Final = write_config(tmp_path, {"include": ["broken.yaml"], "router_settings": _RETRIES})
    refusal: Final = refused_with_exit(gateway, tmp_path, root)
    assert refusal.exit_code != 0
    assert (
        f"Error loading yaml file {broken.resolve()}" in refusal.log
        or f"Error loading yaml file {broken}" in refusal.log
    ), refusal.log[-4000:]
