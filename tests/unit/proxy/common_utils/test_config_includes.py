"""Unit tests for litellm.proxy.common_utils.config_includes."""

import os

import pytest

from litellm.proxy.common_utils.config_includes import resolve_include_file_path


def test_resolve_include_rejects_path_outside_config_dir(tmp_path):
    """
    Include entries that climb out of the root config directory must be rejected
    before the file is opened (regression for #43480).
    """
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    root_config = cfg_dir / "config.yaml"
    root_config.write_text("model_list: []\n")

    # Same shape as the issue report: ../../etc/passwd from cfg/config.yaml
    with pytest.raises(ValueError, match="outside the config directory"):
        resolve_include_file_path("../../etc/passwd", str(root_config), str(root_config))

    # Absolute path outside the config dir
    with pytest.raises(ValueError, match="outside the config directory"):
        resolve_include_file_path("/etc/passwd", str(root_config), str(root_config))


def test_resolve_include_allows_file_under_config_dir(tmp_path):
    """Sibling and nested includes under the config directory still resolve."""
    cfg_dir = tmp_path / "cfg"
    nested = cfg_dir / "nested"
    nested.mkdir(parents=True)
    root_config = cfg_dir / "config.yaml"
    root_config.write_text("model_list: []\n")
    sibling = cfg_dir / "models.yaml"
    sibling.write_text("model_list: []\n")
    nested_include = nested / "extra.yaml"
    nested_include.write_text("model_list: []\n")

    assert resolve_include_file_path("models.yaml", str(root_config), str(root_config)) == str(
        sibling.resolve()
    )
    # Nested config may climb one level with .. and still stay under cfg/
    assert resolve_include_file_path(
        "../models.yaml", str(nested_include), str(root_config)
    ) == str(sibling.resolve())


def test_resolve_include_rejects_escape_via_legacy_root_fallback(tmp_path):
    """
    When the include is missing next to a nested declarer but present next to the
    root, the legacy fallback must still refuse a path that left the config dir.
    """
    cfg_dir = tmp_path / "cfg"
    nested = cfg_dir / "nested"
    nested.mkdir(parents=True)
    root_config = cfg_dir / "config.yaml"
    root_config.write_text("model_list: []\n")
    nested_config = nested / "part.yaml"
    nested_config.write_text("model_list: []\n")

    # declared next to nested; ../../outside.yaml from nested → tmp_path/outside.yaml
    with pytest.raises(ValueError, match="outside the config directory"):
        resolve_include_file_path("../../outside.yaml", str(nested_config), str(root_config))
