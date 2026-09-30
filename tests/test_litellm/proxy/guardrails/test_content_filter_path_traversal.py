import os
import pathlib
import re
from unittest.mock import patch

import pytest

import litellm
from litellm.proxy.guardrails.content_filter_data import CATEGORIES_DIR, DATA_DIR, LEGACY_DATA_DIR as INSTALLED_LEGACY_DATA_DIR

LEGACY_DATA_DIR = "litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter"


class TestContentFilterPathTraversal:
    """Tests that _resolve_category_file_path rejects path traversal."""

    def _get_guardrail(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        return ContentFilterGuardrail.__new__(ContentFilterGuardrail)

    def test_traversal_via_relative_dotdot_raises(self):
        guardrail = self._get_guardrail()
        with pytest.raises(ValueError, match="outside the allowed categories"):
            guardrail._resolve_category_file_path("../../../../etc/passwd")

    def test_traversal_via_absolute_path_raises(self):
        guardrail = self._get_guardrail()
        with pytest.raises(ValueError, match="outside the allowed categories"):
            guardrail._resolve_category_file_path("/etc/passwd")

    def test_valid_category_file_inside_categories_dir_allowed(self):
        guardrail = self._get_guardrail()
        valid_file = os.path.join(CATEGORIES_DIR, "harmful_self_harm.yaml")
        if not os.path.exists(valid_file):
            pytest.skip("harmful_self_harm.yaml not present in this environment")
        result = guardrail._resolve_category_file_path(valid_file)
        assert result == valid_file

    @pytest.mark.parametrize(
        "legacy_path",
        [
            f"{LEGACY_DATA_DIR}/policy_templates/eu_ai_act_article5.yaml",
            f"{LEGACY_DATA_DIR}/categories/harmful_self_harm.yaml",
        ],
    )
    def test_paths_recorded_before_the_data_move_still_resolve(self, legacy_path, monkeypatch, tmp_path):
        """Policies saved by older releases point at the old package-internal folders."""
        monkeypatch.chdir(tmp_path)
        resolved = self._get_guardrail()._resolve_category_file_path(legacy_path)
        assert os.path.isfile(resolved)
        assert os.path.realpath(resolved) == os.path.realpath(os.path.join(DATA_DIR, *legacy_path.split("/")[-2:]))

    def test_every_category_file_published_in_policy_templates_resolves(self, monkeypatch, tmp_path):
        """The proxy fetches policy_templates.json from main, so every path in it must exist in the package."""
        monkeypatch.chdir(tmp_path)
        published = os.path.join(os.path.dirname(os.path.dirname(litellm.__file__)), "policy_templates.json")
        category_files = re.findall(r'"category_file":\s*"([^"]+)"', open(published).read())
        assert category_files
        guardrail = self._get_guardrail()
        missing = [p for p in category_files if not os.path.isfile(guardrail._resolve_category_file_path(p))]
        assert missing == []

    def test_invalid_category_name_skipped(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        guardrail = ContentFilterGuardrail.__new__(ContentFilterGuardrail)
        guardrail.loaded_categories = {}
        guardrail.severity_threshold = "medium"
        guardrail.category_keywords = {}
        guardrail.always_block_category_keywords = {}
        guardrail.conditional_categories = {}
        # category name with path traversal chars must be skipped, not crash
        guardrail._load_categories([{"category": "../../etc/passwd", "enabled": True}])
        assert "../../etc/passwd" not in guardrail.loaded_categories

    def test_category_name_with_slash_skipped(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        guardrail = ContentFilterGuardrail.__new__(ContentFilterGuardrail)
        guardrail.loaded_categories = {}
        guardrail.severity_threshold = "medium"
        guardrail.category_keywords = {}
        guardrail.always_block_category_keywords = {}
        guardrail.conditional_categories = {}
        guardrail._load_categories(
            [{"category": "foo/../../etc/passwd", "enabled": True}]
        )
        assert "foo/../../etc/passwd" not in guardrail.loaded_categories

    def test_assert_within_data_roots_blocks_parent_traversal(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        with pytest.raises(ValueError, match="outside the allowed categories"):
            ContentFilterGuardrail._assert_within_data_roots("/etc/passwd", (CATEGORIES_DIR,))

    def test_assert_within_data_roots_allows_valid_file(self, tmp_path):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        categories_dir = str(tmp_path)
        valid_file = str(tmp_path / "test.yaml")
        # Should not raise
        ContentFilterGuardrail._assert_within_data_roots(valid_file, (categories_dir,))

    def test_assert_within_data_roots_commonpath_raises_valueerror(self, tmp_path):
        """Cover the except-ValueError branch (Windows cross-drive paths)."""
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        categories_dir = str(tmp_path)
        valid_file = str(tmp_path / "test.yaml")
        with patch(
            "os.path.commonpath", side_effect=ValueError("Paths on different drives")
        ):
            with pytest.raises(
                ValueError, match="outside the allowed categories directory"
            ):
                ContentFilterGuardrail._assert_within_data_roots(valid_file, (categories_dir,))

    def test_resolve_category_file_path_direct_join_hit(self):
        """Cover the first-join-attempt success branch (lines 383-384)."""
        guardrail = self._get_guardrail()
        # "categories/<file>" joined directly to the data dir resolves to an existing file.
        yaml_files = [f for f in os.listdir(CATEGORIES_DIR) if f.endswith(".yaml")]
        if not yaml_files:
            pytest.skip("No category YAML files present in this environment")
        relative_path = os.path.join("categories", yaml_files[0])
        result = guardrail._resolve_category_file_path(relative_path)
        assert os.path.isabs(result) or os.path.exists(result)

    def test_resolve_category_file_path_component_strip_hit(self):
        """Cover the component-stripping loop success branch (lines 392-393)."""
        guardrail = self._get_guardrail()
        yaml_files = [f for f in os.listdir(CATEGORIES_DIR) if f.endswith(".yaml")]
        if not yaml_files:
            pytest.skip("No category YAML files present in this environment")
        # Prefix with a fake leading component so the first-join attempt misses,
        # but stripping that component reveals categories/<file> which exists.
        prefixed_path = "some_prefix/categories/" + yaml_files[0]
        result = guardrail._resolve_category_file_path(prefixed_path)
        assert os.path.isabs(result) or os.path.exists(result)

    def test_load_categories_traversal_category_file_skipped(self):
        """Cover the except-ValueError branch in _load_categories (lines 451-454)."""
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        guardrail = ContentFilterGuardrail.__new__(ContentFilterGuardrail)
        guardrail.loaded_categories = {}
        guardrail.severity_threshold = "medium"
        guardrail.category_keywords = {}
        guardrail.always_block_category_keywords = {}
        guardrail.conditional_categories = {}
        # A traversal path in category_file must be skipped (not crash) via ValueError.
        guardrail._load_categories(
            [
                {
                    "category": "valid_name",
                    "enabled": True,
                    "category_file": "../../../../etc/passwd",
                }
            ]
        )
        assert "valid_name" not in guardrail.loaded_categories

    def test_allow_external_paths_env_var_bypasses_jail(self, tmp_path):
        """LITELLM_CONTENT_FILTER_ALLOW_EXTERNAL_PATHS=true skips the directory jail."""
        import os as _os
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        guardrail = ContentFilterGuardrail.__new__(ContentFilterGuardrail)
        # Create a real file outside the module directory (simulates mounted volume).
        external_file = tmp_path / "external_categories.yaml"
        external_file.write_text("category_name: test\n")

        with patch.dict(
            _os.environ, {"LITELLM_CONTENT_FILTER_ALLOW_EXTERNAL_PATHS": "true"}
        ):
            # Should return the path without raising ValueError.
            result = guardrail._resolve_category_file_path(str(external_file))
        assert result == str(external_file)

    def test_traversal_blocked_when_allow_external_not_set(self):
        """Without the env var the jail still blocks traversal paths."""
        import os as _os

        guardrail = self._get_guardrail()
        with patch.dict(_os.environ, {}, clear=False):
            _os.environ.pop("LITELLM_CONTENT_FILTER_ALLOW_EXTERNAL_PATHS", None)
            with pytest.raises(ValueError, match="outside the allowed categories"):
                guardrail._resolve_category_file_path("/etc/passwd")


def _fresh_guardrail():
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )

    guardrail = ContentFilterGuardrail.__new__(ContentFilterGuardrail)
    guardrail.loaded_categories = {}
    guardrail.severity_threshold = "medium"
    guardrail.category_keywords = {}
    guardrail.always_block_category_keywords = {}
    guardrail.conditional_categories = {}
    return guardrail


CUSTOM_CATEGORY_YAML = """category_name: custom_legacy
display_name: Custom Legacy
description: copied into the old package folder by a deployment
default_action: BLOCK
keywords:
  - keyword: legacycopyword
    severity: high
"""


@pytest.fixture
def legacy_root(tmp_path):
    """A stand-in for the pre-move package dir with a deployment's own category file inside."""
    root = tmp_path / "litellm_content_filter"
    (root / "categories").mkdir(parents=True)
    (root / "categories" / "custom_legacy.yaml").write_text(CUSTOM_CATEGORY_YAML)
    return str(root)


class TestLegacyPackageRootStaysSearchable:
    """Files a deployment copied into the old guardrail package dir must keep working after the move."""

    def test_installed_legacy_root_is_the_old_package_dir(self):
        assert INSTALLED_LEGACY_DATA_DIR.endswith(os.path.join("guardrail_hooks", "litellm_content_filter"))
        assert os.path.isdir(INSTALLED_LEGACY_DATA_DIR)

    def test_custom_category_file_under_legacy_root_resolves(self, legacy_root):
        roots = (DATA_DIR, legacy_root)
        custom = os.path.join(legacy_root, "categories", "custom_legacy.yaml")
        assert _fresh_guardrail()._resolve_category_file_path(custom, roots) == custom

    def test_custom_category_file_relative_to_legacy_root_resolves(self, legacy_root, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        resolved = _fresh_guardrail()._resolve_category_file_path("categories/custom_legacy.yaml", (DATA_DIR, legacy_root))
        assert os.path.realpath(resolved) == os.path.realpath(os.path.join(legacy_root, "categories", "custom_legacy.yaml"))

    def test_bundled_root_wins_when_both_roots_hold_the_name(self, legacy_root):
        resolved = _fresh_guardrail()._resolve_category_file_path(
            "categories/harmful_self_harm.yaml", (DATA_DIR, legacy_root)
        )
        assert os.path.realpath(resolved) == os.path.realpath(os.path.join(CATEGORIES_DIR, "harmful_self_harm.yaml"))

    def test_custom_category_loads_by_name_from_legacy_root(self, legacy_root):
        guardrail = _fresh_guardrail()
        guardrail._load_categories([{"category": "custom_legacy", "enabled": True}], (DATA_DIR, legacy_root))
        assert "custom_legacy" in guardrail.loaded_categories
        assert "legacycopyword" in guardrail.category_keywords

    def test_custom_category_loads_via_category_file_under_legacy_root(self, legacy_root):
        guardrail = _fresh_guardrail()
        guardrail._load_categories(
            [
                {
                    "category": "custom_legacy",
                    "enabled": True,
                    "category_file": os.path.join(legacy_root, "categories", "custom_legacy.yaml"),
                }
            ],
            (DATA_DIR, legacy_root),
        )
        assert "custom_legacy" in guardrail.loaded_categories

    def test_traversal_still_rejected_with_two_roots(self, legacy_root):
        with pytest.raises(ValueError, match="outside the allowed categories"):
            _fresh_guardrail()._resolve_category_file_path("../../../../etc/passwd", (DATA_DIR, legacy_root))

    def test_file_outside_every_root_rejected(self, legacy_root, tmp_path):
        outside = tmp_path / "elsewhere.yaml"
        outside.write_text(CUSTOM_CATEGORY_YAML)
        with pytest.raises(ValueError, match="outside the allowed categories"):
            _fresh_guardrail()._resolve_category_file_path(str(outside), (DATA_DIR, legacy_root))

    def test_ui_listing_includes_legacy_root_and_lists_each_name_once(self, legacy_root):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
            get_available_content_categories,
        )

        listed = get_available_content_categories((DATA_DIR, legacy_root))
        names = [c["name"] for c in listed]
        assert "custom_legacy" in names
        assert "harmful_self_harm" in names
        assert len(names) == len(set(names))
        assert names == sorted(names)

    def test_ui_listing_prefers_bundled_copy_on_name_clash(self, legacy_root):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
            get_available_content_categories,
        )

        clash = CUSTOM_CATEGORY_YAML.replace("custom_legacy", "harmful_self_harm").replace(
            "Custom Legacy", "Shadowed Copy"
        )
        (pathlib.Path(legacy_root) / "categories" / "harmful_self_harm.yaml").write_text(clash)
        listed = {c["name"]: c for c in get_available_content_categories((DATA_DIR, legacy_root))}
        assert listed["harmful_self_harm"]["display_name"] != "Shadowed Copy"

    def test_find_category_file_falls_through_to_legacy_root(self, legacy_root):
        from litellm.proxy.guardrails.content_filter_data import find_category_file

        roots = (DATA_DIR, legacy_root)
        assert find_category_file("custom_legacy", roots) == os.path.join(legacy_root, "categories", "custom_legacy.yaml")
        assert find_category_file("harmful_self_harm", roots) == os.path.join(CATEGORIES_DIR, "harmful_self_harm.yaml")
        assert find_category_file("no_such_category_anywhere", roots) is None
