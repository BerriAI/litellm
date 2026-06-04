import os
from unittest.mock import patch
import pytest


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
        categories_dir = os.path.join(
            os.path.dirname(
                __import__(
                    "litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter",
                    fromlist=["content_filter"],
                ).__file__
            ),
            "categories",
        )
        valid_file = os.path.join(categories_dir, "harmful_self_harm.yaml")
        if not os.path.exists(valid_file):
            pytest.skip("harmful_self_harm.yaml not present in this environment")
        result = guardrail._resolve_category_file_path(valid_file)
        assert result == valid_file

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

    def test_assert_within_categories_dir_blocks_parent_traversal(self):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        categories_dir = os.path.join(
            os.path.dirname(
                __import__(
                    "litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter",
                    fromlist=["content_filter"],
                ).__file__
            ),
            "categories",
        )
        with pytest.raises(ValueError, match="outside the allowed categories"):
            ContentFilterGuardrail._assert_within_categories_dir(
                "/etc/passwd", categories_dir
            )

    def test_assert_within_categories_dir_allows_valid_file(self, tmp_path):
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
            ContentFilterGuardrail,
        )

        categories_dir = str(tmp_path)
        valid_file = str(tmp_path / "test.yaml")
        # Should not raise
        ContentFilterGuardrail._assert_within_categories_dir(valid_file, categories_dir)

    def test_assert_within_categories_dir_commonpath_raises_valueerror(self, tmp_path):
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
                ContentFilterGuardrail._assert_within_categories_dir(
                    valid_file, categories_dir
                )

    def test_resolve_category_file_path_direct_join_hit(self):
        """Cover the first-join-attempt success branch (lines 383-384)."""
        guardrail = self._get_guardrail()
        # "categories/<file>" joined directly to module_dir resolves to an existing file.
        categories_dir = os.path.join(
            os.path.dirname(
                __import__(
                    "litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter",
                    fromlist=["content_filter"],
                ).__file__
            ),
            "categories",
        )
        yaml_files = [f for f in os.listdir(categories_dir) if f.endswith(".yaml")]
        if not yaml_files:
            pytest.skip("No category YAML files present in this environment")
        relative_path = os.path.join("categories", yaml_files[0])
        result = guardrail._resolve_category_file_path(relative_path)
        assert os.path.isabs(result) or os.path.exists(result)

    def test_resolve_category_file_path_component_strip_hit(self):
        """Cover the component-stripping loop success branch (lines 392-393)."""
        guardrail = self._get_guardrail()
        categories_dir = os.path.join(
            os.path.dirname(
                __import__(
                    "litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter",
                    fromlist=["content_filter"],
                ).__file__
            ),
            "categories",
        )
        yaml_files = [f for f in os.listdir(categories_dir) if f.endswith(".yaml")]
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
