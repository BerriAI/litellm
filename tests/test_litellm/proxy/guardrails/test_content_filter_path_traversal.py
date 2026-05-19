import os
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
