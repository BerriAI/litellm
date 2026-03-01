"""
Tests for DailyTagSpend SQL view logic.

Validates that the view correctly attributes spend across tags by dividing
each request's spend by the number of tags, preventing double-counting when
a request has multiple tags.

See: https://github.com/BerriAI/litellm/issues/21894
"""

import re
import textwrap


def _get_daily_tag_spend_sql() -> str:
    """Extract the DailyTagSpend view SQL from create_views.py source file."""
    import os

    filepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        "litellm", "proxy", "db", "create_views.py",
    )
    with open(filepath, "r") as f:
        source = f.read()
    match = re.search(
        r'(CREATE OR REPLACE VIEW "DailyTagSpend" AS.*?;)',
        source,
        re.DOTALL,
    )
    assert match, "Could not find DailyTagSpend view SQL in create_views.py"
    return textwrap.dedent(match.group(1)).strip()


class TestDailyTagSpendViewSQL:
    """Test the DailyTagSpend SQL view definition."""

    def test_view_divides_spend_by_tag_count(self):
        """The view must divide spend by jsonb_array_length(request_tags)."""
        sql = _get_daily_tag_spend_sql()
        assert "jsonb_array_length(request_tags)" in sql, (
            "View must use jsonb_array_length to divide spend across tags"
        )
        assert "spend / jsonb_array_length(request_tags)" in sql, (
            "View must divide spend by number of tags for fair attribution"
        )

    def test_view_filters_null_request_tags(self):
        """The view must filter out rows with NULL request_tags."""
        sql = _get_daily_tag_spend_sql()
        assert "request_tags IS NOT NULL" in sql, (
            "View must exclude rows where request_tags is NULL"
        )

    def test_view_filters_empty_request_tags(self):
        """The view must filter out rows with empty request_tags arrays."""
        sql = _get_daily_tag_spend_sql()
        assert "jsonb_array_length(request_tags) > 0" in sql, (
            "View must exclude rows with zero-length tag arrays to avoid division by zero"
        )

    def test_view_uses_jsonb_array_elements_text(self):
        """The view must use jsonb_array_elements_text to explode tags."""
        sql = _get_daily_tag_spend_sql()
        assert "jsonb_array_elements_text(request_tags)" in sql

    def test_view_groups_by_tag_and_date(self):
        """The view must group by individual_request_tag and spend_date."""
        sql = _get_daily_tag_spend_sql()
        assert 'individual_request_tag, DATE(s."startTime")' in sql

    def test_db_scripts_create_views_matches_proxy(self):
        """db_scripts/create_views.py must have the same DailyTagSpend SQL as litellm/proxy/db/create_views.py."""
        import os

        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        )
        proxy_path = os.path.join(repo_root, "litellm", "proxy", "db", "create_views.py")
        db_scripts_path = os.path.join(repo_root, "db_scripts", "create_views.py")

        def _extract_view_sql(filepath: str) -> str:
            with open(filepath, "r") as f:
                content = f.read()
            match = re.search(
                r'(CREATE OR REPLACE VIEW "DailyTagSpend" AS.*?;)',
                content,
                re.DOTALL,
            )
            assert match, f"DailyTagSpend SQL not found in {filepath}"
            # Normalize whitespace for comparison
            return " ".join(match.group(1).split())

        proxy_sql = _extract_view_sql(proxy_path)
        db_scripts_sql = _extract_view_sql(db_scripts_path)
        assert proxy_sql == db_scripts_sql, (
            "DailyTagSpend SQL must be identical in proxy/db/create_views.py and db_scripts/create_views.py"
        )
