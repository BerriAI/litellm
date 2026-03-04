"""Tests for litellm.proxy.db.db_schema helper."""



import pytest

from litellm.proxy.db.db_schema import get_database_schema


class TestGetDatabaseSchema:
    """Verify get_database_schema reads DATABASE_SCHEMA env var correctly."""

    def test_default_is_public(self, monkeypatch):
        monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
        assert get_database_schema() == "public"

    def test_custom_schema(self, monkeypatch):
        monkeypatch.setenv("DATABASE_SCHEMA", "myschema")
        assert get_database_schema() == "myschema"

    def test_empty_string_falls_back_to_public(self, monkeypatch):
        monkeypatch.setenv("DATABASE_SCHEMA", "")
        assert get_database_schema() == "public"

    def test_whitespace_only_falls_back_to_public(self, monkeypatch):
        """Whitespace-only value is treated as empty and falls back to 'public'."""
        monkeypatch.setenv("DATABASE_SCHEMA", "  ")
        assert get_database_schema() == "public"

    @pytest.mark.parametrize(
        "bad_value",
        [
            "my schema",        # space
            "my-schema",        # hyphen
            "schema;DROP TABLE",  # SQL injection attempt
            "schema'",          # single quote
            'schema"',          # double quote
            "schema.table",     # dot
        ],
    )
    def test_invalid_characters_raise_value_error(self, monkeypatch, bad_value):
        """Schema names containing non-alphanumeric/underscore chars raise ValueError."""
        monkeypatch.setenv("DATABASE_SCHEMA", bad_value)
        with pytest.raises(ValueError, match="Invalid DATABASE_SCHEMA"):
            get_database_schema()
