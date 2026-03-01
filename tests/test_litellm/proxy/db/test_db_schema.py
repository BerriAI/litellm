"""Tests for litellm.proxy.db.db_schema helper."""

import os

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

    def test_whitespace_only_returns_as_is(self, monkeypatch):
        """Whitespace-only value is treated as a truthy string by os.getenv."""
        monkeypatch.setenv("DATABASE_SCHEMA", "  ")
        assert get_database_schema() == "  "
