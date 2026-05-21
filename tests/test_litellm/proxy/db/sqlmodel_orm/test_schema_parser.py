"""Unit tests for the ``schema.prisma`` parser.

Run with::

    uv run pytest tests/test_litellm/proxy/db/sqlmodel/test_schema_parser.py -vv
"""

from __future__ import annotations

import textwrap

import pytest

from litellm.proxy.db.sqlmodel.schema_parser import (
    PrismaField,
    PrismaRelation,
    parse_schema,
)


def test_parse_simple_model():
    src = textwrap.dedent(
        """
        model Foo {
            id String @id @default(uuid())
            name String @unique
        }
        """
    )
    schema = parse_schema(src)
    assert "Foo" in schema.models
    foo = schema.models["Foo"]
    assert foo.table_name == "Foo"
    assert foo.primary_key == ("id",)
    assert [f.name for f in foo.fields] == ["id", "name"]
    assert foo.fields[0].is_id
    assert foo.fields[0].has_default
    assert foo.fields[0].default_raw == "uuid()"
    assert foo.fields[1].is_unique


def test_optional_and_array_fields():
    src = textwrap.dedent(
        """
        model Foo {
            id String @id
            tags String[] @default([])
            note String?
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    f_tags = foo.field_by_name("tags")
    assert f_tags is not None
    assert f_tags.is_list and not f_tags.is_optional
    assert f_tags.has_default and f_tags.default_raw == "[]"
    f_note = foo.field_by_name("note")
    assert f_note is not None
    assert f_note.is_optional and not f_note.is_list


def test_composite_primary_key_and_index():
    src = textwrap.dedent(
        """
        model Foo {
            a String
            b String
            c Int @default(0)

            @@id([a, b])
            @@index([c])
            @@unique([a, c])
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    assert foo.primary_key == ("a", "b")
    assert len(foo.indexes) == 1
    assert foo.indexes[0].fields == ("c",)
    assert len(foo.uniques) == 1
    assert foo.uniques[0].fields == ("a", "c")


def test_index_with_map_and_sort():
    src = textwrap.dedent(
        """
        model Foo {
            a String @id
            b DateTime
            c String

            @@index([a, b, c(sort: Desc)], map: "Foo_custom_idx")
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    assert len(foo.indexes) == 1
    idx = foo.indexes[0]
    assert idx.map_name == "Foo_custom_idx"
    assert idx.fields == ("a", "b", "c")


def test_at_map_renames_column():
    src = textwrap.dedent(
        """
        model Foo {
            id String @id
            created String @map("created_at")
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    f = foo.field_by_name("created")
    assert f is not None
    assert f.column_name == "created_at"


def test_at_at_map_renames_table():
    src = textwrap.dedent(
        """
        model Foo {
            id String @id
            @@map("foo_table")
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    assert foo.table_name == "foo_table"


def test_relations_are_separated_from_fields():
    src = textwrap.dedent(
        """
        model Bar {
            id String @id
        }

        model Foo {
            id String @id
            bar_id String?
            bar Bar? @relation(fields: [bar_id], references: [id])
            many Bar[]
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    field_names = {f.name for f in foo.fields}
    rel_names = {r.name for r in foo.relations}
    assert field_names == {"id", "bar_id"}
    assert rel_names == {"bar", "many"}
    rel_bar = next(r for r in foo.relations if r.name == "bar")
    assert rel_bar.target_model == "Bar"
    assert rel_bar.is_optional and not rel_bar.is_list


def test_enum_parsed():
    src = textwrap.dedent(
        """
        enum Status {
            ACTIVE
            INACTIVE
        }

        model Foo {
            id String @id
            status Status @default(INACTIVE)
        }
        """
    )
    schema = parse_schema(src)
    assert schema.enums["Status"].values == ("ACTIVE", "INACTIVE")
    f = schema.models["Foo"].field_by_name("status")
    assert f is not None
    assert f.base_type == "Status"
    assert f.default_raw == "INACTIVE"


def test_handles_trailing_block_comment_on_model_line():
    """``model Foo { // comment`` should still be recognized as a model."""
    src = textwrap.dedent(
        """
        model Foo { // a trailing comment after the brace
            id String @id
        }
        """
    )
    schema = parse_schema(src)
    assert "Foo" in schema.models


def test_strip_comment_handles_quoted_double_slash():
    """A `//` inside a quoted string default must not be treated as a comment."""
    src = textwrap.dedent(
        """
        model Foo {
            id String @id
            url String @default("https://example.com")
        }
        """
    )
    foo = parse_schema(src).models["Foo"]
    f = foo.field_by_name("url")
    assert f is not None
    assert f.default_raw == '"https://example.com"'


def test_real_schema_round_trip(tmp_path):
    """Parse the actual repository ``schema.prisma`` and assert basic shape.

    This is a smoke test -- the deeper structural parity check lives in
    ``test_parity.py``.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve()
    while not (repo_root / "schema.prisma").exists():
        if repo_root.parent == repo_root:
            pytest.skip("schema.prisma not found in any ancestor directory")
        repo_root = repo_root.parent
    schema = parse_schema((repo_root / "schema.prisma").read_text())
    assert len(schema.models) >= 60
    assert "LiteLLM_VerificationToken" in schema.models
    assert "LiteLLM_TeamMembership" in schema.models
    # composite PK on TeamMembership
    assert schema.models["LiteLLM_TeamMembership"].primary_key == (
        "user_id",
        "team_id",
    )
