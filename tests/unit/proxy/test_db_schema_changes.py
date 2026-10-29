from typing import Final

import pytest

from scripts.check_schema_compatibility import check_breaking_changes, parse_model_fields

ORIGINAL: Final = "model Item {\n id String @id\n label String?\n}\n"


def test_aaaaaschema_compatibility() -> None:
    added: Final = "model Item {\n id String @id\n label String?\n count Int?\n}\nmodel Extra {\n id String @id\n}\n"
    assert check_breaking_changes(parse_model_fields(ORIGINAL), parse_model_fields(added)) == ()


def test_parse_model_fields_retains_types_and_attributes() -> None:
    assert parse_model_fields(ORIGINAL) == {"Item": {"id": "String @id", "label": "String?"}}


def test_parse_model_fields_keeps_json_defaults_and_empty_models() -> None:
    schema: Final = 'model Item {\n data Json @default("{}")\n}\nmodel Empty {\n}\n'
    assert parse_model_fields(schema) == {"Item": {"data": 'Json @default("{}")'}, "Empty": {}}


@pytest.mark.parametrize(
    ("schema", "expected"),
    (
        ("", ("Breaking: Model Item was removed",)),
        ("model Item {\n id String @id\n}\n", ("Breaking: Field Item.label was removed",)),
        (
            "model Item {\n id String @id\n label String\n}\n",
            ("Breaking: Field Item.label changed from optional to required",),
        ),
        (
            "model Item {\n id String @id\n label Int?\n}\n",
            ("Breaking: Field Item.label changed type from String? to Int?",),
        ),
    ),
)
def test_schema_compatibility_rejects_removed_models_fields_and_narrowed_types(
    schema: str, expected: tuple[str, ...]
) -> None:
    assert check_breaking_changes(parse_model_fields(ORIGINAL), parse_model_fields(schema)) == expected


RELATED: Final = (
    "model Owner {\n id String @id\n items Item[] @relation(\"Owned\")\n created Item[] @relation(\"Created\")\n}\n"
    "model Item {\n id String @id\n owner_id String?\n created_by String\n"
    ' owner Owner? @relation("Owned", fields: [owner_id], references: [id])\n'
    ' creator Owner @relation("Created", fields: [created_by], references: [id])\n}\n'
)


def test_schema_compatibility_allows_dropping_relations_but_not_their_columns() -> None:
    relations_dropped: Final = (
        "model Owner {\n id String @id\n}\nmodel Item {\n id String @id\n owner_id String?\n created_by String\n}\n"
    )
    columns_dropped: Final = "model Owner {\n id String @id\n}\nmodel Item {\n id String @id\n}\n"
    assert check_breaking_changes(parse_model_fields(RELATED), parse_model_fields(relations_dropped)) == ()
    assert check_breaking_changes(parse_model_fields(RELATED), parse_model_fields(columns_dropped)) == (
        "Breaking: Field Item.owner_id was removed",
        "Breaking: Field Item.created_by was removed",
    )


def test_schema_compatibility_still_rejects_changes_to_a_kept_relation() -> None:
    old_creator: Final = 'Owner @relation("Created", fields: [created_by], references: [id])'
    new_creator: Final = 'Owner @relation("Created", fields: [created_by], references: [id], onDelete: Restrict)'
    restricted: Final = RELATED.replace(old_creator, new_creator)
    assert check_breaking_changes(parse_model_fields(RELATED), parse_model_fields(restricted)) == (
        f"Breaking: Field Item.creator changed type from {old_creator} to {new_creator}",
    )
