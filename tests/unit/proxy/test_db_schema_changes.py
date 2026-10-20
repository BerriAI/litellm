from typing import Final

import pytest

from scripts.check_schema_compatibility import check_breaking_changes, parse_model_fields

ORIGINAL: Final = "model Item {\n id String @id\n label String?\n}\n"


def test_aaaaaschema_compatibility() -> None:
    added: Final = "model Item {\n id String @id\n label String?\n count Int?\n}\nmodel Extra {\n id String @id\n}\n"
    assert check_breaking_changes(parse_model_fields(ORIGINAL), parse_model_fields(added)) == ()


def test_parse_model_fields_retains_types_and_attributes() -> None:
    assert parse_model_fields(ORIGINAL) == {"Item": {"id": "String @id", "label": "String?"}}


def test_block_attributes_and_comments_are_not_fields() -> None:
    schema: Final = (
        "model Item {\n id String @id\n label String?\n // free text\n"
        ' @@unique([label, id], map: "item_label_id_key")\n @@index([label, id])\n @@index([id])\n}\n'
    )
    assert parse_model_fields(schema) == {"Item": {"id": "String @id", "label": "String?"}}


def test_changing_a_composite_unique_key_is_not_a_field_change() -> None:
    before: Final = "model Item {\n id String @id\n label String?\n @@unique([label, id])\n}\n"
    after: Final = "model Item {\n id String @id\n label String?\n @@unique([label, id, tenant])\n}\n"
    assert check_breaking_changes(parse_model_fields(before), parse_model_fields(after)) == ()


def test_a_narrowed_column_is_still_caught_next_to_block_attributes() -> None:
    before: Final = "model Item {\n id String @id\n label String?\n @@index([label, id])\n}\n"
    after: Final = "model Item {\n id String @id\n label String\n @@index([label, id])\n}\n"
    assert check_breaking_changes(parse_model_fields(before), parse_model_fields(after)) == (
        "Breaking: Field Item.label changed from optional to required",
    )


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
