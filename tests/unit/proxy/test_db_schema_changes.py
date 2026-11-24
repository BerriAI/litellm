from typing import Final

import pytest

from litellm.scripts.check_schema_compatibility import check_breaking_changes, parse_model_fields

ORIGINAL: Final = "model Item {\n id String @id\n label String?\n}\n"


def test_aaaaaschema_compatibility() -> None:
    added: Final = "model Item {\n id String @id\n label String?\n count Int?\n}\nmodel Extra {\n id String @id\n}\n"
    assert check_breaking_changes(parse_model_fields(ORIGINAL), parse_model_fields(added)) == ()


def test_changing_only_a_field_attribute_is_not_breaking() -> None:
    previous: Final = parse_model_fields('model Item {\n id String @id @default(dbgenerated("a()::text"))\n}\n')
    current: Final = parse_model_fields('model Item {\n id String @id @default(dbgenerated("(a())::text"))\n}\n')
    assert check_breaking_changes(previous, current) == ()


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


def test_only_the_dropped_accounting_models_may_be_removed() -> None:
    accounting: Final = (
        "model LiteLLM_AccountingEvent {\n event_id String @id\n}\n"
        "model LiteLLM_AccountingDelivery {\n event_id String\n note String?\n}\n"
    )
    previous: Final = parse_model_fields(ORIGINAL + accounting)

    assert check_breaking_changes(previous, parse_model_fields(ORIGINAL)) == ()
    assert check_breaking_changes(previous, parse_model_fields("")) == ("Breaking: Model Item was removed",)
    assert check_breaking_changes(
        previous, parse_model_fields(ORIGINAL + "model LiteLLM_AccountingDelivery {\n event_id String\n}\n")
    ) == ("Breaking: Field LiteLLM_AccountingDelivery.note was removed",)


def test_only_the_renamed_customer_member_fields_may_be_removed() -> None:
    member: Final = (
        "model LiteLLM_CustomerMemberTable {\n alias String?\n budget_template_id String?\n budget_template Item?\n}\n"
    )
    previous: Final = parse_model_fields(ORIGINAL + member)
    renamed: Final = "model LiteLLM_CustomerMemberTable {\n alias String?\n budget_id String?\n}\n"
    without_alias: Final = "model LiteLLM_CustomerMemberTable {\n budget_id String?\n}\n"

    assert check_breaking_changes(previous, parse_model_fields(ORIGINAL + renamed)) == ()
    assert check_breaking_changes(previous, parse_model_fields(ORIGINAL + without_alias)) == (
        "Breaking: Field LiteLLM_CustomerMemberTable.alias was removed",
    )
