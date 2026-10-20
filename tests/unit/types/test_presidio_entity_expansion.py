"""
Test that PiiEntityType / PII_ENTITY_CATEGORIES_MAP match the entity names of
current upstream Presidio recognizers (presidio-analyzer predefined_recognizers).
"""

from typing import Final

import pytest

from litellm.types.guardrails import PII_ENTITY_CATEGORIES_MAP, PiiEntityCategory, PiiEntityType

EXPECTED_CATEGORY_ENTITIES: Final[dict[PiiEntityCategory, frozenset[str]]] = {
    PiiEntityCategory.GENERAL: frozenset(
        {
            "DATE_TIME",
            "EMAIL_ADDRESS",
            "IP_ADDRESS",
            "NRP",
            "LOCATION",
            "PERSON",
            "PHONE_NUMBER",
            "MEDICAL_LICENSE",
            "URL",
            "MAC_ADDRESS",
            "UUID",
        }
    ),
    PiiEntityCategory.USA: frozenset(
        {
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
            "US_MBI",
            "US_NPI",
        }
    ),
    PiiEntityCategory.UK: frozenset(
        {
            "UK_NHS",
            "UK_NINO",
            "UK_PASSPORT",
            "UK_POSTCODE",
            "UK_VEHICLE_REGISTRATION",
            "UK_DRIVING_LICENCE",
        }
    ),
    PiiEntityCategory.SPAIN: frozenset({"ES_NIF", "ES_NIE", "ES_PASSPORT"}),
    PiiEntityCategory.INDIA: frozenset(
        {
            "IN_PAN",
            "IN_AADHAAR",
            "IN_VEHICLE_REGISTRATION",
            "IN_VOTER",
            "IN_PASSPORT",
            "IN_GSTIN",
        }
    ),
    PiiEntityCategory.GERMANY: frozenset(
        {
            "DE_TAX_ID",
            "DE_TAX_NUMBER",
            "DE_VAT_ID",
            "DE_PASSPORT",
            "DE_ID_CARD",
            "DE_FUEHRERSCHEIN",
            "DE_SOCIAL_SECURITY",
            "DE_HEALTH_INSURANCE",
            "DE_LANR",
            "DE_BSNR",
            "DE_KFZ",
            "DE_HANDELSREGISTER",
            "DE_PLZ",
        }
    ),
    PiiEntityCategory.KOREA: frozenset({"KR_RRN", "KR_FRN", "KR_PASSPORT", "KR_DRIVER_LICENSE", "KR_BRN"}),
    PiiEntityCategory.CANADA: frozenset({"CA_SIN"}),
    PiiEntityCategory.SWEDEN: frozenset({"SE_PERSONNUMMER", "SE_ORGANISATIONSNUMMER"}),
    PiiEntityCategory.THAILAND: frozenset({"TH_TNIN"}),
    PiiEntityCategory.TURKEY: frozenset({"TR_NATIONAL_ID", "TR_LICENSE_PLATE"}),
    PiiEntityCategory.NIGERIA: frozenset({"NG_NIN", "NG_VEHICLE_REGISTRATION"}),
    PiiEntityCategory.PHILIPPINES: frozenset({"PH_TIN", "PH_UMID", "PH_PASSPORT"}),
    PiiEntityCategory.SOUTH_AFRICA: frozenset({"ZA_ID_NUMBER"}),
}


@pytest.mark.parametrize("category", sorted(EXPECTED_CATEGORY_ENTITIES, key=lambda c: c.value))
def test_category_exactly_matches_presidio_recognizers(category: PiiEntityCategory) -> None:
    actual: Final = {entity.value for entity in PII_ENTITY_CATEGORIES_MAP[category]}
    assert actual == set(EXPECTED_CATEGORY_ENTITIES[category])


def test_every_entity_belongs_to_exactly_one_category() -> None:
    all_mapped: Final = [entity for entities in PII_ENTITY_CATEGORIES_MAP.values() for entity in entities]
    assert len(all_mapped) == len(set(all_mapped))
    assert set(all_mapped) == set(PiiEntityType)


def test_entity_names_equal_their_wire_values() -> None:
    assert all(entity.name == entity.value for entity in PiiEntityType)
