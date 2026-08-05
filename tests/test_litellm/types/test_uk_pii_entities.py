"""
Test UK PII entity types in guardrails module
"""

from litellm.types.guardrails import PiiEntityType, PiiEntityCategory, PII_ENTITY_CATEGORIES_MAP


class TestUKPiiEntities:
    """Test UK PII entity type definitions and mappings"""

    def test_uk_pii_entity_types_exist(self):
        """Test all UK PII entity types are defined"""
        assert hasattr(PiiEntityType, "UK_NHS")
        assert hasattr(PiiEntityType, "UK_NINO")
        assert hasattr(PiiEntityType, "UK_PASSPORT")
        assert hasattr(PiiEntityType, "UK_POSTCODE")
        assert hasattr(PiiEntityType, "UK_VEHICLE_REGISTRATION")

    def test_uk_pii_entity_values(self):
        """Test UK PII entity types have correct string values"""
        assert PiiEntityType.UK_NHS == "UK_NHS"
        assert PiiEntityType.UK_NINO == "UK_NINO"
        assert PiiEntityType.UK_PASSPORT == "UK_PASSPORT"
        assert PiiEntityType.UK_POSTCODE == "UK_POSTCODE"
        assert PiiEntityType.UK_VEHICLE_REGISTRATION == "UK_VEHICLE_REGISTRATION"

    def test_uk_category_exists(self):
        """Test UK category exists in PII_ENTITY_CATEGORIES_MAP"""
        assert PiiEntityCategory.UK in PII_ENTITY_CATEGORIES_MAP

    def test_uk_category_contains_all_entities(self):
        """Test UK category contains all UK PII entity types"""
        uk_entities = PII_ENTITY_CATEGORIES_MAP[PiiEntityCategory.UK]

        assert PiiEntityType.UK_NHS in uk_entities
        assert PiiEntityType.UK_NINO in uk_entities
        assert PiiEntityType.UK_PASSPORT in uk_entities
        assert PiiEntityType.UK_POSTCODE in uk_entities
        assert PiiEntityType.UK_VEHICLE_REGISTRATION in uk_entities

    def test_uk_entities_match_presidio_recognizers(self):
        """Test UK entity type names match Presidio recognizer names"""
        expected_entities = {
            "UK_NHS",
            "UK_NINO",
            "UK_PASSPORT",
            "UK_POSTCODE",
            "UK_VEHICLE_REGISTRATION",
        }

        uk_entities = PII_ENTITY_CATEGORIES_MAP[PiiEntityCategory.UK]
        actual_entities = set(uk_entities)

        assert actual_entities == expected_entities
