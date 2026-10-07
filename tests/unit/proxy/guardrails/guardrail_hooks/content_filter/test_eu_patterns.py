from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    get_compiled_pattern,
)


class TestFrenchNIR:
    """Test French NIR/INSEE detection"""

    def test_valid_nir_detected(self):
        pattern = get_compiled_pattern("fr_nir")
        # Valid NIR: sex=1, year=92, month=05, dept=75, commune=123, order=456, key=78
        assert pattern.search("192057512345678") is not None
        assert pattern.search("292057512345678") is not None  # Female

    def test_invalid_month_rejected(self):
        pattern = get_compiled_pattern("fr_nir")
        assert pattern.search("192137512345678") is None  # Month 13
        assert pattern.search("192007512345678") is None  # Month 00

    def test_invalid_sex_digit_rejected(self):
        pattern = get_compiled_pattern("fr_nir")
        assert pattern.search("392057512345678") is None  # Sex digit 3


class TestEUIBANEnhanced:
    """Test enhanced EU IBAN detection"""

    def test_french_iban(self):
        pattern = get_compiled_pattern("eu_iban_enhanced")
        assert pattern.search("FR7630006000011234567890189") is not None

    def test_german_iban(self):
        pattern = get_compiled_pattern("eu_iban_enhanced")
        assert pattern.search("DE89370400440532013000") is not None


class TestFrenchPhone:
    """Test French phone number detection"""

    def test_formats(self):
        pattern = get_compiled_pattern("fr_phone")
        assert pattern.search("+33612345678") is not None
        assert pattern.search("0033612345678") is not None
        assert pattern.search("0612345678") is not None

    def test_invalid_first_digit(self):
        pattern = get_compiled_pattern("fr_phone")
        assert pattern.search("0012345678") is None  # First digit can't be 0


class TestEUVAT:
    """Test EU VAT number detection"""

    def test_major_eu_countries(self):
        pattern = get_compiled_pattern("eu_vat")
        assert pattern.search("FR12345678901") is not None
        assert pattern.search("DE123456789") is not None
        assert pattern.search("IT12345678901") is not None

    def test_pattern_requires_keyword_context(self):
        """
        NOTE: The eu_vat raw pattern CAN match common words like DEPARTMENT (DE+PARTMENT).
        This is why the pattern REQUIRES keyword_pattern in production use.
        The ContentFilterGuardrail enforces keyword context, preventing false positives.
        This test documents the raw pattern's broad matching behavior.
        """
        pattern = get_compiled_pattern("eu_vat")
        # These WILL match the raw pattern (by design - pattern is broad)
        assert pattern.search("DEPARTMENT") is not None  # DE + PARTMENT
        assert pattern.search("ITALY12345678") is not None  # IT + digits

        # But in production, keyword_pattern guard prevents these false positives


class TestEUPassportGeneric:
    """Test generic EU passport detection"""

    def test_format(self):
        pattern = get_compiled_pattern("eu_passport_generic")
        assert pattern.search("12AB34567") is not None


class TestFrenchPostalCode:
    """Test French postal code contextual detection"""

    def test_with_context(self):
        # This test validates the pattern exists
        # Contextual matching is tested in integration tests
        pattern = get_compiled_pattern("fr_postal_code")
        assert pattern.search("75001") is not None
