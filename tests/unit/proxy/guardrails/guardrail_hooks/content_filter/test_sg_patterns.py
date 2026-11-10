"""
Test Singapore PII regex patterns added for PDPA compliance.

Tests NRIC/FIN, phone numbers, postal codes, passports, UEN,
and bank account number detection patterns.
"""

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    get_compiled_pattern,
)


class TestSingaporeNRIC:
    """Test Singapore NRIC/FIN detection"""

    def test_valid_nric_detected(self):
        pattern = get_compiled_pattern("sg_nric")
        # S-series (citizens born 1968–1999)
        assert pattern.search("S1234567A") is not None
        # T-series (citizens born 2000+)
        assert pattern.search("T0123456Z") is not None
        # F-series (foreigners before 2000)
        assert pattern.search("F9876543B") is not None
        # G-series (foreigners 2000+)
        assert pattern.search("G1234567X") is not None
        # M-series (foreigners from 2022)
        assert pattern.search("M1234567K") is not None

    def test_nric_in_sentence(self):
        pattern = get_compiled_pattern("sg_nric")
        assert pattern.search("My NRIC is S1234567A please check") is not None

    def test_lowercase_letter_prefix_detected_case_insensitive(self):
        pattern = get_compiled_pattern("sg_nric")
        # Patterns are compiled with re.IGNORECASE in patterns.py
        assert pattern.search("s1234567A") is not None

    def test_wrong_prefix_rejected(self):
        pattern = get_compiled_pattern("sg_nric")
        assert pattern.search("A1234567Z") is None
        assert pattern.search("X9876543B") is None

    def test_too_few_digits_rejected(self):
        pattern = get_compiled_pattern("sg_nric")
        assert pattern.search("S123456A") is None  # Only 6 digits

    def test_too_many_digits_rejected(self):
        pattern = get_compiled_pattern("sg_nric")
        assert pattern.search("S12345678A") is None  # 8 digits


class TestSingaporePhone:
    """Test Singapore phone number detection"""

    def test_with_plus65_prefix(self):
        pattern = get_compiled_pattern("sg_phone")
        assert pattern.search("+6591234567") is not None
        assert pattern.search("+65 91234567") is not None

    def test_with_0065_prefix(self):
        pattern = get_compiled_pattern("sg_phone")
        assert pattern.search("006591234567") is not None

    def test_with_65_prefix(self):
        pattern = get_compiled_pattern("sg_phone")
        assert pattern.search("6591234567") is not None

    def test_mobile_numbers_starting_with_8_or_9(self):
        pattern = get_compiled_pattern("sg_phone")
        assert pattern.search("+6581234567") is not None  # 8xxx
        assert pattern.search("+6591234567") is not None  # 9xxx

    def test_landline_starting_with_6(self):
        pattern = get_compiled_pattern("sg_phone")
        assert pattern.search("+6561234567") is not None  # 6xxx

    def test_invalid_first_digit(self):
        pattern = get_compiled_pattern("sg_phone")
        # Singapore numbers start with 6, 8, or 9
        assert pattern.search("+6511234567") is None
        assert pattern.search("+6521234567") is None


class TestSingaporePostalCode:
    """Test Singapore postal code detection (contextual pattern)"""

    def test_valid_postal_codes(self):
        pattern = get_compiled_pattern("sg_postal_code")
        assert pattern.search("018956") is not None  # CBD
        assert pattern.search("520123") is not None  # HDB
        assert pattern.search("119077") is not None  # NUS area
        assert pattern.search("800123") is not None  # High range

    def test_invalid_starting_digit(self):
        pattern = get_compiled_pattern("sg_postal_code")
        assert pattern.search("918956") is None  # 9xxxxx invalid


class TestSingaporePassport:
    """Test Singapore passport number detection"""

    def test_e_series_passport(self):
        pattern = get_compiled_pattern("passport_singapore")
        assert pattern.search("E1234567") is not None

    def test_k_series_passport(self):
        pattern = get_compiled_pattern("passport_singapore")
        assert pattern.search("K9876543") is not None

    def test_wrong_prefix_rejected(self):
        pattern = get_compiled_pattern("passport_singapore")
        assert pattern.search("A1234567") is None
        assert pattern.search("X9876543") is None

    def test_too_few_digits_rejected(self):
        pattern = get_compiled_pattern("passport_singapore")
        assert pattern.search("E123456") is None  # Only 6 digits


class TestSingaporeUEN:
    """Test Singapore Unique Entity Number (UEN) detection"""

    def test_local_company_uen_8digit(self):
        pattern = get_compiled_pattern("sg_uen")
        # 8 digits + 1 letter (local companies)
        assert pattern.search("12345678A") is not None

    def test_local_company_uen_9digit(self):
        pattern = get_compiled_pattern("sg_uen")
        # 9 digits + 1 letter (businesses)
        assert pattern.search("123456789Z") is not None

    def test_roc_uen(self):
        pattern = get_compiled_pattern("sg_uen")
        # T or R + 2 digits + 2 letters + 4 digits + 1 letter
        assert pattern.search("T08LL0001A") is not None
        assert pattern.search("R12AB3456Z") is not None

    def test_lowercase_suffix_detected_case_insensitive(self):
        pattern = get_compiled_pattern("sg_uen")
        assert pattern.search("12345678a") is not None


class TestSingaporeBankAccount:
    """Test Singapore bank account number detection"""

    def test_standard_format(self):
        pattern = get_compiled_pattern("sg_bank_account")
        assert pattern.search("123-45678-9") is not None
        assert pattern.search("001-23456-12") is not None
        assert pattern.search("999-123456-123") is not None

    def test_without_dashes_rejected(self):
        pattern = get_compiled_pattern("sg_bank_account")
        # Pattern requires dash format
        assert pattern.search("12345678901") is None
