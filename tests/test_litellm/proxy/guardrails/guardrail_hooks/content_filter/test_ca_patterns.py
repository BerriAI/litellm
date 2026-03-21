"""
Test Canadian PII regex patterns added for PIPEDA compliance.

Tests SIN, OHIP, Ontario driver's licence, immigration documents,
bank account, and postal code detection patterns.
"""

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    get_compiled_pattern,
)


class TestCanadianSIN:
    """Test Canadian Social Insurance Number detection"""

    def test_dashed_format(self):
        pattern = get_compiled_pattern("ca_sin")
        assert pattern.search("123-456-789") is not None

    def test_spaced_format(self):
        pattern = get_compiled_pattern("ca_sin")
        assert pattern.search("123 456 789") is not None

    def test_sin_in_sentence(self):
        pattern = get_compiled_pattern("ca_sin")
        assert pattern.search("My SIN is 987-654-321 for tax") is not None

    def test_compact_nine_digits_not_matched(self):
        """Compact 9-digit format (no dashes/spaces) should NOT match the dashed pattern"""
        pattern = get_compiled_pattern("ca_sin")
        assert pattern.search("123456789") is None

    def test_too_few_digits_rejected(self):
        pattern = get_compiled_pattern("ca_sin")
        assert pattern.search("12-345-678") is None

    def test_too_many_digits_rejected(self):
        pattern = get_compiled_pattern("ca_sin")
        assert pattern.search("1234-567-890") is None


class TestCanadianOHIP:
    """Test Ontario Health Insurance Plan Number detection"""

    def test_full_format_with_version_code(self):
        pattern = get_compiled_pattern("ca_ohip")
        assert pattern.search("1234-567-890-AB") is not None

    def test_compact_with_version_code(self):
        pattern = get_compiled_pattern("ca_ohip")
        assert pattern.search("1234567890AB") is not None

    def test_spaced_format(self):
        pattern = get_compiled_pattern("ca_ohip")
        assert pattern.search("1234 567 890 XY") is not None

    def test_ohip_in_sentence(self):
        pattern = get_compiled_pattern("ca_ohip")
        assert (
            pattern.search("My OHIP number is 9876543210ZZ for my appointment")
            is not None
        )

    def test_without_version_code_not_matched(self):
        """OHIP pattern requires the 2-letter version code"""
        pattern = get_compiled_pattern("ca_ohip")
        # 10 digits alone without letters should not match the full OHIP pattern
        assert pattern.search("1234567890") is None

    def test_lowercase_version_code_detected(self):
        """Compiled with IGNORECASE"""
        pattern = get_compiled_pattern("ca_ohip")
        assert pattern.search("1234567890ab") is not None


class TestCanadianOntarioDriversLicence:
    """Test Ontario Driver's Licence detection"""

    def test_dashed_format(self):
        pattern = get_compiled_pattern("ca_on_drivers_licence")
        assert pattern.search("A1234-56789-01234") is not None

    def test_spaced_format(self):
        pattern = get_compiled_pattern("ca_on_drivers_licence")
        assert pattern.search("B9876 54321 09876") is not None

    def test_in_sentence(self):
        pattern = get_compiled_pattern("ca_on_drivers_licence")
        assert (
            pattern.search("Driver's licence C1111-22222-33333 on file") is not None
        )

    def test_compact_format_not_matched(self):
        """Compact format (no dashes/spaces) uses a separate pattern"""
        pattern = get_compiled_pattern("ca_on_drivers_licence")
        assert pattern.search("A12345678901234") is None

    def test_missing_letter_prefix_rejected(self):
        pattern = get_compiled_pattern("ca_on_drivers_licence")
        assert pattern.search("11234-56789-01234") is None

    def test_wrong_digit_groups_rejected(self):
        pattern = get_compiled_pattern("ca_on_drivers_licence")
        assert pattern.search("A123-456789-01234") is None


class TestCanadianImmigrationDoc:
    """Test Canadian IRCC Immigration Document detection"""

    def test_imm_document_reference(self):
        pattern = get_compiled_pattern("ca_immigration_doc")
        assert pattern.search("IMM-5257") is not None
        assert pattern.search("IMM 1234") is not None
        assert pattern.search("IMM5257") is not None

    def test_work_study_permit(self):
        pattern = get_compiled_pattern("ca_immigration_doc")
        assert pattern.search("T123456789") is not None
        assert pattern.search("F1234567890") is not None
        assert pattern.search("W12345678") is not None

    def test_uci_dashed(self):
        pattern = get_compiled_pattern("ca_immigration_doc")
        assert pattern.search("1234-5678-90") is not None

    def test_uci_compact_rejected(self):
        """Compact 10-digit UCI without separators should NOT match (too broad)"""
        pattern = get_compiled_pattern("ca_immigration_doc")
        assert pattern.search("1234567890") is None

    def test_imm_in_sentence(self):
        pattern = get_compiled_pattern("ca_immigration_doc")
        assert (
            pattern.search("Submit immigration form IMM-5645 with your application")
            is not None
        )

    def test_too_short_imm_rejected(self):
        pattern = get_compiled_pattern("ca_immigration_doc")
        assert pattern.search("IMM-12") is None  # Less than 4 digits


class TestCanadianBankAccount:
    """Test Canadian Bank Account routing detection"""

    def test_standard_format_dashed(self):
        pattern = get_compiled_pattern("ca_bank_account")
        assert pattern.search("12345-003-1234567") is not None

    def test_spaced_format(self):
        pattern = get_compiled_pattern("ca_bank_account")
        assert pattern.search("00456 001 9876543210") is not None

    def test_longer_account_number(self):
        pattern = get_compiled_pattern("ca_bank_account")
        assert pattern.search("12345-003-123456789012") is not None

    def test_in_sentence(self):
        pattern = get_compiled_pattern("ca_bank_account")
        assert (
            pattern.search("Direct deposit to bank account 12345-003-1234567 please")
            is not None
        )

    def test_without_separators_rejected(self):
        pattern = get_compiled_pattern("ca_bank_account")
        assert pattern.search("123450031234567") is None


class TestCanadianPostalCode:
    """Test Canadian Postal Code detection"""

    def test_spaced_format(self):
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("M5V 2T6") is not None
        assert pattern.search("K1A 0B1") is not None
        assert pattern.search("V6B 3K9") is not None

    def test_compact_format(self):
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("M5V2T6") is not None
        assert pattern.search("K1A0B1") is not None

    def test_dashed_format(self):
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("M5V-2T6") is not None

    def test_in_sentence(self):
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("Ship to postal code M5V 2T6 in Toronto") is not None

    def test_invalid_first_letter_rejected(self):
        """Letters D, F, I, O, Q, U are not valid as first character"""
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("D5V 2T6") is None
        assert pattern.search("F1A 0B1") is None
        assert pattern.search("I5V 2T6") is None
        assert pattern.search("O1A 0B1") is None
        assert pattern.search("Q5V 2T6") is None
        assert pattern.search("U1A 0B1") is None

    def test_lowercase_detected(self):
        """Compiled with IGNORECASE"""
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("m5v 2t6") is not None

    def test_all_digits_rejected(self):
        pattern = get_compiled_pattern("ca_postal_code")
        assert pattern.search("123 456") is None
