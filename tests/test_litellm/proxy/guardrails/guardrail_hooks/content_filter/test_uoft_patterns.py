"""
Test University of Toronto identifier regex patterns added for FIPPA compliance.

Tests UTORid, student/employee number, and TCard number detection patterns.
"""

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    get_compiled_pattern,
)


class TestUofTStudentId:
    """Test University of Toronto Student/Employee Number detection"""

    def test_standard_student_number(self):
        pattern = get_compiled_pattern("uoft_student_id")
        assert pattern.search("1012345678") is not None

    def test_another_valid_number(self):
        pattern = get_compiled_pattern("uoft_student_id")
        assert pattern.search("1099999999") is not None

    def test_in_sentence(self):
        pattern = get_compiled_pattern("uoft_student_id")
        assert (
            pattern.search("My student number is 1012345678 for registration")
            is not None
        )

    def test_prefix_must_be_10(self):
        """All UofT numbers start with 10"""
        pattern = get_compiled_pattern("uoft_student_id")
        assert pattern.search("2012345678") is None

    def test_too_few_digits_rejected(self):
        pattern = get_compiled_pattern("uoft_student_id")
        assert pattern.search("101234567") is None  # 9 digits

    def test_too_many_digits_rejected(self):
        pattern = get_compiled_pattern("uoft_student_id")
        assert pattern.search("10123456789") is None  # 11 digits

    def test_non_numeric_rejected(self):
        pattern = get_compiled_pattern("uoft_student_id")
        assert pattern.search("10abcdefgh") is None


class TestUofTUTORid:
    """Test University of Toronto UTORid detection"""

    def test_standard_utorid(self):
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("smithj12") is not None

    def test_short_name_prefix(self):
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("li5") is not None

    def test_longer_name_prefix(self):
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("kcheng42") is not None

    def test_max_length_prefix(self):
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("abcdef1234") is not None

    def test_in_sentence(self):
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("My UTORid is smithj12 for ACORN login") is not None

    def test_single_letter_prefix_rejected(self):
        """Minimum 2 letters required"""
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("a1") is None

    def test_too_many_letters_rejected(self):
        """Maximum 6 letters"""
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("abcdefg1") is None

    def test_no_digits_rejected(self):
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("smith") is None

    def test_too_many_digits_rejected(self):
        """Maximum 4 digits"""
        pattern = get_compiled_pattern("uoft_utorid")
        assert pattern.search("ab12345") is None


class TestUofTTCard:
    """Test University of Toronto TCard number detection"""

    def test_standard_tcard(self):
        pattern = get_compiled_pattern("uoft_tcard")
        assert pattern.search("1234567890123456") is not None

    def test_in_sentence(self):
        pattern = get_compiled_pattern("uoft_tcard")
        assert (
            pattern.search("My TCard number is 1234567890123456 for library access")
            is not None
        )

    def test_too_few_digits_rejected(self):
        pattern = get_compiled_pattern("uoft_tcard")
        assert pattern.search("123456789012345") is None  # 15 digits

    def test_too_many_digits_rejected(self):
        pattern = get_compiled_pattern("uoft_tcard")
        assert pattern.search("12345678901234567") is None  # 17 digits

    def test_non_numeric_rejected(self):
        pattern = get_compiled_pattern("uoft_tcard")
        assert pattern.search("123456789012345a") is None
