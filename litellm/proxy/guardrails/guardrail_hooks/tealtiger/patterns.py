"""
Regex-based PII / secrets detection patterns for TealTiger.

47 patterns across five categories: government IDs, financial data,
contact info, network/device identifiers, and credentials/API keys.

These are deterministic regex matches — fast (no ML, no network calls) but
inherently heuristic. Expect some false positives/negatives, especially on
the loosest patterns (e.g. generic credit card, generic driver's license).
Callers can narrow scope via TealEngine(policies=[{"patterns": [...]}])
instead of "all", or override this module's PII_PATTERNS entirely.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

PII_PATTERNS: Final = MappingProxyType(
    {
        # ---------- government / national IDs ----------
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "ssn_no_dash": re.compile(r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b"),
        "itin": re.compile(r"\b9\d{2}-(7\d|8[0-8])-\d{4}\b"),
        "ein": re.compile(r"\b\d{2}-\d{7}\b"),
        "us_passport": re.compile(r"\b[A-Z]{1,2}\d{7,9}\b"),
        "passport_generic_intl": re.compile(r"\b[A-Z]{2}\d{6,9}\b"),
        "drivers_license_generic": re.compile(r"\b[A-Z]{1,2}\d{6,8}\b"),
        "medicare_id_us": re.compile(r"\b\d[A-Z]\d{2}-[A-Z]\d{2}-[A-Z]\d{2}\b"),
        "npi_number": re.compile(r"\b\d{10}\b"),
        "uk_nino": re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b"),
        "date_of_birth": re.compile(r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b"),
        "vin": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
        # ---------- financial ----------
        "credit_card_visa": re.compile(r"\b4\d{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
        "credit_card_mastercard": re.compile(r"\b5[1-5]\d{2}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
        "credit_card_amex": re.compile(r"\b3[47]\d{2}[ -]?\d{6}[ -]?\d{5}\b"),
        "credit_card_discover": re.compile(r"\b6(?:011|5\d{2})[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
        "credit_card_generic": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
        "swift_bic": re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
        "us_bank_routing": re.compile(r"\b\d{9}\b"),
        "us_bank_account": re.compile(r"\b\d{8,17}\b"),
        "bitcoin_address": re.compile(r"\b(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b"),
        "ethereum_address": re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
        # ---------- contact info ----------
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "phone_intl": re.compile(r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?(?:[-.\s]?\d{2,4}){2,4}\b"),
        "us_zip_plus4": re.compile(r"\b\d{5}-\d{4}\b"),
        "po_box": re.compile(r"\bP\.?O\.?\s?Box\s?\d+\b", re.IGNORECASE),
        "lat_long_coordinates": re.compile(r"\b-?\d{1,3}\.\d{3,},\s?-?\d{1,3}\.\d{3,}\b"),
        # ---------- network / device identifiers ----------
        "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "ipv6": re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"),
        "mac_address": re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
        "imei": re.compile(r"\b\d{15}\b"),
        "url_with_credentials": re.compile(r"\bhttps?://[^:\s]+:[^@\s]+@[^\s]+\b"),
        # ---------- credentials / API keys / secrets ----------
        "aws_access_key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        "aws_secret_key": re.compile(r"\b(?=.*[A-Za-z])(?=.*\d)(?=.*[+/])[A-Za-z0-9+/]{40}\b"),
        "gcp_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "azure_client_secret": re.compile(r"\b[A-Za-z0-9_~.-]{3}\dQ~[A-Za-z0-9_~.-]{31,34}\b"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
        "gitlab_token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20}\b"),
        "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,72}\b"),
        "stripe_live_key": re.compile(r"\b(sk|pk)_live_[A-Za-z0-9]{24,}\b"),
        "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
        "private_key_pem": re.compile(r"-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----"),
        "generic_bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9_\-.]{20,}\b"),
        "basic_auth_header": re.compile(r"\bBasic\s+[A-Za-z0-9+/]{20,}={0,2}\b"),
    }
)


def default_patterns() -> Mapping[str, re.Pattern[str]]:
    """Return the built-in, immutable pattern mapping."""
    return PII_PATTERNS


PATTERN_CATEGORIES: Final = MappingProxyType(
    {
        "government_id": (
            "ssn",
            "ssn_no_dash",
            "itin",
            "ein",
            "us_passport",
            "passport_generic_intl",
            "drivers_license_generic",
            "medicare_id_us",
            "npi_number",
            "uk_nino",
            "date_of_birth",
            "vin",
        ),
        "financial": (
            "credit_card_visa",
            "credit_card_mastercard",
            "credit_card_amex",
            "credit_card_discover",
            "credit_card_generic",
            "iban",
            "swift_bic",
            "us_bank_routing",
            "us_bank_account",
            "bitcoin_address",
            "ethereum_address",
        ),
        "contact": (
            "email",
            "phone_us",
            "phone_intl",
            "us_zip_plus4",
            "po_box",
            "lat_long_coordinates",
        ),
        "network_device": (
            "ipv4",
            "ipv6",
            "mac_address",
            "imei",
            "url_with_credentials",
        ),
        "credentials": (
            "aws_access_key",
            "aws_secret_key",
            "gcp_api_key",
            "azure_client_secret",
            "github_token",
            "gitlab_token",
            "slack_token",
            "stripe_live_key",
            "openai_api_key",
            "jwt",
            "private_key_pem",
            "generic_bearer_token",
            "basic_auth_header",
        ),
    }
)
