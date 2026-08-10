"""
Text redaction utility for the GCS logger.

This module uses the existing guardrail regex patterns (loaded from
``litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns``)
plus a small set of supplemental regexes defined here to redact PII,
credentials, network identifiers, and payment card data from text payloads
before they are uploaded to GCS.

Patterns are compiled once at module import time (lazy compilation: the work
happens the first time the module is imported, then reused for every
subsequent call). Categories that are intended for *blocking* (for example
"Dangerous Content*") are intentionally excluded from redaction.

Redaction is opt-IN and disabled by default. Set the ``GCS_REDACT_PII``
environment variable to any value other than ``"false"``, ``"0"``,
``"no"``, or ``"off"`` (case-insensitive) to enable it; :func:`redact_text`
and :func:`redact_messages` become no-ops when disabled.
"""

import logging
import os
import re
from typing import Any, List, Optional, Pattern, Tuple

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    PATTERN_CATEGORIES,
    PREBUILT_PATTERNS,
)


# ---------------------------------------------------------------------------
# Environment toggle
# ---------------------------------------------------------------------------
# Read once at module import time. Default is OFF — set GCS_REDACT_PII to any
# truthy value (anything other than "false"/"0"/"no"/"off") to enable.
_REDACT_ENABLED_RAW = os.getenv("GCS_REDACT_PII", "false")
REDACT_ENABLED: bool = _REDACT_ENABLED_RAW.strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
)


# Categories whose matches should be redacted from outbound logs.
#
# ORDER MATTERS: patterns from earlier categories are applied first, so the
# most specific patterns must come first to avoid looser patterns gobbling
# fragments of a specific-pattern match. Specifically:
#
#   * ``Credential Patterns`` comes first so that GitHub/Slack/AWS tokens
#     and private keys are captured whole before a broader pattern can
#     step on a fragment of them.
#   * ``Network Patterns`` comes before ``Payment Card Patterns`` so that
#     an IPv4 octet is not mis-flagged as a ``cvv_contextual`` 3-4 digit
#     run.
REDACTION_CATEGORIES: List[str] = [
    # Specific secrets first - the patterns here are anchored enough that
    # they only fire on real credentials and never fragment neighbors.
    "Credential Patterns",
    # Network identifiers are tightly shaped (dotted quads, MAC-form);
    # let them claim their targets before any looser pattern can shatter
    # them.
    "Network Patterns",
    # Payment Card Patterns comes BEFORE phone/CPF patterns because the
    # contextual card regex ``\d[\s%\-]?{12,18}\d`` is fragmentable by
    # phone-shaped patterns if they run first.
    "Payment Card Patterns",
    # Financial PII (account numbers, routing, etc.) - similar logic, must
    # fire before the looser phone/CPF patterns.
    "Financial PII Patterns",
    # Generic PII (SSN, passport, DL) - mostly shape-specific, safe above.
    "PII Patterns",
    # Region-specific patterns last. Many of these have looser regexes
    # (e.g. Brazilian phone has no word boundary and matches arbitrary
    # 8-9 digit runs, Canadian FIPPA patterns are English words).
    "EU PII Patterns",
    "Indian PII Patterns",
    "Singapore PII Patterns",
    "Brazilian PII Patterns",
    "Canadian PII Patterns",
]


# Specific prebuilt pattern names to skip because their upstream regex is
# context-gated (they require a nearby keyword like "cvv"/"card"/"account")
# but the keyword gate is NOT available in a pure-redaction path. Applying
# them bare would shred IPv4 octets, short numeric IDs, MAC fragments, etc.
SKIPPED_PATTERN_NAMES: frozenset = frozenset(
    {
        # r"\b\d{3,4}\b" - matches every 3- or 4-digit run.
        "cvv_contextual",
        # r"\b(?:0[1-9]|1[0-2])\b" - matches every MM-like 2-digit run.
        "card_expiry_month_contextual",
        # r"\b(?:\d{2}|\d{4})\b" - matches every 2- or 4-digit run
        # (i.e., every year, every MM/DD/yy prefix, etc.).
        "card_expiry_year_contextual",
        # r"\b\d{9,18}\b" - matches many legitimate IDs, timestamps, etc.
        # The contextual keyword gate (``account``/``acct``/etc.) is the
        # only thing keeping this one from being a firehose; drop it here.
        "bank_account_number_contextual",
        # r"\b[1-9][0-9]{5}\b" - matches every 6-digit number; shreds
        # zip codes, order IDs, timestamps, etc.
        "otp_contextual",
        # Phone-shaped: matches ``2025-08-05`` (ISO date), ``08-05-2025``
        # (DD-MM-YYYY), and any dash/space-separated 10-14 digit run.
        # Without its keyword gate this redacts every date in every log
        # line.
        "phone_number_international_contextual",
        # Card-expiry-shaped: matches ``08-05`` (MM-DD) and ``08/2025``
        # (MM-YYYY) without any card context. Eats every date fragment.
        "card_expiry_date_contextual",
        # Broad 40-char base64 — matches git SHAs, package hashes, tokens.
        "aws_secret_key",
        # "(?<!\\d)(?:\\d[\\s%\\-]?){12,18}\\d(?!\\d)" - any 13-19 digit run.
        # Matches epoch millis, K8s pod IDs, timestamps. The actual
        # card-specific shapes (visa/mc/amex/discover) remain active.
        "payment_card_number_contextual",
        # Bare 9-digit SSN. Shreds Jira IDs, timestamps, port numbers.
        "us_ssn_no_dash",
        # Any 10-digit phone-looking run; no anchor.
        "us_phone",
        # Passport-shaped but really just "any 9 digits" or "A+8digits".
        "passport_us",
        "passport_uk",
        "passport_germany",
        "passport_france",
        "passport_netherlands",
        "passport_canada",
        "passport_india",
        "passport_australia",
        "passport_china",
        "passport_japan",
        # Bare 9 digits, 8-9 digits, and digit-runs. Shred IDs everywhere.
        "nl_bsn_contextual",
        "au_tfn",
        "au_abn",
        "au_medicare",
        # 15-digit run starting with 1 or 2 — matches epoch timestamps.
        "fr_nir",
        # IBAN-enhanced — matches 13+ char base64/alphanumeric tokens.
        "eu_iban_enhanced",
        # French phone, mostly `0[1-9][0-9]{8}` — collides with 10-digit runs.
        "fr_phone",
        # 2-letter country code + 8-12 alnum. Matches tool/deployment IDs.
        "eu_vat",
        # Generic "NNLLNNNNN" shape. Matches config keys.
        "eu_passport_generic",
        # Bare 5-digit run. Shreds port numbers, build numbers, ZIP codes.
        "fr_postal_code",
        # 5 letters + 4 digits + letter. Matches tool/class names like
        # "READX1X", "ABCD4Z".
        "in_pan",
        # 15-char alphanumeric with embedded 'Z'. Matches version strings:
        # "18.0.0+sdk@1.0.0+json-..." produced false positives in production.
        "in_gstin",
        # Singapore 6-digit postal — collides with short hashes.
        "sg_postal_code",
        # "[STFGM]dddddddL" — matches tool IDs.
        "sg_nric",
        # "E" or "K" + 7 digits. Matches env var names.
        "passport_singapore",
        # Broad alphanum with optional letter suffix.
        "sg_uen",
        # Bare 11 digits — matches timestamps.
        "br_cpf_unformatted",
        # Loose Brazilian phone shapes. Match any 8-10 digit run.
        "br_phone_landline",
        "br_phone_mobile",
        # 8-digit run.
        "br_cep",
        # Broad Canadian patterns.
        "ca_ohip",
        "ca_on_drivers_licence",
        "ca_immigration_doc",
        "ca_bank_account",
    }
)


# Stable mapping from the (human-facing) pattern category name to the label
# that will appear inside the ``[REDACTED_<LABEL>]`` placeholder. Keeping this
# explicit avoids accidental leakage of raw category strings (which can
# contain spaces or parentheses) into redacted output.
_CATEGORY_LABEL_MAP = {
    "PII Patterns": "PII",
    "Credential Patterns": "CREDENTIALS",
    "Payment Card Patterns": "PAYMENT_CARD",
    "Financial PII Patterns": "FINANCIAL_PII",
    "EU PII Patterns": "EU_PII",
    "Network Patterns": "NETWORK",
    "Indian PII Patterns": "INDIAN_PII",
    "Singapore PII Patterns": "SINGAPORE_PII",
    "Brazilian PII Patterns": "BRAZILIAN_PII",
    "Canadian PII Patterns": "CANADIAN_PII",
}


def _category_to_label(category: str) -> str:
    """Return the redaction label used in the ``[REDACTED_<LABEL>]`` placeholder."""
    if category in _CATEGORY_LABEL_MAP:
        return _CATEGORY_LABEL_MAP[category]
    # Fallback: strip characters that are awkward inside ``[REDACTED_*]``.
    cleaned = (
        category.upper()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    return cleaned


# Supplemental patterns that are NOT present upstream but fill known gaps
# (JWT, UK NI, IFSC, full MAC, full OpenAI key, full GitHub token).
#
# Each entry is ``(compiled_pattern, replacement_label)``. These are applied
# AFTER the upstream prebuilt patterns to catch residuals they missed; any
# overlap is safe because applying two patterns to the same substring just
# results in nested markers being left in place.
_SUPPLEMENTAL_PATTERNS: List[Tuple[Pattern[str], str]] = [
    # JSON Web Token: header.payload.signature (base64url, three segments).
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
        "[REDACTED_CREDENTIALS]",
    ),
    # OpenAI API keys: ``sk-...``, ``sk-proj-...``, ``sk-ant-...``. Whole
    # string -- catches what ``api_key_openai`` missed when prefixed with
    # e.g. ``sk-proj-``.
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "[REDACTED_CREDENTIALS]",
    ),
    # Google API keys: ``AIza`` + 35 chars. The upstream
    # ``generic_api_key`` pattern only fires on ``API_KEY=value`` shaped
    # text, so a bare Google API key value (especially after ``"..."``)
    # would otherwise leak through.
    (
        re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
        "[REDACTED_CREDENTIALS]",
    ),
    # GitHub tokens: ``ghp_/gho_/ghu_/ghs_/ghr_`` + 36 alnum, plus new-style
    # ``github_pat_...``. The prebuilt regex misses some lengths; this
    # catches the full token.
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED_CREDENTIALS]",
    ),
    # Slack tokens: ``xox[pboars]-...`` full form (catches segment count
    # variations the prebuilt regex can miss).
    (
        re.compile(r"\bxox[pboars]-[A-Za-z0-9-]{10,}\b"),
        "[REDACTED_CREDENTIALS]",
    ),
    # UK National Insurance Number: 2 letters, 6 digits, optional A/B/C/D.
    (
        re.compile(
            r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]?\b"
        ),
        "[REDACTED_EU_PII]",
    ),
    # India IFSC: 4 letters + 0 + 6 alphanumeric.
    (
        re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        "[REDACTED_INDIAN_PII]",
    ),
    # Full MAC address (6 octets colon- or dash-separated). The prebuilt
    # ``mac_address`` pattern only catches the first octet.
    (
        re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
        "[REDACTED_NETWORK]",
    ),
]


def _build_compiled_patterns() -> List[Tuple[Pattern[str], str]]:
    """
    Compile every pattern from the configured redaction categories.

    Each entry is a ``(compiled_regex, label)`` tuple. Patterns that fail to
    compile are skipped so a malformed pattern can never break the redaction
    pipeline entirely.

    Category order (from :data:`REDACTION_CATEGORIES`) is honoured - earlier
    categories are compiled earlier, so more-specific patterns get applied
    first.
    """
    compiled: List[Tuple[Pattern[str], str]] = []
    seen_pattern_ids = set()

    for category in REDACTION_CATEGORIES:
        pattern_names = PATTERN_CATEGORIES.get(category) or []
        label = _category_to_label(category)
        replacement = f"[REDACTED_{label}]"

        for pattern_name in pattern_names:
            if pattern_name in SKIPPED_PATTERN_NAMES:
                continue

            raw_pattern = PREBUILT_PATTERNS.get(pattern_name)
            if raw_pattern is None:
                continue

            # Guard against duplicate regex strings being compiled more than
            # once when the same pattern appears under multiple categories.
            pattern_id = (raw_pattern, replacement)
            if pattern_id in seen_pattern_ids:
                continue
            seen_pattern_ids.add(pattern_id)

            try:
                compiled.append(
                    (re.compile(raw_pattern, re.IGNORECASE), replacement)
                )
            except re.error as exc:  # pragma: no cover - defensive
                logging.getLogger(__name__).warning(
                    "Skipping uncompilable pattern %s: %s", pattern_name, exc
                )

    # Append supplemental patterns last. They are for catching residual
    # values the upstream prebuilt patterns missed.
    compiled.extend(_SUPPLEMENTAL_PATTERNS)

    return compiled


# Module-level cache: compiled once on first import and reused for every
# subsequent call. This is the "lazy compilation" requirement - the cost is
# paid exactly once, the first time anything in this module is imported.
_COMPILED_REDACTION_PATTERNS: List[Tuple[Pattern[str], str]] = (
    _build_compiled_patterns()
)


def _apply_redactions(text: str) -> str:
    """Apply every compiled redaction pattern to ``text``."""
    for pattern, replacement in _COMPILED_REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_text(text: Any) -> Any:
    """
    Redact PII, credentials, and payment card data from a string.

    Non-string inputs (including ``None``) and empty strings are returned
    unchanged so callers can pass arbitrary fields through without
    pre-validating their type.

    If redaction is disabled via ``GCS_REDACT_PII`` this is a no-op.
    """
    if not REDACT_ENABLED:
        return text
    if text is None:
        return text
    if not isinstance(text, str):
        return text
    if not text:
        return text
    return _apply_redactions(text)


def _redact_value(value: Any) -> Any:
    """Recursively walk ``value`` and redact any strings found inside it."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def redact_dict_values(value: Any) -> Any:
    if not REDACT_ENABLED:
        return value
    return _redact_value(value)


def redact_messages(messages: Optional[List[Any]]) -> Optional[List[Any]]:
    """
    Redact content inside a list of OpenAI-style message dicts.

    Each message is expected to look like::

        {"role": "user", "content": "..."}

    but the function is defensive about shape:

    * ``content`` may be a string, a list of content parts, a nested dict,
      or any other JSON-compatible value.
    * Non-dict messages are passed through unchanged.
    * ``None`` and non-list inputs are returned unchanged.

    If redaction is disabled via ``GCS_REDACT_PII`` this is a no-op.
    """
    if not REDACT_ENABLED:
        return messages
    if messages is None:
        return messages
    if not isinstance(messages, list):
        return messages

    redacted: List[Any] = []
    for message in messages:
        if isinstance(message, dict):
            redacted.append(_redact_value(message))
        else:
            redacted.append(message)
    return redacted


__all__ = [
    "REDACT_ENABLED",
    "REDACTION_CATEGORIES",
    "SKIPPED_PATTERN_NAMES",
    "redact_text",
    "redact_dict_values",
    "redact_messages",
]
