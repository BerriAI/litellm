"""
Single chokepoint for reading prompt-compression token savings out of a parsed
SpendLog ``metadata`` JSON dict. Imported by the daily-spend DB writer and by
cost-savings read endpoints.
"""

from collections.abc import Mapping

HEADROOM_GUARDRAIL_PROVIDER = "headroom"


def _saved_tokens_or_zero(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if value < 0:
        return 0
    return int(value)


def _tokens_saved_from_stats(stats: object) -> int:
    if not isinstance(stats, Mapping):
        return 0
    return _saved_tokens_or_zero(stats.get("tokens_saved"))


def _headroom_entry_saved_tokens(entry: object) -> int:
    if not isinstance(entry, Mapping):
        return 0
    if entry.get("guardrail_provider") != HEADROOM_GUARDRAIL_PROVIDER:
        return 0
    return _tokens_saved_from_stats(entry.get("guardrail_response"))


def _headroom_saved_tokens(guardrail_information: object) -> int:
    entries = [guardrail_information] if isinstance(guardrail_information, Mapping) else guardrail_information
    if not isinstance(entries, list):
        return 0
    return sum(_headroom_entry_saved_tokens(entry) for entry in entries)


def extract_compression_saved_tokens(metadata: Mapping[str, object]) -> int:
    """
    Return the total prompt tokens saved by compression for one request.

    Sums two disjoint sources:

    - the native ``compression_savings`` key, written only by
      ``CompressionInterceptionLogger`` in its pre-call deployment hook
    - ``guardrail_information`` entries with ``guardrail_provider ==
      "headroom"``, written only by the Headroom guardrail

    Each writer records only its own transform pass and the two run at
    different stages (guardrail pre-call vs deployment pre-call), so when both
    fire on one request their measured savings are independent and additive;
    summing them never double-counts. Malformed or missing values contribute 0.
    A bare dict ``guardrail_information`` is treated as a single entry, matching
    the spend-log redactor's normalization of that legacy shape.
    """
    return _tokens_saved_from_stats(metadata.get("compression_savings")) + _headroom_saved_tokens(
        metadata.get("guardrail_information")
    )
