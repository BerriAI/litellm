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


# Postgres twin of ``extract_compression_saved_tokens``, for aggregating saved
# tokens without pulling every spend log into Python. Assumes the enclosing
# query aliases "LiteLLM_SpendLogs" as ``sl``. The ``jsonb_typeof(...) =
# 'number'`` guards mirror ``_saved_tokens_or_zero`` (non-numbers, including
# JSON booleans, contribute 0 instead of raising a cast error) and the
# ``jsonb_typeof`` switch on ``guardrail_information`` mirrors the bare-dict
# normalization. Any change to the Python reader has to land here in the same
# commit, or hourly savings will disagree with the daily rollup they are drawn
# against.
COMPRESSION_SAVED_TOKENS_SQL = """
GREATEST(
    trunc(
        CASE
            WHEN jsonb_typeof(sl.metadata -> 'compression_savings' -> 'tokens_saved') = 'number'
            THEN (sl.metadata -> 'compression_savings' ->> 'tokens_saved')::numeric
            ELSE 0
        END
    ),
    0
)
+ COALESCE(
    (
        SELECT SUM(
            GREATEST(
                trunc(
                    CASE
                        WHEN jsonb_typeof(entry -> 'guardrail_response' -> 'tokens_saved') = 'number'
                        THEN (entry -> 'guardrail_response' ->> 'tokens_saved')::numeric
                        ELSE 0
                    END
                ),
                0
            )
        )
        FROM jsonb_array_elements(
            CASE jsonb_typeof(sl.metadata -> 'guardrail_information')
                WHEN 'array' THEN sl.metadata -> 'guardrail_information'
                WHEN 'object' THEN jsonb_build_array(sl.metadata -> 'guardrail_information')
                ELSE '[]'::jsonb
            END
        ) AS entry
        WHERE entry ->> 'guardrail_provider' = 'headroom'
    ),
    0
)
"""
