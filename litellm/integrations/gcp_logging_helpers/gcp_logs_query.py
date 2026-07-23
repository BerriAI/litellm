"""
GCP Cloud Logging helpers for querying LiteLLM parallel request metrics.

This module provides functions to query GCP Cloud Logging for parallel request
counters that are emitted by parallel_request_limiter_v3.py via print() statements.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from litellm.integrations.gcp_logging_helpers.parallel_requests_metrics import (
    encode_metric_field,
    parse_parallel_requests_metric_log_line,
)

# Try to import google-cloud-logging
try:
    from google.cloud.logging import Client as LoggingClient

    GCP_LOGGING_AVAILABLE = True
except ImportError:
    GCP_LOGGING_AVAILABLE = False
    print(
        "[gcp_logs_query] google-cloud-logging not installed. "
        "Install with: pip install google-cloud-logging"
    )

def get_gcp_logging_client(project_id: Optional[str] = None):
    """
    Get a GCP Cloud Logging client.

    Args:
        project_id: GCP project ID. If not provided, will try to infer from environment.

    Returns:
        LoggingClient or None if not available
    """
    if not GCP_LOGGING_AVAILABLE:
        print(
            "[gcp_logs_query] Cannot create GCP logging client - "
            "google-cloud-logging is not installed"
        )
        return None

    try:
        # The client will use application default credentials (ADC)
        # Ensure GOOGLE_APPLICATION_CREDENTIALS env var is set, or running on GCP
        print("[gcp_logs_query] Creating LoggingClient...")
        client = LoggingClient(project=project_id)
        print("[gcp_logs_query] Successfully created LoggingClient")
        return client
    except Exception as e:
        import traceback
        print(f"[gcp_logs_query] Failed to create GCP logging client: {e}")
        print(f"[gcp_logs_query] Stack trace: {traceback.format_exc()}")
        return None


def parse_metrics_log_line(text_payload: str) -> Optional[Dict]:
    """
    Parse a METRICS log line and extract the fields.

    Args:
        text_payload: The log line text

    Returns:
        Dictionary with parsed fields or None if parsing fails
    """
    return parse_parallel_requests_metric_log_line(text_payload)


async def query_parallel_requests_metrics_last_n_seconds(
    target_timestamp: float,
    project_id: Optional[str] = None,
    api_key_filter: Optional[str] = None,
    key_alias_filter: Optional[str] = None,
    time_window_seconds: int = 5,  # Default 5 seconds for very fast queries
) -> List[Dict]:
    """
    Query GCP Cloud Logging for parallel requests metrics from last N seconds before target.

    This function queries GCP logs for all [METRICS] log entries from the last N seconds
    before the target timestamp, then finds the LATEST entry per API key (the one with
    the highest log timestamp closest to the target timestamp).

    Args:
        target_timestamp: Unix timestamp (seconds since epoch) to query up to
        project_id: GCP project ID
        api_key_filter: Optional API key to filter by (full key string)
        key_alias_filter: Optional key alias to filter by (partial match supported)
        time_window_seconds: Time window to query (default 5 seconds for fast queries)

    Returns:
        List of dictionaries with keys: token, key_alias, current_count, timestamp
        where current_count represents the Redis counter value at the latest log entry
        for each token.
    """
    # Determine project ID FIRST before creating the client
    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if project_id is None:
        print(
            "[gcp_logs_query] No GCP project ID provided. "
            "Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT env var"
        )
        return []

    client = get_gcp_logging_client(project_id)
    if client is None:
        return []

    try:
        # Calculate time window: last N seconds before target_timestamp
        # Query from (target_timestamp - N seconds) to target_timestamp
        end_time = target_timestamp
        start_time = target_timestamp - time_window_seconds  # N seconds

        # Convert to RFC 3339 format for GCP Logging filter
        start_rfc3339 = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()
        end_rfc3339 = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()

        # Build the filter
        # We filter for textPayload containing our METRICS marker
        filter_parts = [
            f'timestamp>="{start_rfc3339}"',
            f'timestamp<="{end_rfc3339}"',
            'textPayload:"[METRICS] Emitting parallel_requests metric"',
        ]

        if api_key_filter:
            # Add token filter - escape quotes for safety
            escaped_token = encode_metric_field(api_key_filter)
            filter_parts.append(f'textPayload:"token={escaped_token}"')

        if key_alias_filter:
            # Add key_alias filter - escape quotes for safety
            # Note: This does partial match since log line contains key_alias=value
            escaped_alias = encode_metric_field(key_alias_filter)
            filter_parts.append(f'textPayload:"key_alias={escaped_alias}"')

        filter_str = " AND ".join(filter_parts)

        print(
            f"[gcp_logs_query] Querying GCP logs from {start_rfc3339} to {end_rfc3339} "
            f"(last {time_window_seconds} second(s) before target)"
        )

        # Execute the query using Client API
        # Limit entries for performance (we only need latest per token)
        MAX_ENTRIES = int(
            os.environ.get("GCP_LOG_QUERY_MAX_ENTRIES", "15000")
        )
        all_entries = []
        for entry in client.list_entries(
            filter_=filter_str,
            order_by="timestamp desc",  # Most recent first (by receipt time)
            page_size=1000,  # Max page size to minimize API calls and avoid 429 rate limits
        ):
            all_entries.append(entry)
            if len(all_entries) >= MAX_ENTRIES:
                print(
                    f"[gcp_logs_query] Hit max entries limit ({MAX_ENTRIES}), stopping"
                )
                break

        print(
            f"[gcp_logs_query] Found {len(all_entries)} log entries in last {time_window_seconds} second(s)"
        )

        # Parse entries and find the LATEST entry per token (highest log timestamp)
        # This represents the Redis counter value closest to the target timestamp
        token_latest_metrics: Dict[str, Dict] = {}

        for entry in all_entries:
            # Client API returns entry objects with payload attribute
            text_payload = None
            if hasattr(entry, "payload") and entry.payload:
                if isinstance(entry.payload, str):
                    text_payload = entry.payload
                elif hasattr(entry.payload, "text"):
                    text_payload = entry.payload.text
            elif hasattr(entry, "text_payload"):
                text_payload = entry.text_payload
            if not text_payload:
                continue

            parsed = parse_metrics_log_line(text_payload)
            if not parsed:
                continue

            token = parsed["token"]
            log_timestamp = parsed["timestamp"]  # This is the timestamp from the log line itself

            # For each token, keep the entry with the HIGHEST timestamp
            # (closest to target_timestamp, representing the most recent Redis counter value)
            if token not in token_latest_metrics:
                token_latest_metrics[token] = {
                    "token": token,
                    "key_alias": parsed["key_alias"] or token[:16] + "...",
                    "current_count": parsed["current_count"],
                    "timestamp": log_timestamp,
                    "operation": parsed["operation"],
                }
            else:
                # Keep the entry with the higher timestamp (more recent)
                if log_timestamp > token_latest_metrics[token]["timestamp"]:
                    token_latest_metrics[token]["current_count"] = parsed["current_count"]
                    token_latest_metrics[token]["timestamp"] = log_timestamp
                    token_latest_metrics[token]["operation"] = parsed["operation"]
                    token_latest_metrics[token]["key_alias"] = parsed["key_alias"] or token[:16] + "..."

        # Convert to list and sort by token for consistent ordering
        results = sorted(token_latest_metrics.values(), key=lambda x: x["token"])

        print(
            f"[gcp_logs_query] Returning {len(results)} unique token metrics "
            f"(latest log entry per token from last 5 minutes)"
        )

        return results

    except Exception as e:
        print(f"[gcp_logs_query] Error querying GCP logs: {e}")
        return []


async def get_concurrent_requests_from_gcp_logs(
    target_timestamp: float,
    project_id: Optional[str] = None,
    api_key_filter: Optional[str] = None,
    key_alias_filter: Optional[str] = None,
    time_window_seconds: int = 60,  # Kept for backward compatibility, not used
) -> Tuple[List[Dict], bool]:
    """
    Get concurrent requests data from GCP logs for a specific timestamp.

    Queries logs from the last 5 minutes before target_timestamp and returns
    the latest log entry per token (representing the Redis counter value).

    Args:
        target_timestamp: Unix timestamp (seconds) to query at
        project_id: GCP project ID
        api_key_filter: Optional API key filter
        key_alias_filter: Optional key alias filter (partial match)
        time_window_seconds: Deprecated, kept for backward compatibility.
                            The function always queries last 5 minutes.

    Returns:
        Tuple of (results_list, success_boolean)
        results_list contains dicts with: token, key_alias, metrics_concurrency, timestamp
        where metrics_concurrency is the Redis counter value from the latest log entry.
    """
    if not GCP_LOGGING_AVAILABLE:
        print(
            "[gcp_logs_query] google-cloud-logging not available. "
            "Cannot query GCP logs for concurrent requests."
        )
        return [], False

    # Query last N seconds and get latest entry per token
    # Default: 60 seconds (1 minute), configurable via GCP_LOG_QUERY_TIME_WINDOW_SECONDS
    time_window = int(
        os.environ.get("GCP_LOG_QUERY_TIME_WINDOW_SECONDS", "60")
    )
    metrics = await query_parallel_requests_metrics_last_n_seconds(
        target_timestamp=target_timestamp,
        project_id=project_id,
        api_key_filter=api_key_filter,
        key_alias_filter=key_alias_filter,
        time_window_seconds=time_window,
    )

    if not metrics:
        return [], True  # Success but no data

    # Transform to the format expected by the endpoint
    # redis_concurrency represents the Redis counter value (current_count from latest log)
    results = []
    for m in metrics:
        results.append({
            "token": m["token"],
            "key_name": m["token"][:16] + "..." if len(m["token"]) > 16 else m["token"],
            "key_alias": m["key_alias"] or "—",
            "redis_concurrency": m["current_count"],  # This is the Redis counter value
            "timestamp": m["timestamp"],  # Timestamp of the latest log entry
        })

    return results, True


async def count_parallel_request_operations(
    start_timestamp: float,
    end_timestamp: float,
    project_id: Optional[str] = None,
    token_filter: Optional[str] = None,
    key_alias_filter: Optional[str] = None,
) -> Tuple[Dict[str, int], bool, bool]:
    """
    Count increment vs decrement [METRICS] parallel_requests log entries within a
    time range, filtered by token (exact) and/or key_alias (partial match).

    These [METRICS] lines are emitted by parallel_request_limiter_v3.py on every
    max_parallel_requests counter increment/decrement and carry token + key_alias
    only (never the masked key) — so callers must resolve a masked api_key to its
    token before filtering here.

    Args:
        start_timestamp: Unix timestamp (seconds) — start of range (inclusive)
        end_timestamp: Unix timestamp (seconds) — end of range (inclusive)
        project_id: GCP project ID (falls back to env)
        token_filter: Exact token to match (textPayload:"token=<token>")
        key_alias_filter: Key alias to match (partial / substring match)

    Returns:
        (counts, success, truncated) where
        counts = {"increment": int, "decrement": int, "total": int},
        success = False if GCP is unavailable or the client could not be created,
        truncated = True if the max-entries cap was hit (counts are a lower bound).
    """
    counts = {"increment": 0, "decrement": 0, "total": 0}

    if not GCP_LOGGING_AVAILABLE:
        print(
            "[gcp_logs_query] google-cloud-logging not available. "
            "Cannot count parallel request operations."
        )
        return counts, False, False

    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if project_id is None:
        print(
            "[gcp_logs_query] No GCP project ID provided. "
            "Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT env var"
        )
        return counts, False, False

    client = get_gcp_logging_client(project_id)
    if client is None:
        return counts, False, False

    try:
        start_rfc3339 = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()
        end_rfc3339 = datetime.fromtimestamp(end_timestamp, tz=timezone.utc).isoformat()

        filter_parts = [
            f'timestamp>="{start_rfc3339}"',
            f'timestamp<="{end_rfc3339}"',
            'textPayload:"[METRICS] Emitting parallel_requests metric"',
        ]
        if token_filter:
            escaped_token = encode_metric_field(token_filter)
            filter_parts.append(f'textPayload:"token={escaped_token}"')
        if key_alias_filter:
            escaped_alias = encode_metric_field(key_alias_filter)
            filter_parts.append(f'textPayload:"key_alias={escaped_alias}"')
        filter_str = " AND ".join(filter_parts)

        # Bound the scan so a wide range can't page forever. Counts beyond this are
        # reported as truncated (a lower bound).
        max_entries = int(os.environ.get("GCP_LOG_COUNT_MAX_ENTRIES", "50000"))

        print(
            f"[gcp_logs_query] Counting parallel_requests operations from "
            f"{start_rfc3339} to {end_rfc3339} "
            f"(token_filter={'set' if token_filter else 'none'}, "
            f"key_alias_filter={'set' if key_alias_filter else 'none'})"
        )

        truncated = False
        scanned = 0
        for entry in client.list_entries(
            filter_=filter_str,
            order_by="timestamp desc",
            page_size=1000,
        ):
            scanned += 1
            if scanned > max_entries:
                truncated = True
                print(
                    f"[gcp_logs_query] Hit max entries limit ({max_entries}) "
                    f"while counting, stopping"
                )
                break

            text_payload = None
            if hasattr(entry, "payload") and entry.payload:
                if isinstance(entry.payload, str):
                    text_payload = entry.payload
                elif hasattr(entry.payload, "text"):
                    text_payload = entry.payload.text
            elif hasattr(entry, "text_payload"):
                text_payload = entry.text_payload
            if not text_payload:
                continue

            parsed = parse_metrics_log_line(text_payload)
            if not parsed:
                continue

            operation = parsed.get("operation")
            if operation == "increment":
                counts["increment"] += 1
            elif operation == "decrement":
                counts["decrement"] += 1

        counts["total"] = counts["increment"] + counts["decrement"]
        print(
            f"[gcp_logs_query] Counted increment={counts['increment']} "
            f"decrement={counts['decrement']} truncated={truncated}"
        )
        return counts, True, truncated

    except Exception as e:
        print(f"[gcp_logs_query] Error counting parallel request operations: {e}")
        return counts, False, False


async def get_parallel_request_metric_series(
    start_timestamp: float,
    end_timestamp: float,
    project_id: Optional[str] = None,
    token_filter: Optional[str] = None,
    key_alias_filter: Optional[str] = None,
) -> Tuple[List[Dict], bool]:
    """
    Return all [METRICS] parallel_requests log entries within [start, end] for a
    token / key_alias, sorted ascending by the log entry's receive time.

    Used to reconstruct the Redis counter value at each point in a time window (e.g.
    once per minute): for a given target time T, the counter value is the
    current_count of the latest entry whose entry_time <= T.

    Note: the payload's own `timestamp=` field is an unreliable wall-clock isoformat
    (not a unix float), so we use the GCP entry's receive timestamp for ordering.

    Returns:
        (entries, success) where each entry is
        {"entry_time": float (unix seconds), "current_count": int,
         "operation": str, "token": str, "key_alias": Optional[str]}.
    """
    entries: List[Dict] = []

    if not GCP_LOGGING_AVAILABLE:
        print(
            "[gcp_logs_query] google-cloud-logging not available. "
            "Cannot fetch parallel request metric series."
        )
        return entries, False

    if project_id is None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if project_id is None:
        print(
            "[gcp_logs_query] No GCP project ID provided. "
            "Set GOOGLE_CLOUD_PROJECT or GCP_PROJECT env var"
        )
        return entries, False

    client = get_gcp_logging_client(project_id)
    if client is None:
        return entries, False

    try:
        start_rfc3339 = datetime.fromtimestamp(start_timestamp, tz=timezone.utc).isoformat()
        end_rfc3339 = datetime.fromtimestamp(end_timestamp, tz=timezone.utc).isoformat()

        filter_parts = [
            f'timestamp>="{start_rfc3339}"',
            f'timestamp<="{end_rfc3339}"',
            'textPayload:"[METRICS] Emitting parallel_requests metric"',
        ]
        if token_filter:
            escaped_token = encode_metric_field(token_filter)
            filter_parts.append(f'textPayload:"token={escaped_token}"')
        if key_alias_filter:
            escaped_alias = encode_metric_field(key_alias_filter)
            filter_parts.append(f'textPayload:"key_alias={escaped_alias}"')
        filter_str = " AND ".join(filter_parts)

        max_entries = int(os.environ.get("GCP_LOG_COUNT_MAX_ENTRIES", "50000"))

        print(
            f"[gcp_logs_query] Fetching parallel_requests metric series from "
            f"{start_rfc3339} to {end_rfc3339}"
        )

        scanned = 0
        for entry in client.list_entries(
            filter_=filter_str,
            order_by="timestamp asc",
            page_size=1000,
        ):
            scanned += 1
            if scanned > max_entries:
                print(
                    f"[gcp_logs_query] Hit max entries limit ({max_entries}) "
                    f"while building metric series, stopping"
                )
                break

            entry_time = getattr(entry, "timestamp", None)
            if entry_time is None:
                continue

            text_payload = None
            if hasattr(entry, "payload") and entry.payload:
                if isinstance(entry.payload, str):
                    text_payload = entry.payload
                elif hasattr(entry.payload, "text"):
                    text_payload = entry.payload.text
            elif hasattr(entry, "text_payload"):
                text_payload = entry.text_payload
            if not text_payload:
                continue

            parsed = parse_metrics_log_line(text_payload)
            if not parsed:
                continue

            entries.append({
                "entry_time": entry_time.timestamp(),
                "current_count": parsed["current_count"],
                "operation": parsed["operation"],
                "token": parsed["token"],
                "key_alias": parsed["key_alias"],
            })

        # Already ascending by order_by, but sort defensively.
        entries.sort(key=lambda x: x["entry_time"])

        print(f"[gcp_logs_query] Returning {len(entries)} metric series entries")
        return entries, True

    except Exception as e:
        print(f"[gcp_logs_query] Error fetching parallel request metric series: {e}")
        return entries, False
