"""
GCP Cloud Logging helpers for querying LiteLLM parallel request metrics.

This module provides functions to query GCP Cloud Logging for parallel request
counters that are emitted by parallel_request_limiter_v3.py via print() statements.
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# Try to import google-cloud-logging
try:
    from google.cloud.logging import Client as LoggingClient
    from google.oauth2 import service_account

    GCP_LOGGING_AVAILABLE = True
except ImportError:
    GCP_LOGGING_AVAILABLE = False
    print(
        "[gcp_logs_query] google-cloud-logging not installed. "
        "Install with: pip install google-cloud-logging"
    )

# Regex pattern to parse the METRICS log line
METRICS_LOG_PATTERN = re.compile(
    r'\[METRICS\] Emitting parallel_requests metric: '
    r'token=(?P<token>[^,]+), '
    r'key_alias=(?P<key_alias>[^,]+), '
    r'previous_count=(?P<previous_count>\d+), '
    r'current_count=(?P<current_count>\d+), '
    r'operation=(?P<operation>[^,]+), '
    r'timestamp=(?P<timestamp>[\d.]+)'
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
    match = METRICS_LOG_PATTERN.search(text_payload)
    if not match:
        return None

    try:
        return {
            "token": match.group("token"),
            "key_alias": match.group("key_alias") if match.group("key_alias") != "None" else None,
            "previous_count": int(match.group("previous_count")),
            "current_count": int(match.group("current_count")),
            "operation": match.group("operation"),
            "timestamp": float(match.group("timestamp")),
        }
    except (ValueError, IndexError) as e:
        print(f"[gcp_logs_query] Failed to parse log line: {e}")
        return None


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
            escaped_token = api_key_filter.replace('"', '\\"')
            filter_parts.append(f'textPayload:"token={escaped_token}"')

        if key_alias_filter:
            # Add key_alias filter - escape quotes for safety
            # Note: This does partial match since log line contains key_alias=value
            escaped_alias = key_alias_filter.replace('"', '\\"')
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
