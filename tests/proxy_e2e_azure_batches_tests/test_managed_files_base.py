"""Base class for managed files and batch API tests."""

import json
import os
import sys
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
import openai
import psycopg2
import pytest
from tenacity import Retrying, stop_after_delay, wait_fixed

sys.path.insert(0, os.path.abspath("../.."))

from base_integration_test import (
    BaseLiteLLMIntegrationTest,
    get_mock_server_base_url,
    use_mock_models,
)


class ManagedFilesState:
    """Query and pretty print the state of managed files and objects tables."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not provided and not in environment")

    def _get_connection(self):
        parsed = urlparse(self.database_url)
        return psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            dbname=parsed.path.lstrip("/"),
        )

    def _shorten_id(self, id_str: str, max_len: int = 24) -> str:
        if id_str is None:
            return "None"
        if len(id_str) <= max_len:
            return id_str
        return id_str[:10] + "..." + id_str[-10:]

    def _format_timestamp(self, ts) -> str:
        if ts is None:
            return "None"
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)

    def get_managed_files(self, limit: int = 20) -> list:
        query = """
            SELECT unified_file_id, file_purpose, created_by, created_at,
                   updated_at, model_mappings, storage_backend
            FROM "LiteLLM_ManagedFileTable"
            ORDER BY created_at DESC
            LIMIT %s
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_managed_objects(
        self,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> list:
        query = """
            SELECT id, unified_object_id, status, file_purpose,
                   created_by, created_at, updated_at
            FROM "LiteLLM_ManagedObjectTable"
        """
        params = []
        if status:
            query += " WHERE status = %s"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def print_managed_files(self, limit: int = 20):
        files = self.get_managed_files(limit)
        print(f"\n{'=' * 80}")
        print(f"MANAGED FILES TABLE ({len(files)} rows)")
        print(f"{'=' * 80}")

        if not files:
            print("  (no rows)")
            return

        for i, f in enumerate(files, 1):
            print(f"\n[{i}] unified_file_id: {self._shorten_id(f['unified_file_id'])}")
            print(f"    purpose: {f['file_purpose']}")
            print(f"    created_by: {f['created_by']}")
            print(f"    created_at: {self._format_timestamp(f['created_at'])}")
            print(f"    storage_backend: {f.get('storage_backend', 'None')}")
            if f.get("model_mappings"):
                mappings = f["model_mappings"]
                if isinstance(mappings, dict):
                    print(f"    model_mappings: {len(mappings)} model(s)")
                    for model_id, file_id in list(mappings.items())[:3]:
                        print(
                            f"      - {self._shorten_id(model_id)}: {self._shorten_id(file_id)}",
                        )
                    if len(mappings) > 3:
                        print(f"      ... and {len(mappings) - 3} more")

    def print_managed_objects(self, limit: int = 20, status: Optional[str] = None):
        """Pretty print the managed objects table."""
        objects = self.get_managed_objects(limit, status)
        status_filter = f" (status={status})" if status else ""
        print(f"\n{'=' * 80}")
        print(f"MANAGED OBJECTS TABLE{status_filter} ({len(objects)} rows)")
        print(f"{'=' * 80}")

        if not objects:
            print("  (no rows)")
            return

        for i, o in enumerate(objects, 1):
            print(f"\n[{i}] id: {o['id']}")
            print(f"    unified_object_id: {self._shorten_id(o['unified_object_id'])}")
            print(f"    status: {o['status']}")
            print(f"    file_purpose: {o['file_purpose']}")
            print(f"    created_by: {o['created_by']}")
            print(f"    created_at: {self._format_timestamp(o['created_at'])}")

    def print_validating_batches(self):
        """Print batches that are stuck in validating state."""
        self.print_managed_objects(status="validating")

    def print_all(self, limit: int = 10):
        """Print both tables."""
        self.print_managed_files(limit)
        self.print_managed_objects(limit)

    def count_by_status(self) -> dict:
        """Count managed objects by status."""
        query = """
            SELECT status, COUNT(*) as count
            FROM "LiteLLM_ManagedObjectTable"
            GROUP BY status
            ORDER BY count DESC
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return {row[0]: row[1] for row in cur.fetchall()}

    def print_summary(self):
        """Print a summary of table states."""
        print(f"\n{'=' * 80}")
        print("DATABASE STATE SUMMARY")
        print(f"{'=' * 80}")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM "LiteLLM_ManagedFileTable"')
                file_count = cur.fetchone()[0]

                cur.execute('SELECT COUNT(*) FROM "LiteLLM_ManagedObjectTable"')
                object_count = cur.fetchone()[0]

        print(f"\nManaged Files: {file_count} total")
        print(f"Managed Objects: {object_count} total")

        status_counts = self.count_by_status()
        if status_counts:
            print("\nObjects by status:")
            for status, count in status_counts.items():
                print(f"  - {status}: {count}")

    def get_file_by_unified_id(self, unified_file_id: str) -> Optional[dict]:
        """Get a managed file by its unified file ID."""
        query = """
            SELECT unified_file_id, file_object, created_by, created_at,
                   updated_at, model_mappings, storage_backend
            FROM "LiteLLM_ManagedFileTable"
            WHERE unified_file_id = %s
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (unified_file_id,))
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
                return None

    def get_batch_by_unified_id(self, unified_object_id: str) -> Optional[dict]:
        """Get a managed batch/object by its unified object ID."""
        query = """
            SELECT id, unified_object_id, model_object_id, status, file_purpose,
                   created_by, created_at, updated_at
            FROM "LiteLLM_ManagedObjectTable"
            WHERE unified_object_id = %s
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (unified_object_id,))
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
                return None

    def get_batch_by_id(self, batch_id: int) -> Optional[dict]:
        """Get a managed batch/object by its integer ID."""
        query = """
            SELECT id, unified_object_id, status, file_purpose,
                   created_by, created_at, updated_at
            FROM "LiteLLM_ManagedObjectTable"
            WHERE id = %s
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (batch_id,))
                row = cur.fetchone()
                if row:
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
                return None


MIN_EXPIRY_SECONDS = 259200


class _BaseSubTracker:
    """Shared helpers for sub-trackers."""

    def _shorten_id(self, id_str: str, max_len: int = 20) -> str:
        if id_str is None:
            return "None"
        if len(id_str) <= max_len:
            return id_str
        return id_str[:8] + "..." + id_str[-8:]

    def _format_timestamp(self, ts) -> str:
        if ts is None:
            return "None"
        if isinstance(ts, datetime):
            return ts.strftime("%H:%M:%S")
        if isinstance(ts, int):
            return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        return str(ts)


class BatchDbStateTracker(_BaseSubTracker):
    """Tracks batch/file state in the LiteLLM database."""

    def __init__(self, db_state: ManagedFilesState):
        self.db_state = db_state

    def get_file_state(self, file_id: str) -> Optional[dict]:
        return self.db_state.get_file_by_unified_id(file_id)

    def get_batch_state(self, batch_id: str) -> Optional[dict]:
        return self.db_state.get_batch_by_unified_id(batch_id)

    def format_file_lines(self, file_id: str) -> tuple[str, list[str]]:
        """Return (header, detail_lines) for the DB file state."""
        db_file = self.get_file_state(file_id)
        header_id = (
            self._shorten_id(db_file.get("unified_file_id")) if db_file else "N/A"
        )
        header = f"FILE (DB): {header_id}"

        if not db_file:
            return header, ["  (not found in DB)"]

        file_obj = db_file.get("file_object") or {}
        if isinstance(file_obj, str):
            try:
                file_obj = json.loads(file_obj)
            except Exception:
                file_obj = {}
        lines = [
            f"  purpose: {file_obj.get('purpose', 'N/A')}",
            f"  storage: {db_file.get('storage_backend', 'N/A')}",
            f"  created: {self._format_timestamp(db_file.get('created_at'))}",
            f"  updated: {self._format_timestamp(db_file.get('updated_at'))}",
        ]
        mappings = db_file.get("model_mappings")
        if mappings and isinstance(mappings, dict):
            lines.append(f"  mappings: {len(mappings)} model(s)")
        return header, lines

    def format_batch_lines(self, batch_id: str) -> tuple[str, list[str]]:
        """Return (header, detail_lines) for the DB batch state."""
        db_batch = self.get_batch_state(batch_id)
        header_id = (
            self._shorten_id(db_batch.get("unified_object_id")) if db_batch else "N/A"
        )
        header = f"BATCH (DB): {header_id}"

        if not db_batch:
            return header, ["  (not found in DB)"]

        lines = [
            f"  status: {db_batch.get('status', 'N/A')}",
            f"  purpose: {db_batch.get('file_purpose', 'N/A')}",
            f"  created: {self._format_timestamp(db_batch.get('created_at'))}",
            f"  updated: {self._format_timestamp(db_batch.get('updated_at'))}",
        ]
        return header, lines


class BatchProviderStateTracker(_BaseSubTracker):
    """Tracks batch/file state as reported by the LLM provider (via OpenAI client)."""

    def __init__(self, openai_client: openai.OpenAI):
        self.client = openai_client

    def get_file_state(self, file_id: str) -> Optional[dict]:
        try:
            file_obj = self.client.files.retrieve(file_id)
            return {
                "id": file_obj.id,
                "status": file_obj.status,
                "purpose": file_obj.purpose,
                "bytes": file_obj.bytes,
                "filename": file_obj.filename,
                "created_at": file_obj.created_at,
                "expires_at": file_obj.expires_at,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_batch_state(self, batch_id: str) -> Optional[dict]:
        try:
            batch = self.client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "input_file_id": batch.input_file_id,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "request_counts": batch.request_counts,
            }
        except Exception as e:
            return {"error": str(e)}

    def format_file_lines(
        self,
        file_id: str,
        db_state: Optional[BatchDbStateTracker] = None,
    ) -> tuple[str, list[str]]:
        """Return (header, detail_lines) for the provider file state."""
        raw_file_id = "N/A"
        if db_state:
            db_file = db_state.get_file_state(file_id)
            if db_file:
                mappings = db_file.get("model_mappings")
                if mappings and isinstance(mappings, dict) and mappings:
                    first_file_id = next(iter(mappings.values()), None)
                    raw_file_id = (
                        self._shorten_id(first_file_id) if first_file_id else "N/A"
                    )
        header = f"FILE (RAW): {raw_file_id}"

        provider_file = self.get_file_state(file_id)
        if provider_file and "error" not in provider_file:
            lines = [
                f"  status: {provider_file.get('status', 'N/A')}",
                f"  purpose: {provider_file.get('purpose', 'N/A')}",
                f"  bytes: {provider_file.get('bytes', 0)}",
                f"  created: {self._format_timestamp(provider_file.get('created_at'))}",
                f"  expires: {self._format_timestamp(provider_file.get('expires_at'))}",
            ]
        elif provider_file and "error" in provider_file:
            lines = [f"  ERROR: {provider_file['error'][:35]}"]
        else:
            lines = ["  (not found)"]
        return header, lines

    def format_batch_lines(
        self,
        batch_id: str,
        db_state: Optional[BatchDbStateTracker] = None,
    ) -> tuple[str, list[str]]:
        """Return (header, detail_lines) for the provider batch state."""
        raw_prov_id = "N/A"
        if db_state:
            db_batch = db_state.get_batch_state(batch_id)
            if db_batch:
                raw_prov_id = self._shorten_id(db_batch.get("model_object_id"))
        header = f"BATCH (RAW): {raw_prov_id}"

        provider_batch = self.get_batch_state(batch_id)
        if provider_batch and "error" not in provider_batch:
            lines = [
                f"  status: {provider_batch.get('status', 'N/A')}",
                f"  input: {self._shorten_id(provider_batch.get('input_file_id'))}",
                f"  output: {self._shorten_id(provider_batch.get('output_file_id'))}",
                f"  created: {self._format_timestamp(provider_batch.get('created_at'))}",
                f"  completed: {self._format_timestamp(provider_batch.get('completed_at'))}",
            ]
            req_counts = provider_batch.get("request_counts")
            if req_counts:
                lines.append(
                    f"  requests: {req_counts.total} total, {req_counts.completed} done",
                )
        elif provider_batch and "error" in provider_batch:
            lines = [f"  ERROR: {provider_batch['error'][:35]}"]
        else:
            lines = ["  (not found)"]
        return header, lines


class BatchS3StateTracker(_BaseSubTracker):
    """Tracks S3 callback state from the mock S3 server."""

    def __init__(self, mock_server_base_url: str):
        self.mock_server_base_url = mock_server_base_url

    def get_callbacks(self, limit: int = 100) -> list[dict]:
        try:
            response = httpx.get(
                f"{self.mock_server_base_url}/mock-s3/callbacks",
                params={"limit": limit},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json().get("callbacks", [])
            return []
        except Exception:
            return []

    def get_batch_callbacks(self) -> list[dict]:
        """Return only callbacks related to batch operations."""
        batch_call_types = {
            "acreate_batch",
            "aretrieve_batch",
            "acreate_file",
            "afile_content",
        }
        return [
            cb
            for cb in self.get_callbacks()
            if cb.get("content", {}).get("call_type", "") in batch_call_types
        ]

    def get_cost_callbacks(self) -> list[dict]:
        """Return CheckBatchCost callbacks (aretrieve_batch with no user_api_key_hash)."""
        result = []
        for cb in self.get_callbacks():
            content = cb.get("content", {})
            if content.get("call_type") != "aretrieve_batch":
                continue
            metadata = content.get("metadata") or {}
            if metadata.get("user_api_key_hash") is None:
                result.append(cb)
        return result

    def format_batch_lines(self, batch_id: str) -> tuple[str, list[str]]:
        """Return (header, detail_lines) summarising S3 callback state for this batch."""
        all_cbs = self.get_callbacks()
        batch_cbs = self.get_batch_callbacks()
        cost_cbs = self.get_cost_callbacks()

        header = f"S3 CALLBACKS: {len(all_cbs)} total"
        lines = [
            f"  batch-related: {len(batch_cbs)}",
            f"  cost events: {len(cost_cbs)}",
        ]

        # Summarise call_type breakdown for batch callbacks
        type_counts: dict[str, int] = {}
        for cb in batch_cbs:
            ct = cb.get("content", {}).get("call_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        for ct, count in sorted(type_counts.items()):
            lines.append(f"    {ct}: {count}")

        # Show cost info from the latest cost callback (if any)
        if cost_cbs:
            latest = cost_cbs[-1].get("content", {})
            lines.append(f"  latest cost event:")
            lines.append(f"    model: {latest.get('model', 'N/A')}")
            lines.append(f"    response_cost: {latest.get('response_cost', 'N/A')}")
            lines.append(f"    total_tokens: {latest.get('total_tokens', 0)}")

        return header, lines

    def print_all_callbacks(self):
        """Print every S3 callback object in detail, ordered by S3 key timestamp."""
        callbacks = self.get_callbacks()

        # Sort by the timestamp embedded in the S3 key (e.g. "2026-02-15/time-13-01-31-269789_...")
        callbacks.sort(key=lambda cb: cb.get("key", ""))

        print(f"\n{'=' * 90}")
        print(
            f"S3 CALLBACK DETAIL — {len(callbacks)} object(s), ordered by received time",
        )
        print(f"{'=' * 90}")

        if not callbacks:
            print("  (no callbacks)")
            return

        for i, cb in enumerate(callbacks, 1):
            content = cb.get("content", {})
            metadata = content.get("metadata") or {}
            hidden = content.get("hidden_params") or {}

            print(f"\n[{i}] call_type: {content.get('call_type', 'N/A')}")
            print(
                f"    s3_received_at: {cb.get('received_at', cb.get('timestamp', 'N/A'))}",
            )
            print(f"    id: {self._shorten_id(content.get('id', ''))}")
            print(f"    model: {content.get('model', 'N/A')}")
            print(f"    status: {content.get('status', 'N/A')}")
            print(f"    response_cost: {content.get('response_cost', 'N/A')}")
            print(f"    total_tokens: {content.get('total_tokens', 0)}")
            print(f"    prompt_tokens: {content.get('prompt_tokens', 0)}")
            print(f"    completion_tokens: {content.get('completion_tokens', 0)}")
            print(
                f"    custom_llm_provider: {content.get('custom_llm_provider', 'N/A')}",
            )
            print(f"    api_base: {self._shorten_id(content.get('api_base', ''), 40)}")
            print(f"    cache_hit: {content.get('cache_hit', 'N/A')}")

            print(f"    metadata:")
            print(
                f"      user_api_key_hash: {self._shorten_id(metadata.get('user_api_key_hash', 'None'))}",
            )
            print(
                f"      user_api_key_alias: {metadata.get('user_api_key_alias', 'None')}",
            )
            print(
                f"      user_api_key_team_id: {metadata.get('user_api_key_team_id', 'None')}",
            )
            print(
                f"      user_api_key_team_alias: {metadata.get('user_api_key_team_alias', 'None')}",
            )
            print(
                f"      user_api_key_user_id: {metadata.get('user_api_key_user_id', 'None')}",
            )

            batch_models = hidden.get("batch_models")
            if batch_models:
                print(f"    batch_models: {batch_models}")

            response = content.get("response") or {}
            if isinstance(response, dict) and response.get("status"):
                print(f"    response.status: {response.get('status')}")
                req_counts = response.get("request_counts") or {}
                if req_counts:
                    print(
                        f"    response.request_counts: total={req_counts.get('total', 0)}, completed={req_counts.get('completed', 0)}, failed={req_counts.get('failed', 0)}",
                    )
                out_file = response.get("output_file_id")
                if out_file:
                    print(f"    response.output_file_id: {self._shorten_id(out_file)}")

            s3_key = cb.get("key", "")
            if s3_key:
                print(f"    s3_key: {s3_key}")

        print(f"\n{'=' * 90}\n")


class NoOpStateTracker:
    """No-op tracker used when state tracking is disabled."""

    def set_file_id(self, file_id: str):
        pass

    def set_batch_id(self, batch_id: str):
        pass

    def print_state(self, step_name: str):
        pass

    def wait_and_print_s3_callbacks(self):
        pass

    def assert_batch_cost_callback(self):
        pass


class StateTracker:
    """Tracks and prints DB, Provider, and S3 state after each step."""

    def __init__(
        self,
        db_tracker: BatchDbStateTracker,
        provider_tracker: BatchProviderStateTracker,
        s3_tracker: Optional[BatchS3StateTracker] = None,
    ):
        self.db_tracker = db_tracker
        self.provider_tracker = provider_tracker
        self.s3_tracker = s3_tracker
        self.current_file_id: Optional[str] = None
        self.current_batch_id: Optional[str] = None
        self.step_number = 0

    def set_file_id(self, file_id: str):
        """Set the file ID to track."""
        self.current_file_id = file_id

    def set_batch_id(self, batch_id: str):
        """Set the batch ID to track."""
        self.current_batch_id = batch_id

    def print_state(self, step_name: str):
        """Print DB, provider, and S3 state for tracked file and batch."""
        self.step_number += 1
        has_s3 = self.s3_tracker is not None
        col_width = 40
        num_cols = 3 if has_s3 else 2
        total_width = (col_width + 3) * num_cols

        print(f"\n{'─' * total_width}")
        print(f"│ STEP {self.step_number}: {step_name}")
        print(f"{'─' * total_width}")

        col_headers = [
            f"{'DATABASE STATE':<{col_width}}",
            f"{'PROVIDER STATE':<{col_width}}",
        ]
        if has_s3:
            col_headers.append(f"{'S3 STATE':<{col_width}}")
        print("│ " + " │ ".join(col_headers))
        print(f"{'─' * total_width}")

        if self.current_file_id:
            self._print_file_state(col_width, has_s3)

        if self.current_batch_id:
            self._print_batch_state(col_width, has_s3)

        print(f"{'─' * total_width}\n")

    def _has_completed_batch_cost_callback(self) -> bool:
        """Check if an aretrieve_batch callback with completed status and cost>0 exists."""
        for cb in self.s3_tracker.get_callbacks():
            content = cb.get("content", {})
            if content.get("call_type") != "aretrieve_batch":
                continue
            response = content.get("response") or {}
            if not isinstance(response, dict) or response.get("status") != "completed":
                continue
            cost = content.get("response_cost", 0)
            if cost and cost > 0:
                return True
        return False

    def wait_and_print_s3_callbacks(self):
        """Wait for the S3 v2 logger to flush, then print all callbacks in detail.

        Waits until the cost callback arrives or max_wait is reached.
        After detecting the cost callback, waits one extra flush interval
        for the proxy to finalize batch_processed before returning.
        """
        if not self.s3_tracker:
            return

        s3_flush_interval = int(os.environ.get("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", 10))
        batch_poll_interval = int(os.environ.get("PROXY_BATCH_POLLING_INTERVAL", 10))
        max_wait = batch_poll_interval * 3 + s3_flush_interval * 5
        prev_count = len(self.s3_tracker.get_callbacks())
        waited = 0
        cost_detected = False
        while waited < max_wait:
            print(
                f"Waiting for {s3_flush_interval} secs for S3 callbacks to be flushed",
            )
            time.sleep(s3_flush_interval)
            waited += s3_flush_interval
            curr_count = len(self.s3_tracker.get_callbacks())
            print(
                f"[S3 flush wait] {waited}s/{max_wait}s — "
                f"callbacks: {prev_count} → {curr_count}",
            )
            prev_count = curr_count

            if not cost_detected and self._has_completed_batch_cost_callback():
                print(
                    "Cost callback detected — waiting one more interval "
                    "for batch_processed finalization"
                )
                cost_detected = True
            elif cost_detected:
                break

        self.s3_tracker.print_all_callbacks()

    def assert_batch_cost_callback(self):
        """Assert that a completed-batch S3 callback with non-zero cost exists."""
        if not self.s3_tracker:
            return

        callbacks = self.s3_tracker.get_callbacks()
        valid_callbacks = []
        for cb in callbacks:
            content = cb.get("content", {})
            if content.get("call_type") != "aretrieve_batch":
                continue
            response = content.get("response") or {}
            if not isinstance(response, dict) or response.get("status") != "completed":
                continue
            cost = content.get("response_cost", 0)
            if cost and cost > 0:
                valid_callbacks.append(cb)

        if len(valid_callbacks) != 1:
            print(
                f"\n❌ Assertion failed: Found {len(valid_callbacks)} valid callbacks (expected 1)",
            )
            print(
                "\nAll valid callbacks with call_type=aretrieve_batch, status=completed, cost>0:",
            )
            for idx, cb in enumerate(valid_callbacks, 1):
                content = cb.get("content", {})
                print(f"\n[{idx}] Callback:")
                print(f"    id: {content.get('id', 'N/A')}")
                print(f"    response_cost: {content.get('response_cost', 0)}")
                print(f"    litellm_call_id: {content.get('litellm_call_id', 'N/A')}")
                response = content.get("response", {})
                print(f"    response.id: {response.get('id', 'N/A')}")
                print(f"    response.status: {response.get('status', 'N/A')}")
                metadata = content.get("metadata", {})
                print(
                    f"    user_api_key_user_id: {metadata.get('user_api_key_user_id', 'N/A')}",
                )
                print(
                    f"    user_api_key_alias: {metadata.get('user_api_key_alias', 'N/A')}",
                )
                print(
                    f"    user_api_key_hash: {metadata.get('user_api_key_hash', 'N/A')}",
                )
                print(f"    source: {metadata.get('source', 'NOT SET')}")
            raise AssertionError(
                f"Expected 1 valid callback with call_type=aretrieve_batch, "
                f"response.status=completed, and response_cost > 0. "
                f"Found {len(valid_callbacks)} valid callbacks.",
            )

        valid_callback = valid_callbacks[0]
        callback_user_alias = (
            valid_callback.get("content", {})
            .get("metadata", {})
            .get("user_api_key_alias")
        )
        if not callback_user_alias:
            raise AssertionError(
                f"Expected user_api_key_alias to be set. Found {callback_user_alias}.",
            )

        if callback_user_alias == "default_user_alias":
            raise AssertionError(
                f"Expected user_api_key_alias to be set to the user who created the batch. "
                f"Expected user_api_key_alias to be 'default_user_alias'. "
                f"Found {callback_user_alias}.",
            )

    def _print_columns(self, columns: list[list[str]], col_width: int):
        """Print multiple columns side-by-side."""
        max_lines = max(len(col) for col in columns)
        for i in range(max_lines):
            parts = []
            for col in columns:
                line = col[i] if i < len(col) else ""
                parts.append(f"{line:<{col_width}}")
            print("│ " + " │ ".join(parts))

    def _print_file_state(self, col_width: int, has_s3: bool):
        db_header, db_lines = self.db_tracker.format_file_lines(self.current_file_id)
        prov_header, prov_lines = self.provider_tracker.format_file_lines(
            self.current_file_id,
            db_state=self.db_tracker,
        )

        headers = [db_header, prov_header]
        columns = [db_lines, prov_lines]
        if has_s3:
            headers.append("")
            columns.append([])

        header_parts = [f"{h:<{col_width}}" for h in headers]
        print("│ " + " │ ".join(header_parts))
        self._print_columns(columns, col_width)

    def _print_batch_state(self, col_width: int, has_s3: bool):
        db_header, db_lines = self.db_tracker.format_batch_lines(self.current_batch_id)
        prov_header, prov_lines = self.provider_tracker.format_batch_lines(
            self.current_batch_id,
            db_state=self.db_tracker,
        )

        headers = [db_header, prov_header]
        columns = [db_lines, prov_lines]
        if has_s3:
            s3_header, s3_lines = self.s3_tracker.format_batch_lines(
                self.current_batch_id,
            )
            headers.append(s3_header)
            columns.append(s3_lines)

        # blank separator row
        blank = [f"{'':<{col_width}}"] * len(headers)
        print("│ " + " │ ".join(blank))

        header_parts = [f"{h:<{col_width}}" for h in headers]
        print("│ " + " │ ".join(header_parts))
        self._print_columns(columns, col_width)


def get_batch_model_names():
    if use_mock_models():
        return [
            "azure-fake-gpt-5-batch-2025-08-07",
        ]
    return [
        "gpt-5-batch-2025-08-07",
    ]


class ManagedFilesBase(BaseLiteLLMIntegrationTest):
    """Base class with shared helpers for managed files and batch tests."""

    @pytest.fixture(autouse=True)
    def setup_test(self, request):
        print(
            f"Base URL: {self.base_url}, Using mock models: {use_mock_models()}\n",
        )

    def create_state_tracker(self) -> "StateTracker | NoOpStateTracker":
        """Create a StateTracker for observing DB, Provider, and S3 state.

        Returns a NoOpStateTracker if USE_STATE_TRACKER is not 'true' or
        if DATABASE_URL is not set.
        """
        use_tracker = os.environ.get("USE_STATE_TRACKER", "").lower() == "true"
        if not use_tracker:
            return NoOpStateTracker()

        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            print("Warning: DATABASE_URL not set, state tracking disabled")
            return NoOpStateTracker()
        try:
            db_state = ManagedFilesState(database_url)
            db_tracker = BatchDbStateTracker(db_state)
            provider_tracker = BatchProviderStateTracker(self.openai_client)

            s3_tracker = None
            try:
                mock_url = get_mock_server_base_url()
                s3_tracker = BatchS3StateTracker(mock_url)
            except Exception:
                pass

            return StateTracker(db_tracker, provider_tracker, s3_tracker)
        except Exception as e:
            print(f"Warning: Could not create state tracker: {e}")
            return NoOpStateTracker()

    def create_openai_client_with_key(self, api_key: str) -> openai.OpenAI:
        """Create an OpenAI client with a specific API key."""
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=self._get_ssl_verify_setting()),
        )

    def create_batch_request_file_on_disk(self, tmpdir, model: str):
        request_id = self.generate_request_id()
        batch_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                ],
            },
        }

        request_file = os.path.join(tmpdir, f"request-{request_id}.jsonl")
        with open(request_file, "w") as f:
            f.write(json.dumps(batch_request))

        return request_file

    def create_batch_input_file(
        self,
        client: openai.OpenAI,
        request_file: str,
        expiry_seconds: int = MIN_EXPIRY_SECONDS,
        target_model_names: str = None,
    ):
        extra_body = {
            "expires_after": {
                "seconds": expiry_seconds,
                "anchor": "created_at",
            },
        }
        if target_model_names:
            extra_body["target_model_names"] = target_model_names

        batch_input_file = client.files.create(
            file=open(request_file, "rb"),
            purpose="batch",
            extra_body=extra_body,
        )
        return batch_input_file

    def create_batch(
        self,
        client: openai.OpenAI,
        input_file_id: str,
        expiry_seconds: int = MIN_EXPIRY_SECONDS,
    ):
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            extra_body={
                "output_expires_after": {
                    "seconds": expiry_seconds,
                    "anchor": "created_at",
                },
            },
        )
        return batch

    def wait_for_batch_state(
        self,
        client: openai.OpenAI,
        batch_id: str,
        expected_status: str,
        max_seconds: int = 60,
        wait_seconds: int = 5,
        state_tracker: "StateTracker | NoOpStateTracker | None" = None,
    ):
        if state_tracker is None:
            state_tracker = NoOpStateTracker()
        poll_count = 0
        for attempt in Retrying(
            stop=stop_after_delay(max_seconds),
            wait=wait_fixed(wait_seconds),
        ):
            with attempt:
                poll_count += 1
                batch_response = client.batches.retrieve(batch_id=batch_id)
                print(
                    f"[{time.strftime('%H:%M:%S')}] Poll #{poll_count}: Batch status: {batch_response.status}, expected: {expected_status}",
                )
                state_tracker.print_state(
                    f"Poll #{poll_count} - status: {batch_response.status}",
                )
                if batch_response.status == expected_status:
                    return batch_response
                if batch_response.status in ["failed", "expired", "cancelled"]:
                    raise Exception(
                        f"Batch failed with status: {batch_response.status}",
                    )
                raise Exception(f"Batch not in {expected_status} state yet")
        return None

    def wait_for_batch_completed(
        self,
        client: openai.OpenAI,
        batch_id: str,
        max_seconds: int = 120,
        wait_seconds: int = 5,
    ):
        return self.wait_for_batch_state(
            client,
            batch_id,
            "completed",
            max_seconds,
            wait_seconds,
        )

    def shorten_id(self, id_str: str) -> str:
        if id_str is None:
            return "None"
        if len(id_str) <= 20:
            return id_str
        return id_str[:8] + "..." + id_str[-8:]

    def reset_mock_server(self):
        if not use_mock_models():
            return
        print("Resetting mock server state...")
        reset_response = httpx.post(f"{get_mock_server_base_url()}/reset")
        assert reset_response.status_code == 200, f"Reset failed: {reset_response.text}"

    def print_file_metadata(self, file_obj, label="File"):
        print(f"{label} metadata:")
        print(f"\tid={self.shorten_id(file_obj.id)}")
        print(f"\tobject={file_obj.object}")
        print(f"\tbytes={file_obj.bytes}")
        print(f"\tfilename={file_obj.filename}")
        print(f"\tpurpose={file_obj.purpose}")
        print(f"\tstatus={file_obj.status}")
        print(f"\tcreated_at={file_obj.created_at}")
        print(f"\texpires_at={file_obj.expires_at}")
        if file_obj.status_details:
            print(f"\tstatus_details={file_obj.status_details}")

    def print_batch_metadata(self, batch):
        print("Batch metadata:")
        print(f"\tid={self.shorten_id(batch.id)}")
        print(f"\tstatus={batch.status}")
        print(f"\tendpoint={batch.endpoint}")
        print(f"\tcompletion_window={batch.completion_window}")
        print(f"\tinput_file_id={self.shorten_id(batch.input_file_id)}")
        print(f"\tcreated_at={batch.created_at}")
        print(f"\texpires_at={batch.expires_at}")
        print(f"\tin_progress_at={batch.in_progress_at}")
        print(f"\tcompleted_at={batch.completed_at}")
        print(f"\toutput_file_id={self.shorten_id(batch.output_file_id)}")
        print(f"\trequest_counts={batch.request_counts}")

    def wait_for_batch_list(self, model_name, max_seconds=90, wait_seconds=10):
        for attempt in Retrying(
            stop=stop_after_delay(max_seconds),
            wait=wait_fixed(wait_seconds),
        ):
            with attempt:
                batches_list = self.openai_client.batches.list(
                    limit=10,
                    # extra query is not supported by managed batches
                    # extra_query={"target_model_names": model_name},
                )
                print(
                    f"Batches in list: {len(batches_list.data)}",
                )
                if len(batches_list.data) == 0:
                    raise Exception("No batches found in list yet")
                print("Batches in list:")
                for batch in batches_list.data:
                    print(
                        f"  ID: {self.shorten_id(batch.id)} Status: {batch.status}, Created at: {batch.created_at}, Completed at: {batch.completed_at}",
                    )
                return batches_list
        return None

    def wait_for_batch_in_list(
        self,
        client: openai.OpenAI,
        batch_id: str,
        max_seconds: int = 10,
        wait_seconds: float = 0.5,
    ):
        """Wait for a specific batch to appear in the batch list.

        This handles the race condition where batch creation returns before
        the database insert completes (due to asyncio.create_task).
        """
        for attempt in Retrying(
            stop=stop_after_delay(max_seconds),
            wait=wait_fixed(wait_seconds),
        ):
            with attempt:
                batches_list = client.batches.list(limit=20)
                batch_ids = [b.id for b in batches_list.data]
                if batch_id not in batch_ids:
                    raise Exception(
                        f"Batch {self.shorten_id(batch_id)} not found in list yet",
                    )
                return batches_list
        return None