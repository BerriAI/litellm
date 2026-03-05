import base64
import os
import sys
import time
import warnings

import httpx
import openai
import pytest
from tenacity import RetryError

sys.path.insert(0, os.path.abspath("../.."))

from base_integration_test import (
    get_mock_server_base_url,
    model_id,
    use_mock_models,
    UserKeyTestMixin,
)
from test_managed_files_base import (
    ManagedFilesBase,
    MIN_EXPIRY_SECONDS,
    get_batch_model_names,
)

MANAGED_FILE_ID_PREFIX = "litellm_proxy"

pytestmark = [
    pytest.mark.usefixtures("mock_azure_server", "litellm_proxy_server"),
    pytest.mark.skipif(
        os.environ.get("SKIP_E2E_TESTS", "false").lower() == "true",
        reason="E2E tests disabled via SKIP_E2E_TESTS env var"
    ),
]


def is_managed_id(file_id: str) -> bool:
    """Check if a file ID is a base64-encoded LiteLLM managed/unified ID."""
    try:
        padded = file_id + "=" * (-len(file_id) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()
        return decoded.startswith(MANAGED_FILE_ID_PREFIX)
    except Exception:
        return False


def assert_managed_id(file_id: str, label: str):
    assert is_managed_id(file_id), f"{label} should be a managed ID, got raw: {file_id}"


def wip_features_enabled() -> bool:
    return os.environ.get("WIP_FEATURES", "").lower() == "true"


class TestManagedFilesAPI(ManagedFilesBase, UserKeyTestMixin):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.setup_admin_client()

    @classmethod
    def teardown_class(cls):
        cls.teardown_admin_client()

    @pytest.fixture(autouse=True)
    def setup_test(self):
        print(
            f"\nBase URL: {self.base_url}, Using mock models: {use_mock_models()}",
        )
        self.clear_s3_callbacks()

        user_id, api_key, user_email, client = self.create_user_key_and_client(
            "e2e-batch",
        )
        self.test_user_id = user_id
        self.openai_client = client
        print(f"Using user {user_email} (id={user_id})")

    def _create_and_verify_batch_input_file(self, tmp_path, model_name):
        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)

        print("Creating batch input file...")
        batch_input_file = self.create_batch_input_file(
            self.openai_client,
            request_file,
            MIN_EXPIRY_SECONDS,
            target_model_names=model_name,
        )
        print(f"Created batch input file: {self.shorten_id(batch_input_file.id)}")
        assert_managed_id(batch_input_file.id, "batch_input_file.id")

        print("Retrieving batch input file metadata...")
        metadata = self.openai_client.files.retrieve(batch_input_file.id)
        assert_managed_id(metadata.id, "files.retrieve(input).id")
        assert metadata.id == batch_input_file.id, (
            f"Input file ID mismatch: retrieve returned '{metadata.id}' but expected '{batch_input_file.id}'"
        )
        assert metadata.object == "file"
        assert metadata.bytes > 0, "bytes not set"
        assert metadata.filename == "modified_file.jsonl"
        assert metadata.purpose == "batch"
        assert metadata.status in ["uploaded", "processed", "error"]
        assert metadata.created_at > 0
        if wip_features_enabled():
            assert metadata.expires_at > 0, "expires_at not set"
        self.print_file_metadata(metadata, "Input file")

        return batch_input_file

    def _create_and_verify_batch(self, input_file_id):
        print("\nCreating batch...")
        batch = self.create_batch(
            self.openai_client,
            input_file_id,
            MIN_EXPIRY_SECONDS,
        )
        print(f"Created batch: {self.shorten_id(batch.id)}")

        assert batch.id, "No batch ID returned"
        assert_managed_id(batch.id, "batch.id")
        assert_managed_id(batch.input_file_id, "batch.input_file_id")
        assert batch.input_file_id == input_file_id, "batch.input_file_id mismatch"
        assert batch.status in ["validating", "in_progress", "finalizing", "completed"]
        if not batch.expires_at:
            warnings.warn("batch expires_at not set")
        else:
            assert batch.expires_at > 0
        if not batch.endpoint:
            warnings.warn("batch.endpoint empty - Azure API quirk, not a bug")
        else:
            assert batch.endpoint == "/v1/chat/completions"
        assert batch.completion_window == "24h"
        assert batch.created_at > 0
        self.print_batch_metadata(batch)

        return batch

    def _list_batches(self, batch_id, model_name):
        if not wip_features_enabled():
            return
        print("\nListing batches...")
        try:
            batches_list = self.wait_for_batch_list(
                model_name,
                max_seconds=30,
                wait_seconds=5,
            )
            batch_ids = [b.id for b in (batches_list.data if batches_list else [])]
            if batch_id not in batch_ids:
                warnings.warn(
                    f"Batch {batch_id} not found in list. "
                    f"batches.list returns raw IDs, not encoded IDs. raw IDs: {batch_ids}",
                )
        except openai.APIError as e:
            pytest.fail(f"batches.list() failed: {e}")

    def _wait_for_batch_completion(self, batch_id, tracker):
        print(f"\nWaiting for batch {self.shorten_id(batch_id)} to complete...")
        try:
            batch_response = self.wait_for_batch_state(
                self.openai_client,
                batch_id,
                "completed",
                max_seconds=25 * 60,
                wait_seconds=15,
                state_tracker=tracker,
            )
        except RetryError:
            tracker.print_state("Timeout waiting for batch completion")
            raise TimeoutError("Timed out waiting for batch to be in state: completed")

        assert_managed_id(batch_response.id, "batch_response.id")
        assert batch_response.id == batch_id, (
            f"batch_response.id mismatch: got '{batch_response.id}' but expected '{batch_id}'"
        )
        assert_managed_id(batch_response.input_file_id, "batch_response.input_file_id")
        assert_managed_id(
            batch_response.output_file_id,
            "batch_response.output_file_id",
        )

        return batch_response

    def _get_and_verify_batch_output(self, output_file_id):
        print("\nRetrieving batch output file metadata...")
        metadata = self.openai_client.files.retrieve(output_file_id)
        assert_managed_id(metadata.id, "files.retrieve(output_file_id).id")
        assert metadata.id == output_file_id, (
            f"Output file ID mismatch: retrieve returned '{metadata.id}' but expected '{output_file_id}'"
        )
        assert metadata.object == "file"
        assert metadata.bytes > 0, "bytes not set"
        assert metadata.filename, "filename not set"
        assert metadata.purpose in ["batch_output", "batch"]
        assert metadata.created_at > 0
        self.print_file_metadata(metadata, "Output file")

        print("\nFetching batch output file content...")
        content = self.openai_client.files.content(output_file_id)
        assert content.text, "No batch file content returned"
        assert len(content.text) > 0, "Batch file content is empty"
        print(f"Output file content ({len(content.text)} bytes):")
        for line in content.text.strip().split("\n")[:3]:
            print(f"\t{line}")

        return metadata

    def _delete_file(self, file_id, label, max_retries=6, retry_delay=10):
        print(f"\nDeleting {label}: {self.shorten_id(file_id)}")
        for attempt in range(max_retries):
            try:
                self.openai_client.files.delete(file_id)
                return
            except openai.BadRequestError as e:
                if "batch_processed" in str(e) and attempt < max_retries - 1:
                    print(
                        f"  File still referenced by unprocessed batch, "
                        f"retrying in {retry_delay}s ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    pytest.fail(f"files.delete({label}) failed: {e}")
            except openai.APIError as e:
                pytest.fail(f"files.delete({label}) failed: {e}")

    def _verify_file_deleted(self, file_id, label):
        print(f"Verifying {label} is deleted...")
        try:
            self.openai_client.files.content(file_id)
            assert False, f"{label} {file_id} still accessible after deletion"
        except openai.NotFoundError:
            print(f"{label} correctly not accessible after deletion")

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "model_name",
        get_batch_model_names(),
        ids=model_id,
    )
    def test_e2e_managed_batch(self, tmp_path, model_name):
        print(
            f"\n\nStarting test with base_url={self.base_url} and model_name={model_name}\n",
        )
        self.reset_mock_server()
        tracker = self.create_state_tracker()

        batch_input_file = self._create_and_verify_batch_input_file(
            tmp_path,
            model_name,
        )
        tracker.set_file_id(batch_input_file.id)
        tracker.print_state("After creating batch input file")

        batch = self._create_and_verify_batch(batch_input_file.id)
        tracker.set_batch_id(batch.id)
        tracker.print_state("After creating batch")

        self._list_batches(batch.id, model_name)

        batch_response = self._wait_for_batch_completion(batch.id, tracker)
        tracker.print_state("After batch completed")

        self._get_and_verify_batch_output(batch_response.output_file_id)
        tracker.print_state("After retrieving output file")

        tracker.print_state("Final state after cleanup")
        tracker.wait_and_print_s3_callbacks()
        tracker.assert_batch_cost_callback()

        self._delete_file(batch_input_file.id, "input file")
        self._delete_file(batch_response.output_file_id, "output file")

        self._verify_file_deleted(batch_input_file.id, "input file")
        self._verify_file_deleted(batch_response.output_file_id, "output file")

    def cleanup_batches_in_database(self):
        import psycopg2

        print("Cleaning up stale batch records from database...")
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="litellm",
                user="llmproxy",
                password="dbpassword9090",
            )
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM "LiteLLM_ManagedObjectTable"
                    WHERE file_purpose = 'batch' AND status = 'validating'
                """)
                deleted = cur.rowcount
                conn.commit()
                if deleted > 0:
                    print(f"Deleted {deleted} stale batch records")
            conn.close()
        except Exception as e:
            print(f"Warning: Could not clean up database: {e}")

    def clear_s3_callbacks(self):
        clear_response = httpx.delete(f"{get_mock_server_base_url()}/mock-s3/callbacks")
        assert clear_response.status_code == 200, (
            f"Failed to clear callbacks: {clear_response.text}"
        )
        return clear_response.json()

    @pytest.mark.skipif(
        True,
        reason="Skipping managed files test till managed files feature is available",
    )
    @pytest.mark.parametrize(
        "model_name",
        get_batch_model_names(),
        ids=model_id,
    )
    def test_error_files(self, tmp_path, model_name):
        raise NotImplementedError(
            "To implement. Fail a batch and retrieve the error file.",
        )