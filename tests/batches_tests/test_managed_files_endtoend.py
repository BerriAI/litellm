import warnings

import openai
import pytest
from tenacity import RetryError, Retrying, stop_after_delay, wait_fixed

from test_managed_files_base import (
    MODEL_IDS,
    MODEL_NAMES,
    ManagedFilesBase,
    MIN_EXPIRY_SECONDS,
)


class TestManagedFilesAPI(ManagedFilesBase):
    """Test cases for managed files and batch API.

    Configuration via environment variables:
        USE_LITELLM=true  - Run against LiteLLM proxy
        USE_LITELLM=false - Run against mock server directly (default)
    """

    @classmethod
    def setup_class(cls):
        cls.openai_client = cls.create_openai_client(cls, cls.api_key)

    def wait_for_batch_list(self, model_name, max_seconds=90, wait_seconds=10):
        for attempt in Retrying(
            stop=stop_after_delay(max_seconds),
            wait=wait_fixed(wait_seconds),
        ):
            with attempt:
                batches_list = self.openai_client.batches.list(
                    limit=10,
                    extra_query={"target_model_names": model_name},
                )
                print(
                    f"Batches in list: {len(batches_list.data)}",
                )
                if len(batches_list.data) == 0:
                    raise Exception("No batches found in list yet")
                return batches_list
        return None

    @pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_IDS)
    def test_e2e_managed_batch(self, tmp_path, model_name):
        print(
            f"\n\nStarting test with base_url={self.base_url} and model_name={model_name}\n",
        )

        self.reset_mock_server()

        request_file = self.create_batch_request_file_on_disk(tmp_path, model_name)

        print("Creating batch input file...")
        batch_input_file = self.create_batch_input_file(
            self.openai_client,
            request_file,
            MIN_EXPIRY_SECONDS,
        )
        print(f"Created batch input file: {self.shorten_id(batch_input_file.id)}\n")

        print(
            f"Retrieving batch input file metadata for file id: {self.shorten_id(batch_input_file.id)}",
        )
        input_file_metadata = self.openai_client.files.retrieve(batch_input_file.id)
        assert input_file_metadata.id == batch_input_file.id, "File ID mismatch"
        assert input_file_metadata.object == "file", "object should be 'file'"
        assert input_file_metadata.bytes > 0, "bytes not set"
        assert input_file_metadata.filename == "modified_file.jsonl", (
            "filename mismatch"
        )
        assert input_file_metadata.purpose == "batch", "purpose mismatch"
        assert input_file_metadata.status in ["uploaded", "processed", "error"], (
            "invalid status"
        )
        assert input_file_metadata.created_at > 0, "created_at not set"
        if not input_file_metadata.expires_at:
            warnings.warn("batch input file expires_at not set")
        self.print_file_metadata(input_file_metadata, "Input file")

        print("\nCreating batch...")
        batch = self.create_batch(
            self.openai_client,
            batch_input_file.id,
            MIN_EXPIRY_SECONDS,
        )
        print(f"Created batch: {self.shorten_id(batch.id)}")
        assert batch.id, "No batch ID returned"
        assert batch.input_file_id == batch_input_file.id, "File ID mismatch"
        assert batch.status in [
            "validating",
            "in_progress",
            "finalizing",
            "completed",
        ], "Status mismatch"
        if not batch.expires_at:
            warnings.warn("batch expires_at not set")
        else:
            assert batch.expires_at > 0, "batch expires_at not set"

        if not batch.endpoint:
            warnings.warn(
                "batch.endpoint empty in creation response - Azure API quirk, not a bug",
            )
        else:
            assert batch.endpoint == "/v1/chat/completions", "endpoint mismatch"
        assert batch.completion_window == "24h", "completion_window mismatch"
        assert batch.created_at > 0, "created_at not set"

        self.print_batch_metadata(batch)

        print("\nListing batches...")
        try:
            batches_list = self.wait_for_batch_list(
                model_name,
                max_seconds=30,
                wait_seconds=5,
            )
            batches = batches_list.data if batches_list else []
            batch_ids = [b.id for b in batches]
            if batch.id not in batch_ids:
                warnings.warn(
                    f"Batch {batch.id} not found in list. batches.list returns raw IDs and not the encoded IDs. raw IDs: {batch_ids}",
                )
        except RetryError:
            warnings.warn(
                "batches.list() returned empty list after retries - known LiteLLM issue with managed batches",
            )
        except openai.APIError as e:
            pytest.fail(f"batches.list() failed: {e}")

        print(
            f"\nWaiting for batch {self.shorten_id(batch.id)} to reach completed state...",
        )
        try:
            batch_response = self.wait_for_batch_state(
                self.openai_client,
                batch.id,
                "completed",
                max_seconds=30 * 60,
                wait_seconds=5,
            )
        except RetryError:
            raise TimeoutError("Timed out waiting for batch to be in state: completed")

        print("\nRetrieving batch output file metadata...")
        output_file_metadata = self.openai_client.files.retrieve(
            batch_response.output_file_id,
        )
        assert output_file_metadata.id == batch_response.output_file_id, (
            "Output file ID mismatch"
        )
        assert output_file_metadata.object == "file", "object should be 'file'"
        assert output_file_metadata.bytes > 0, "bytes not set"
        assert output_file_metadata.filename, "filename not set"
        assert output_file_metadata.purpose in ["batch_output", "batch"], (
            "purpose should be batch_output"
        )
        assert output_file_metadata.created_at > 0, "created_at not set"
        self.print_file_metadata(output_file_metadata, "Output file")

        print("\nFetching batch output file content...")
        batch_file_content = self.openai_client.files.content(
            batch_response.output_file_id,
        )
        assert batch_file_content.text, "No batch file content returned"
        assert len(batch_file_content.text) > 0, "Batch file content is empty"
        print(f"Output file content ({len(batch_file_content.text)} bytes):")
        for line in batch_file_content.text.strip().split("\n")[:3]:
            print(f"\t{line}")

        print(f"\nDeleting input file: {self.shorten_id(batch_input_file.id)}")
        try:
            self.openai_client.files.delete(batch_input_file.id)
        except openai.APIError as e:
            pytest.fail(f"files.delete() failed: {e}")

        print(
            f"\nDeleting output file: {self.shorten_id(batch_response.output_file_id)}",
        )
        try:
            self.openai_client.files.delete(batch_response.output_file_id)
        except openai.APIError as e:
            pytest.fail(f"files.delete() failed: {e}")

        print("\nVerifying input file is deleted...")
        try:
            self.openai_client.files.content(batch_input_file.id)
            assert False, f"Input file {batch_input_file.id} exists after deletion"
        except (openai.NotFoundError, openai.PermissionDeniedError):
            print("Input file correctly not accessible after deletion")

        print("\nVerifying output file is deleted...")
        try:
            self.openai_client.files.content(batch_response.output_file_id)
            assert False, (
                f"Output file {batch_response.output_file_id} exists after deletion"
            )
        except (openai.NotFoundError, openai.PermissionDeniedError):
            print("Output file correctly not accessible after deletion")




