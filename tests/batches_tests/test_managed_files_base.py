"""Base class for managed files and batch API tests."""

import json
import os
import time
import uuid

import httpx
import openai
import pytest
from tenacity import Retrying, stop_after_delay, wait_fixed


LOCAL_LITELLM_BASE_URL = "http://localhost:4000"
LOCAL_AZURE_BASE_URL = "http://localhost:8090"

USE_LITELLM = os.environ.get("USE_LITELLM", "true").lower() == "true"
if USE_LITELLM:
    base_url = LOCAL_LITELLM_BASE_URL
    api_key = "sk-1234"
else:
    base_url = LOCAL_AZURE_BASE_URL
    api_key = "sk-1234"

USE_MOCK_SERVER = os.environ.get("USE_MOCK_SERVER", "false").lower() == "true"
if USE_MOCK_SERVER:
    model_name = "azure-fake-gpt-5-batch-2025-08-07"
    MODEL_NAMES = [
        "azure-fake-gpt-5-batch-2025-08-07",
        #   "anthropic-fake-claude-sonnet-4-batch-2025-08-07",
        #  "vertex-fake-gemini-2.5-pro-batch-2025-08-07",
    ]
else:
    model_name = "gpt-5-batch-2025-08-07"
    MODEL_NAMES = [
        "gpt-5-batch-2025-08-07",
        # "claude-sonnet-4-batch-2025-08-07",
        # "gemini-2.5-pro-batch-2025-08-07",
    ]


def _extract_model_id(model_name: str) -> str:
    if "gpt" in model_name:
        return "gpt"
    elif "claude" in model_name or "anthropic" in model_name:
        return "anthropic"
    elif "gemini" in model_name or "vertex" in model_name:
        return "gemini"
    return model_name.split("-")[0]


MODEL_IDS = [_extract_model_id(m) for m in MODEL_NAMES]

MIN_EXPIRY_SECONDS = 259200


class ManagedFilesBase:
    """Base class with shared helpers for managed files and batch tests."""

    base_url = base_url
    api_key = api_key

    @pytest.fixture(autouse=True)
    def setup_test(self):
        print(f"Base URL: {self.base_url}, Model: {model_name}\n")
        self.reset_mock_server()

    @staticmethod
    def generate_request_id():
        return f"req-{uuid.uuid4().hex[:8]}"

    def create_openai_client(self, api_key: str) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False),
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
    ):
        batch_input_file = client.files.create(
            file=open(request_file, "rb"),
            purpose="batch",
            extra_body={
                "target_model_names": model_name,
                "expires_after": {
                    "seconds": expiry_seconds,
                    "anchor": "created_at",
                },
            },
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
    ):
        for attempt in Retrying(
            stop=stop_after_delay(max_seconds),
            wait=wait_fixed(wait_seconds),
        ):
            with attempt:
                batch_response = client.batches.retrieve(batch_id=batch_id)
                print(
                    f"[{time.strftime('%H:%M:%S')}] Batch status: {batch_response.status}, expected: {expected_status}",
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
        if not USE_MOCK_SERVER:
            return
        print("Resetting mock server state...")
        reset_response = httpx.post(f"{LOCAL_AZURE_BASE_URL}/reset")
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




