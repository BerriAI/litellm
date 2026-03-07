import asyncio
import json
import os
import tempfile
from typing import List, Optional

import pytest

import litellm


def _write_batch_jsonl(model: str) -> str:
    records = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 10,
            },
        },
        {
            "custom_id": "request-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an unhelpful assistant."},
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 10,
            },
        },
        {
            "custom_id": "request-3",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer with a single word."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 5,
            },
        },
    ]

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def _load_vertex_ai_credentials_from_env() -> None:
    os.environ["GCS_FLUSH_INTERVAL"] = "1"

    private_key_id = os.environ.get("GCS_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("GCS_PRIVATE_KEY", "").replace("\\n", "\n")

    if not private_key_id or not private_key:
        return

    service_account_key_data = {
        "private_key_id": private_key_id,
        "private_key": private_key,
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(service_account_key_data, temp_file, indent=2)

    abs_path = os.path.abspath(temp_file.name)
    os.environ["GCS_PATH_SERVICE_ACCOUNT"] = abs_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path


def _normalize_credential_path(env_key: str) -> Optional[str]:
    path = os.getenv(env_key)
    if not path:
        return None
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None
    os.environ[env_key] = abs_path
    return abs_path


def _extract_custom_ids_from_output(lines: List[dict]) -> List[str]:
    custom_ids: List[str] = []
    for line in lines:
        if "custom_id" in line:
            custom_ids.append(line["custom_id"])
        elif "key" in line:
            custom_ids.append(line["key"])
    return custom_ids


async def _wait_for_batch_completion(
    batch_id: str,
    provider: str,
    timeout_seconds: Optional[int] = None,
    poll_seconds: Optional[int] = None,
    require_output_file_id: bool = True,
):
    timeout_seconds = timeout_seconds or int(
        os.getenv("BATCH_WAIT_TIMEOUT_SECONDS", "1800")
    )
    poll_seconds = poll_seconds or int(os.getenv("BATCH_WAIT_POLL_SECONDS", "10"))

    start = asyncio.get_event_loop().time()
    while True:
        batch = await litellm.aretrieve_batch(
            batch_id=batch_id,
            custom_llm_provider=provider,
            litellm_params={
                "litellm_metadata": {"batch_ignore_default_logging": True},
            },
        )
        if batch.status == "completed":
            if not require_output_file_id or getattr(batch, "output_file_id", None):
                return batch
        if batch.status in ["failed", "cancelled", "expired"]:
            raise AssertionError(f"Batch ended with status={batch.status}")
        if asyncio.get_event_loop().time() - start > timeout_seconds:
            raise AssertionError("Timed out waiting for batch completion")
        await asyncio.sleep(poll_seconds)


@pytest.mark.asyncio
async def test_openai_batch_custom_id_mapping_live():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    file_path = _write_batch_jsonl(model="gpt-4o-mini")

    with open(file_path, "rb") as file_handle:
        file_obj = await litellm.acreate_file(
            file=file_handle,
            purpose="batch",
            custom_llm_provider="openai",
        )

    create_batch_response = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="openai",
    )

    completed_batch = await _wait_for_batch_completion(
        batch_id=create_batch_response.id,
        provider="openai",
    )

    assert completed_batch.output_file_id is not None

    output_content = await litellm.afile_content(
        file_id=completed_batch.output_file_id,
        custom_llm_provider="openai",
    )

    output_lines = [
        json.loads(line)
        for line in output_content.content.decode("utf-8").strip().split("\n")
        if line.strip()
    ]

    custom_ids = _extract_custom_ids_from_output(output_lines)
    assert sorted(custom_ids) == ["request-1", "request-2", "request-3"]


@pytest.mark.asyncio
async def test_vertex_batch_custom_id_mapping_live():
    if not os.getenv("VERTEXAI_PROJECT") or not os.getenv("VERTEXAI_LOCATION"):
        pytest.skip("VERTEXAI_PROJECT or VERTEXAI_LOCATION not set")
    if not os.getenv("GCS_BUCKET_NAME"):
        pytest.skip("GCS_BUCKET_NAME not set")

    _load_vertex_ai_credentials_from_env()

    normalized_gcs_path = _normalize_credential_path("GCS_PATH_SERVICE_ACCOUNT")
    normalized_google_path = _normalize_credential_path(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if not normalized_gcs_path and not normalized_google_path:
        pytest.skip(
            "Vertex credentials not set or file not found "
            "(GCS_PATH_SERVICE_ACCOUNT/GOOGLE_APPLICATION_CREDENTIALS)"
        )

    file_path = _write_batch_jsonl(model="gemini-2.5-flash-lite")

    with open(file_path, "rb") as file_handle:
        file_obj = await litellm.acreate_file(
            file=file_handle,
            purpose="batch",
            custom_llm_provider="vertex_ai",
        )

    create_batch_response = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="vertex_ai",
    )

    completed_batch = await _wait_for_batch_completion(
        batch_id=create_batch_response.id,
        provider="vertex_ai",
    )

    assert completed_batch.output_file_id is not None

    output_content = await litellm.afile_content(
        file_id=completed_batch.output_file_id,
        custom_llm_provider="vertex_ai",
    )

    output_lines = [
        json.loads(line)
        for line in output_content.content.decode("utf-8").strip().split("\n")
        if line.strip()
    ]

    custom_ids = _extract_custom_ids_from_output(output_lines)
    assert sorted(custom_ids) == ["request-1", "request-2", "request-3"]
