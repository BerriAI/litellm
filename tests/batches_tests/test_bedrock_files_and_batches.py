# What is this?
## Unit Tests for OpenAI Batches API
import asyncio
import json as json_module
import os
import traceback
import tempfile
from dotenv import load_dotenv

load_dotenv()


import pytest
from typing import Optional
import litellm
from unittest.mock import patch, MagicMock
import httpx
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


_BEDROCK_TEST_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
    "AWS_REGION": "us-west-2",
    "AWS_DEFAULT_REGION": "us-west-2",
}


class _CaptureAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self):
        self.timeout = None
        self.event_hooks = None
        self.client_alias = "bedrock-test"
        self.put_calls = []
        self.post_calls = []
        self.batch_jobs = {}

    async def put(
        self,
        url: str,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream: bool = False,
        content=None,
    ):
        self.put_calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "params": params,
                "headers": headers or {},
                "timeout": timeout,
                "stream": stream,
                "content": content,
            }
        )
        body = data if data is not None else content
        content_bytes = body.encode("utf-8") if isinstance(body, str) else body or b""
        content_length = len(content_bytes)
        return httpx.Response(
            status_code=200,
            headers={"Content-Length": str(content_length)},
            request=httpx.Request("PUT", url),
        )

    async def post(
        self,
        url: str,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream: bool = False,
        logging_obj=None,
        files=None,
        content=None,
    ):
        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "params": params,
                "headers": headers or {},
                "timeout": timeout,
                "stream": stream,
                "content": content,
            }
        )
        raw = json if json is not None else (data if data is not None else content)
        payload = raw if isinstance(raw, dict) else json_module.loads(raw)
        job_name = payload["jobName"]
        job_arn = f"arn:aws:bedrock:us-west-2:941277531214:model-invocation-job/{job_name}"
        self.batch_jobs[job_arn] = {
            "jobArn": job_arn,
            "jobName": job_name,
            "modelId": payload["modelId"],
            "roleArn": payload["roleArn"],
            "status": "InProgress",
            "submitTime": "2026-06-02T03:50:00Z",
            "lastModifiedTime": "2026-06-02T03:55:00Z",
            "inputDataConfig": payload["inputDataConfig"],
            "outputDataConfig": payload["outputDataConfig"],
        }
        return httpx.Response(
            status_code=200,
            json={"jobArn": job_arn, "jobName": job_name, "status": "Submitted"},
            request=httpx.Request("POST", url),
        )


@pytest.mark.asyncio()
async def test_async_create_file():
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    litellm.turn_on_debug()
    print("Testing async create batch")

    file_name = "bedrock_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    capture_client = _CaptureAsyncHTTPHandler()
    with (
        patch.dict(os.environ, _BEDROCK_TEST_AWS_ENV),
        open(file_path, "rb") as batch_file,
    ):
        file_obj = await litellm.acreate_file(
            file=batch_file,
            purpose="batch",
            custom_llm_provider="bedrock",
            s3_bucket_name="litellm-proxy-941277531214",
            client=capture_client,
        )

    assert len(capture_client.put_calls) == 1
    put_call = capture_client.put_calls[0]
    assert put_call["url"].startswith(
        "https://s3.us-west-2.amazonaws.com/litellm-proxy-941277531214/"
    )
    assert "/litellm-bedrock-files-us.anthropic.claude-haiku-4-5-20251001-v1-0-" in (
        put_call["url"]
    )
    assert put_call["url"].endswith(".jsonl")
    assert put_call["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "recordId" in put_call["data"]
    assert file_obj.id.startswith(
        "s3://litellm-proxy-941277531214/litellm-bedrock-files-"
    )
    assert file_obj.filename.endswith(".jsonl")
