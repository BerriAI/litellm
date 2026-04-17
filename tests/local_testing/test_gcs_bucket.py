import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import json
import logging
import tempfile
from litellm._uuid import uuid
from datetime import datetime

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.gcs_bucket.gcs_bucket import (
    GCSBucketLogger,
    StandardLoggingPayload,
)
from litellm.types.utils import StandardCallbackDynamicParams
from litellm.types.integrations.gcs_bucket import GCSLoggingConfig
from unittest.mock import patch, AsyncMock, MagicMock

verbose_logger.setLevel(logging.DEBUG)


def _make_mock_gcs_logging_config():
    return GCSLoggingConfig(
        bucket_name="test-bucket",
        vertex_instance=MagicMock(),
        path_service_account=None,
    )


@pytest.mark.asyncio
async def test_aaabasic_gcs_logger():
    os.environ["GCS_FLUSH_INTERVAL"] = "1"
    os.environ["GCS_USE_BATCHED_LOGGING"] = "false"
    os.environ["GCS_BUCKET_NAME"] = "test-bucket"

    captured_payloads = []

    async def mock_log_json_data_on_gcs(
        self, headers, bucket_name, object_name, logging_payload
    ):
        captured_payloads.append(
            {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "logging_payload": logging_payload,
            }
        )
        return {"kind": "storage#object", "name": object_name}

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch.object(
            GCSBucketLogger,
            "construct_request_headers",
            new_callable=AsyncMock,
            return_value={"Authorization": "Bearer mock_token"},
        ),
        patch.object(
            GCSBucketLogger,
            "get_gcs_logging_config",
            new_callable=AsyncMock,
            return_value=_make_mock_gcs_logging_config(),
        ),
        patch.object(
            GCSBucketLogger,
            "_log_json_data_on_gcs",
            mock_log_json_data_on_gcs,
        ),
    ):
        gcs_logger = GCSBucketLogger()

        litellm.callbacks = [gcs_logger]
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=[{"role": "user", "content": "This is a test"}],
            max_tokens=10,
            user="ishaan-2",
            mock_response="Hi!",
            metadata={
                "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"],
                "user_api_key": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
                "user_api_key_alias": None,
                "user_api_end_user_max_budget": None,
                "litellm_api_version": "0.0.0",
                "global_max_parallel_requests": None,
                "user_api_key_user_id": "116544810872468347480",
                "user_api_key_org_id": None,
                "user_api_key_team_id": None,
                "user_api_key_team_alias": None,
                "user_api_key_metadata": {},
                "requester_ip_address": "127.0.0.1",
                "requester_metadata": {"foo": "bar"},
                "spend_logs_metadata": {"hello": "world"},
                "headers": {
                    "content-type": "application/json",
                    "user-agent": "PostmanRuntime/7.32.3",
                    "accept": "*/*",
                    "postman-token": "92300061-eeaa-423b-a420-0b44896ecdc4",
                    "host": "localhost:4000",
                    "accept-encoding": "gzip, deflate, br",
                    "connection": "keep-alive",
                    "content-length": "163",
                },
                "endpoint": "http://localhost:4000/chat/completions",
                "model_group": "gpt-3.5-turbo",
                "model_info": {
                    "id": "4bad40a1eb6bebd1682800f16f44b9f06c52a6703444c99c7f9f32e9de3693b4",
                    "db_model": False,
                },
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                "caching_groups": None,
                "raw_request": "\n\nPOST Request Sent from LiteLLM:\ncurl -X POST \\\nhttps://openai-gpt-4-test-v-1.openai.azure.com//openai/ \\\n-H 'Authorization: *****' \\\n-d '{'model': 'chatgpt-v-3', 'messages': [{'role': 'system', 'content': 'you are a helpful assistant.\\n'}, {'role': 'user', 'content': 'bom dia'}], 'stream': False, 'max_tokens': 10, 'user': '116544810872468347480', 'extra_body': {}}'\n",
            },
        )

        print("response", response)

        await asyncio.sleep(3)

        assert (
            len(captured_payloads) == 1
        ), f"Expected 1 GCS upload, got {len(captured_payloads)}"

        gcs_payload = captured_payloads[0]["logging_payload"]

        assert gcs_payload["model"] == "gpt-3.5-turbo"
        assert gcs_payload["messages"] == [
            {"role": "user", "content": "This is a test"}
        ]

        assert gcs_payload["response"]["choices"][0]["message"]["content"] == "Hi!"

        assert gcs_payload["response_cost"] > 0.0

        assert gcs_payload["status"] == "success"

        assert (
            gcs_payload["metadata"]["user_api_key_hash"]
            == "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
        )
        assert (
            gcs_payload["metadata"]["user_api_key_user_id"] == "116544810872468347480"
        )

        assert gcs_payload["metadata"]["requester_metadata"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_basic_gcs_logger_failure():
    os.environ["GCS_FLUSH_INTERVAL"] = "1"
    os.environ["GCS_USE_BATCHED_LOGGING"] = "false"
    os.environ["GCS_BUCKET_NAME"] = "test-bucket"

    captured_payloads = []

    async def mock_log_json_data_on_gcs(
        self, headers, bucket_name, object_name, logging_payload
    ):
        captured_payloads.append(
            {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "logging_payload": logging_payload,
            }
        )
        return {"kind": "storage#object", "name": object_name}

    gcs_log_id = f"failure-test-{uuid.uuid4().hex}"

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch.object(
            GCSBucketLogger,
            "construct_request_headers",
            new_callable=AsyncMock,
            return_value={"Authorization": "Bearer mock_token"},
        ),
        patch.object(
            GCSBucketLogger,
            "get_gcs_logging_config",
            new_callable=AsyncMock,
            return_value=_make_mock_gcs_logging_config(),
        ),
        patch.object(
            GCSBucketLogger,
            "_log_json_data_on_gcs",
            mock_log_json_data_on_gcs,
        ),
    ):
        gcs_logger = GCSBucketLogger()

        litellm.callbacks = [gcs_logger]

        try:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[{"role": "user", "content": "This is a test"}],
                max_tokens=10,
                user="ishaan-2",
                mock_response=litellm.BadRequestError(
                    model="gpt-3.5-turbo",
                    message="Error: 400: Bad Request: Invalid API key, please check your API key and try again.",
                    llm_provider="openai",
                ),
                metadata={
                    "gcs_log_id": gcs_log_id,
                    "tags": ["model-anthropic-claude-v2.1", "app-ishaan-prod"],
                    "user_api_key": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
                    "user_api_key_alias": None,
                    "user_api_end_user_max_budget": None,
                    "litellm_api_version": "0.0.0",
                    "global_max_parallel_requests": None,
                    "user_api_key_user_id": "116544810872468347480",
                    "user_api_key_org_id": None,
                    "user_api_key_team_id": None,
                    "user_api_key_team_alias": None,
                    "user_api_key_metadata": {},
                    "requester_ip_address": "127.0.0.1",
                    "spend_logs_metadata": {"hello": "world"},
                    "headers": {
                        "content-type": "application/json",
                        "user-agent": "PostmanRuntime/7.32.3",
                        "accept": "*/*",
                        "postman-token": "92300061-eeaa-423b-a420-0b44896ecdc4",
                        "host": "localhost:4000",
                        "accept-encoding": "gzip, deflate, br",
                        "connection": "keep-alive",
                        "content-length": "163",
                    },
                    "endpoint": "http://localhost:4000/chat/completions",
                    "model_group": "gpt-3.5-turbo",
                    "model_info": {
                        "id": "4bad40a1eb6bebd1682800f16f44b9f06c52a6703444c99c7f9f32e9de3693b4",
                        "db_model": False,
                    },
                    "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                    "caching_groups": None,
                    "raw_request": "\n\nPOST Request Sent from LiteLLM:\ncurl -X POST \\\nhttps://openai-gpt-4-test-v-1.openai.azure.com//openai/ \\\n-H 'Authorization: *****' \\\n-d '{'model': 'chatgpt-v-3', 'messages': [{'role': 'system', 'content': 'you are a helpful assistant.\\n'}, {'role': 'user', 'content': 'bom dia'}], 'stream': False, 'max_tokens': 10, 'user': '116544810872468347480', 'extra_body': {}}'\n",
                },
            )
        except Exception:
            pass

        await asyncio.sleep(3)

        assert (
            len(captured_payloads) == 1
        ), f"Expected 1 GCS upload, got {len(captured_payloads)}"

        gcs_payload = captured_payloads[0]["logging_payload"]

        assert gcs_payload["model"] == "gpt-3.5-turbo"
        assert gcs_payload["messages"] == [
            {"role": "user", "content": "This is a test"}
        ]

        assert gcs_payload["response_cost"] == 0
        assert gcs_payload["status"] == "failure"

        assert (
            gcs_payload["metadata"]["user_api_key_hash"]
            == "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
        )
        assert (
            gcs_payload["metadata"]["user_api_key_user_id"] == "116544810872468347480"
        )
