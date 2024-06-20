import sys, os
import traceback, uuid
from dotenv import load_dotenv
from fastapi import Request
from fastapi.routing import APIRoute

load_dotenv()
import os, io, time

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm, asyncio
import json
import datetime
from litellm.proxy.utils import (
    get_logging_payload,
    SpendLogsPayload,
    SpendLogsMetadata,
)  # noqa: E402


def test_spend_logs_payload():
    """
    Ensure only expected values are logged in spend logs payload.
    """

    input_args: dict = {
        "kwargs": {
            "model": "chatgpt-v-2",
            "messages": [
                {"role": "system", "content": "you are a helpful assistant.\n"},
                {"role": "user", "content": "bom dia"},
            ],
            "optional_params": {
                "stream": False,
                "max_tokens": 10,
                "user": "116544810872468347480",
                "extra_body": {},
            },
            "litellm_params": {
                "acompletion": True,
                "api_key": "23c217a5b59f41b6b7a198017f4792f2",
                "force_timeout": 600,
                "logger_fn": None,
                "verbose": False,
                "custom_llm_provider": "azure",
                "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com//openai/",
                "litellm_call_id": "b9929bf6-7b80-4c8c-b486-034e6ac0c8b7",
                "model_alias_map": {},
                "completion_call_id": None,
                "metadata": {
                    "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
                    "user_api_key_alias": None,
                    "user_api_end_user_max_budget": None,
                    "litellm_api_version": "0.0.0",
                    "global_max_parallel_requests": None,
                    "user_api_key_user_id": "116544810872468347480",
                    "user_api_key_org_id": None,
                    "user_api_key_team_id": None,
                    "user_api_key_team_alias": None,
                    "user_api_key_metadata": {},
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
                    "deployment": "azure/chatgpt-v-2",
                    "model_info": {
                        "id": "4bad40a1eb6bebd1682800f16f44b9f06c52a6703444c99c7f9f32e9de3693b4",
                        "db_model": False,
                    },
                    "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
                    "caching_groups": None,
                    "raw_request": "\n\nPOST Request Sent from LiteLLM:\ncurl -X POST \\\nhttps://openai-gpt-4-test-v-1.openai.azure.com//openai/ \\\n-H 'Authorization: *****' \\\n-d '{'model': 'chatgpt-v-2', 'messages': [{'role': 'system', 'content': 'you are a helpful assistant.\\n'}, {'role': 'user', 'content': 'bom dia'}], 'stream': False, 'max_tokens': 10, 'user': '116544810872468347480', 'extra_body': {}}'\n",
                },
                "model_info": {
                    "id": "4bad40a1eb6bebd1682800f16f44b9f06c52a6703444c99c7f9f32e9de3693b4",
                    "db_model": False,
                },
                "proxy_server_request": {
                    "url": "http://localhost:4000/chat/completions",
                    "method": "POST",
                    "headers": {
                        "content-type": "application/json",
                        "authorization": "Bearer sk-1234",
                        "user-agent": "PostmanRuntime/7.32.3",
                        "accept": "*/*",
                        "postman-token": "92300061-eeaa-423b-a420-0b44896ecdc4",
                        "host": "localhost:4000",
                        "accept-encoding": "gzip, deflate, br",
                        "connection": "keep-alive",
                        "content-length": "163",
                    },
                    "body": {
                        "messages": [
                            {
                                "role": "system",
                                "content": "you are a helpful assistant.\n",
                            },
                            {"role": "user", "content": "bom dia"},
                        ],
                        "model": "gpt-3.5-turbo",
                        "max_tokens": 10,
                    },
                },
                "preset_cache_key": None,
                "no-log": False,
                "stream_response": {},
                "input_cost_per_token": None,
                "input_cost_per_second": None,
                "output_cost_per_token": None,
                "output_cost_per_second": None,
            },
            "start_time": datetime.datetime(2024, 6, 7, 12, 43, 30, 307665),
            "stream": False,
            "user": "116544810872468347480",
            "call_type": "acompletion",
            "litellm_call_id": "b9929bf6-7b80-4c8c-b486-034e6ac0c8b7",
            "completion_start_time": datetime.datetime(2024, 6, 7, 12, 43, 30, 954146),
            "max_tokens": 10,
            "extra_body": {},
            "custom_llm_provider": "azure",
            "input": [
                {"role": "system", "content": "you are a helpful assistant.\n"},
                {"role": "user", "content": "bom dia"},
            ],
            "api_key": "1234",
            "original_response": "",
            "additional_args": {
                "headers": {"Authorization": "Bearer 1234"},
                "api_base": "openai-gpt-4-test-v-1.openai.azure.com",
                "acompletion": True,
                "complete_input_dict": {
                    "model": "chatgpt-v-2",
                    "messages": [
                        {"role": "system", "content": "you are a helpful assistant.\n"},
                        {"role": "user", "content": "bom dia"},
                    ],
                    "stream": False,
                    "max_tokens": 10,
                    "user": "116544810872468347480",
                    "extra_body": {},
                },
            },
            "log_event_type": "post_api_call",
            "end_time": datetime.datetime(2024, 6, 7, 12, 43, 30, 954146),
            "cache_hit": None,
            "response_cost": 2.4999999999999998e-05,
        },
        "response_obj": litellm.ModelResponse(
            id="chatcmpl-9XZmkzS1uPhRCoVdGQvBqqIbSgECt",
            choices=[
                litellm.Choices(
                    finish_reason="length",
                    index=0,
                    message=litellm.Message(
                        content="Bom dia! Como posso ajudar você", role="assistant"
                    ),
                )
            ],
            created=1717789410,
            model="gpt-35-turbo",
            object="chat.completion",
            system_fingerprint=None,
            usage=litellm.Usage(
                completion_tokens=10, prompt_tokens=20, total_tokens=30
            ),
        ),
        "start_time": datetime.datetime(2024, 6, 7, 12, 43, 30, 308604),
        "end_time": datetime.datetime(2024, 6, 7, 12, 43, 30, 954146),
        "end_user_id": None,
    }

    payload: SpendLogsPayload = get_logging_payload(**input_args)

    # Define the expected metadata keys
    expected_metadata_keys = SpendLogsMetadata.__annotations__.keys()

    # Validate only specified metadata keys are logged
    assert "metadata" in payload
    assert isinstance(payload["metadata"], str)
    payload["metadata"] = json.loads(payload["metadata"])
    assert set(payload["metadata"].keys()) == set(expected_metadata_keys)
