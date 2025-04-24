import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from botocore.credentials import Credentials

import litellm
from litellm.llms.bedrock.base_aws_llm import (
    AwsAuthError,
    BaseAWSLLM,
    Boto3CredentialsInfo,
)

# Global variable for the base_aws_llm.py file path

BASE_AWS_LLM_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../litellm/llms/bedrock/base_aws_llm.py"
)


def test_boto3_init_tracer_wrapping():
    """
    Test that all boto3 initializations are wrapped in tracer.trace or @tracer.wrap

    Ensures observability of boto3 calls in litellm.
    """
    # Get the source code of base_aws_llm.py
    with open(BASE_AWS_LLM_PATH, "r") as f:
        content = f.read()

    # List all boto3 initialization patterns we want to check
    boto3_init_patterns = ["boto3.client", "boto3.Session"]

    lines = content.split("\n")
    # Check each boto3 initialization is wrapped in tracer.trace
    for line_number, line in enumerate(lines, 1):
        for pattern in boto3_init_patterns:
            if pattern in line:
                # Look back up to 5 lines for decorator or trace block
                start_line = max(0, line_number - 5)
                context_lines = lines[start_line:line_number]

                has_trace = (
                    "tracer.trace" in line
                    or any("tracer.trace" in prev_line for prev_line in context_lines)
                    or any("@tracer.wrap" in prev_line for prev_line in context_lines)
                )

                if not has_trace:
                    print(f"\nContext for line {line_number}:")
                    for i, ctx_line in enumerate(context_lines, start=start_line + 1):
                        print(f"{i}: {ctx_line}")

                assert (
                    has_trace
                ), f"boto3 initialization '{pattern}' on line {line_number} is not wrapped with tracer.trace or @tracer.wrap"


def test_auth_functions_tracer_wrapping():
    """
    Test that all _auth functions in base_aws_llm.py are wrapped with @tracer.wrap

    Ensures observability of AWS authentication calls in litellm.
    """
    # Get the source code of base_aws_llm.py
    with open(BASE_AWS_LLM_PATH, "r") as f:
        content = f.read()

    lines = content.split("\n")
    # Check each line for _auth function definitions
    for line_number, line in enumerate(lines, 1):
        if line.strip().startswith("def _auth_"):
            # Look back up to 2 lines for the @tracer.wrap decorator
            start_line = max(0, line_number - 2)
            context_lines = lines[start_line:line_number]

            has_tracer_wrap = any(
                "@tracer.wrap" in prev_line for prev_line in context_lines
            )

            if not has_tracer_wrap:
                print(f"\nContext for line {line_number}:")
                for i, ctx_line in enumerate(context_lines, start=start_line + 1):
                    print(f"{i}: {ctx_line}")

            assert (
                has_tracer_wrap
            ), f"Auth function on line {line_number} is not wrapped with @tracer.wrap: {line.strip()}"


def test_loading_bearer_token_from_env():
    aws_region_name = "us"
    endpoint_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"
    data = ""
    headers = {}

    os.environ['AWS_SECRET_ACCESS_KEY'] = "fake_aws_secret_access_key"
    os.environ['AWS_ACCESS_KEY_ID'] = "fake_aws_access_key_id"

    extra_headers = {}
    base_aws_llm = BaseAWSLLM()
    credentials = base_aws_llm.get_credentials()
    aws_prep_req = base_aws_llm.get_request_headers(credentials, aws_region_name, extra_headers, endpoint_url, data, headers)
    assert 'Authorization' in aws_prep_req.headers.keys()
    assert 'Bearer' not in aws_prep_req.headers["Authorization"]

    os.environ['AWS_EXTRA_HEADERS_AUTH_BEARER_TOKEN'] = "fake_token"
    extra_headers = {}
    base_aws_llm = BaseAWSLLM()
    credentials = base_aws_llm.get_credentials()
    aws_prep_req = base_aws_llm.get_request_headers(credentials, aws_region_name, extra_headers, endpoint_url, data, headers)
    assert 'Authorization' in aws_prep_req.headers.keys()
    assert aws_prep_req.headers['Authorization'] == "Bearer fake_token"
