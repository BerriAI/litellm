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


def test_get_aws_region_name_boto3_fallback():
    """
    Test the boto3 session fallback logic in _get_aws_region_name method.

    This tests the specific code block that tries to get the region from boto3.Session()
    when aws_region_name is None and not found in environment variables.
    """
    base_aws_llm = BaseAWSLLM()

    # Test case 1: boto3.Session() returns a configured region
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:
        mock_get_secret.return_value = None  # No region in env vars

        with patch("boto3.Session") as mock_boto3_session:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_boto3_session.return_value = mock_session

            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "us-east-1"
            mock_boto3_session.assert_called_once()

    # Test case 2: boto3.Session() returns None for region (should default to us-west-2)
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:
        mock_get_secret.return_value = None  # No region in env vars

        with patch("boto3.Session") as mock_boto3_session:
            mock_session = MagicMock()
            mock_session.region_name = None
            mock_boto3_session.return_value = mock_session

            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "us-west-2"
            mock_boto3_session.assert_called_once()

    # Test case 3: boto3 import/session creation raises exception (should default to us-west-2)
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:
        mock_get_secret.return_value = None  # No region in env vars

        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.side_effect = Exception("boto3 not available")

            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "us-west-2"
            mock_boto3_session.assert_called_once()

    # Test case 4: aws_region_name is provided in optional_params (should not use boto3)
    with patch("boto3.Session") as mock_boto3_session:
        optional_params = {"aws_region_name": "eu-west-1"}
        result = base_aws_llm._get_aws_region_name(optional_params)

        assert result == "eu-west-1"
        mock_boto3_session.assert_not_called()

    # Test case 5: aws_region_name found in environment variables (should not use boto3)
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:

        def side_effect(key, default=None):
            if key == "AWS_REGION_NAME":
                return "ap-southeast-1"
            return default

        mock_get_secret.side_effect = side_effect

        with patch("boto3.Session") as mock_boto3_session:
            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "ap-southeast-1"
            mock_boto3_session.assert_not_called()
