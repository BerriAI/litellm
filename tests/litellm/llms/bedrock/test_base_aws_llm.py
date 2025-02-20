import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import litellm


def test_boto3_init_tracer_wrapping():
    """
    Test that all boto3 initializations are wrapped in tracer.trace or @tracer.wrap

    Ensures observability of boto3 calls in litellm.
    """
    # Get the source code of base_aws_llm.py
    base_aws_llm_path = os.path.join(
        os.path.dirname(__file__), "../../../../litellm/llms/bedrock/base_aws_llm.py"
    )
    with open(base_aws_llm_path, "r") as f:
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
