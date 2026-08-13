"""
Tests for Writer Palmyra X5 and X4 models on Bedrock Converse.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import BedrockModelInfo


def test_writer_palmyra_routes_to_converse():
    """
    Test that Writer Palmyra models route to converse API.
    """
    bedrock_model_info = BedrockModelInfo

    # Test base model routes to converse
    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/writer.palmyra-x5-v1:0"
    )
    assert bedrock_route == "converse"

    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/writer.palmyra-x4-v1:0"
    )
    assert bedrock_route == "converse"


def test_writer_palmyra_cross_region_routes_to_converse():
    """
    Test that Writer Palmyra models with cross-region inference prefix route to converse API.
    """
    bedrock_model_info = BedrockModelInfo

    # Test cross-region inference profile routes to converse
    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/us.writer.palmyra-x5-v1:0"
    )
    assert bedrock_route == "converse"

    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/us.writer.palmyra-x4-v1:0"
    )
    assert bedrock_route == "converse"


def test_writer_palmyra_base_model_extraction():
    """
    Test that base model is correctly extracted from Writer Palmyra cross-region models.
    """
    bedrock_model_info = BedrockModelInfo

    # Test us. prefix is stripped correctly
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us.writer.palmyra-x5-v1:0"
    )
    assert base_model == "writer.palmyra-x5-v1:0"

    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us.writer.palmyra-x4-v1:0"
    )
    assert base_model == "writer.palmyra-x4-v1:0"
