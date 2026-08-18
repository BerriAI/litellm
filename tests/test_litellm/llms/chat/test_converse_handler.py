import os
import sys
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.llms.bedrock.chat import BedrockConverseLLM
from litellm.llms.bedrock.chat.converse_handler import make_sync_call
from litellm.llms.bedrock.common_utils import _get_all_bedrock_regions
from litellm.llms.custom_httpx.http_handler import HTTPHandler

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


def test_encode_model_id_with_inference_profile():
    """
    Test instance profile is properly encoded when used as a model
    """
    test_model = "arn:aws:bedrock:us-east-1:12345678910:application-inference-profile/ujdtmcirjhevpi"
    expected_model = "arn%3Aaws%3Abedrock%3Aus-east-1%3A12345678910%3Aapplication-inference-profile%2Fujdtmcirjhevpi"
    bedrock_converse_llm = BedrockConverseLLM()
    returned_model = bedrock_converse_llm.encode_model_id(test_model)
    assert expected_model == returned_model


class TestBedrockRegionInModelPath:
    """
    Tests for region extraction from bedrock/{region}/{model} path format.

    When a user passes model="bedrock/ap-northeast-1/moonshotai.kimi-k2.5",
    get_llm_provider strips "bedrock/" and passes "ap-northeast-1/moonshotai.kimi-k2.5"
    to the converse handler. The handler must:
    1. Strip the region from modelId (so AWS gets "moonshotai.kimi-k2.5", not "ap-northeast-1%2Fmoonshotai.kimi-k2.5")
    2. Use the extracted region as aws_region_name for the API call
    """

    @pytest.mark.parametrize(
        "model,expected_model_id,expected_region",
        [
            # Region embedded in path — both modelId and region must be extracted
            (
                "ap-northeast-1/moonshotai.kimi-k2.5",
                "moonshotai.kimi-k2.5",
                "ap-northeast-1",
            ),
            (
                "us-east-1/moonshotai.kimi-k2.5",
                "moonshotai.kimi-k2.5",
                "us-east-1",
            ),
            (
                "us-west-2/anthropic.claude-haiku-4-5-20251001-v1:0",
                "anthropic.claude-haiku-4-5-20251001-v1%3A0",
                "us-west-2",
            ),
            # No region in path — modelId unchanged, no region injected
            (
                "moonshotai.kimi-k2.5",
                "moonshotai.kimi-k2.5",
                None,
            ),
            # Cross-region inference prefix (us., eu., ap.) — not a region path segment
            (
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "us.anthropic.claude-haiku-4-5-20251001-v1%3A0",
                None,
            ),
        ],
    )
    def test_region_and_model_id_extraction(
        self, model, expected_model_id, expected_region
    ):
        """
        Verify that completion() correctly extracts both modelId and aws_region_name
        from the bedrock/{region}/{model} path format.
        """
        bedrock_converse_llm = BedrockConverseLLM()
        optional_params: dict = {}

        # Simulate the modelId + region extraction logic from completion()
        _model_for_id = model
        _stripped = _model_for_id
        for rp in ["bedrock/converse/", "bedrock/", "converse/"]:
            if _stripped.startswith(rp):
                _stripped = _stripped[len(rp) :]
                break

        _region_from_model = None
        _potential_region = _stripped.split("/", 1)[0]
        if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
            _region_from_model = _potential_region
            _stripped = _stripped.split("/", 1)[1]
            _model_for_id = _stripped

        for _nova_prefix in ["nova-2/", "nova/"]:
            if _stripped.startswith(_nova_prefix):
                _model_for_id = _model_for_id.replace(_nova_prefix, "", 1)
                break

        model_id = bedrock_converse_llm.encode_model_id(model_id=_model_for_id)
        if _region_from_model is not None and "aws_region_name" not in optional_params:
            optional_params["aws_region_name"] = _region_from_model

        assert (
            model_id == expected_model_id
        ), f"modelId mismatch for {model!r}: got {model_id!r}, expected {expected_model_id!r}"
        assert (
            optional_params.get("aws_region_name") == expected_region
        ), f"region mismatch for {model!r}: got {optional_params.get('aws_region_name')!r}, expected {expected_region!r}"

    def test_explicit_aws_region_name_not_overridden(self):
        """
        If aws_region_name is already set in optional_params, the region in the
        model path must NOT override it.
        """
        bedrock_converse_llm = BedrockConverseLLM()
        optional_params = {"aws_region_name": "eu-west-1"}
        model = "ap-northeast-1/moonshotai.kimi-k2.5"

        _model_for_id = model
        _stripped = model
        _region_from_model = None
        _potential_region = _stripped.split("/", 1)[0]
        if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
            _region_from_model = _potential_region
            _stripped = _stripped.split("/", 1)[1]
            _model_for_id = _stripped

        model_id = bedrock_converse_llm.encode_model_id(model_id=_model_for_id)
        if _region_from_model is not None and "aws_region_name" not in optional_params:
            optional_params["aws_region_name"] = _region_from_model

        # modelId is still correctly stripped
        assert model_id == "moonshotai.kimi-k2.5"
        # explicitly set region is preserved
        assert optional_params["aws_region_name"] == "eu-west-1"


def _stream_completion_with_spied_iter_bytes(model: str, **kwargs) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **kwargs,
    )
    return mock_response.iter_bytes


def test_make_sync_call_does_not_rechunk_stream_by_default():
    """Re-chunking the event stream into fixed 1024-byte blocks holds small
    early events in httpx's ByteChunker until 1024 bytes accumulate, delaying
    time-to-first-chunk by the whole generation when Bedrock trickles bytes
    (e.g. buffered tool-use streams)."""
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
    )

    response.iter_bytes.assert_called_once_with(chunk_size=None)


def test_make_sync_call_honors_explicit_stream_chunk_size():
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse-stream",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-6",
        messages=[],
        logging_obj=MagicMock(),
        stream_chunk_size=2048,
    )

    response.iter_bytes.assert_called_once_with(chunk_size=2048)


def test_completion_plumbs_stream_chunk_size_through_converse():
    iter_bytes_spy = _stream_completion_with_spied_iter_bytes(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=None)

    iter_bytes_spy = _stream_completion_with_spied_iter_bytes(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        stream_chunk_size=2048,
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=2048)
