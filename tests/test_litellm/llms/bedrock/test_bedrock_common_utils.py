import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import BedrockModelInfo


# --------------------------------------------------------------------------- #
# get_bedrock_response_stream_shape lazy-load tests                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_bedrock_response_stream_shape_cache():
    """Prevent lru_cache leakage between tests in this module."""
    import litellm.llms.bedrock.common_utils as mod

    mod.get_bedrock_response_stream_shape.cache_clear()
    yield
    mod.get_bedrock_response_stream_shape.cache_clear()


def test_bedrock_response_stream_shape_lazy_loads_once():
    """
    get_bedrock_response_stream_shape() loads from botocore at most once per process.
    """
    from unittest.mock import MagicMock, patch

    import litellm.llms.bedrock.common_utils as mod

    sentinel = MagicMock()
    with patch.object(
        mod, "_load_bedrock_response_stream_shape", return_value=sentinel
    ) as mock_load:
        assert mod.get_bedrock_response_stream_shape() is sentinel
        assert mod.get_bedrock_response_stream_shape() is sentinel
        mock_load.assert_called_once()


def test_bedrock_response_stream_shape_loaded_on_first_access():
    """
    get_bedrock_response_stream_shape() loads once on first use.
    In a standard environment with botocore installed it must be non-None.
    """
    pytest.importorskip("botocore")
    from litellm.llms.bedrock.common_utils import get_bedrock_response_stream_shape

    assert get_bedrock_response_stream_shape() is not None


def test_bedrock_response_stream_shape_load_failure_returns_none():
    """
    If botocore's Loader raises (e.g. missing data files), _load_bedrock_response_stream_shape
    should return None rather than propagating the exception, so the module
    still imports cleanly.
    """
    from unittest.mock import patch

    import litellm.llms.bedrock.common_utils as mod

    pytest.importorskip("botocore")
    with patch(
        "botocore.loaders.Loader.load_service_model",
        side_effect=Exception("no data"),
    ):
        shape = mod._load_bedrock_response_stream_shape()
        assert shape is None


def test_bedrock_response_stream_shape_is_structure_shape():
    """
    The loaded shape should be the botocore StructureShape for ResponseStream,
    not a plain dict or any other type.
    """
    pytest.importorskip("botocore")
    from botocore.model import StructureShape

    from litellm.llms.bedrock.common_utils import get_bedrock_response_stream_shape

    loaded_shape = get_bedrock_response_stream_shape()
    assert (
        loaded_shape is not None
    ), "get_bedrock_response_stream_shape() is None — botocore may not be installed"
    shape: StructureShape = loaded_shape
    assert isinstance(shape, StructureShape)
    assert shape.name == "ResponseStream"


def test_bedrock_response_stream_shape_same_object_across_calls():
    """
    Repeated calls must return the identical cached object.
    """
    from litellm.llms.bedrock.common_utils import get_bedrock_response_stream_shape

    first = get_bedrock_response_stream_shape()
    second = get_bedrock_response_stream_shape()
    assert first is second


def test_bedrock_event_stream_decoder_base_uses_module_shape():
    """
    BedrockEventStreamDecoderBase instances no longer carry their own
    per-instance cache — _parse_message_from_event uses the module constant
    directly, so there is no instance-level _response_stream_shape_cache attr.
    """
    from litellm.llms.bedrock.common_utils import BedrockEventStreamDecoderBase

    decoder_a = BedrockEventStreamDecoderBase()
    decoder_b = BedrockEventStreamDecoderBase()

    assert "_response_stream_shape_cache" not in decoder_a.__dict__
    assert "_response_stream_shape_cache" not in decoder_b.__dict__


def test_bedrock_parse_message_from_event_raises_on_none_shape():
    """
    When get_bedrock_response_stream_shape() returns None (botocore unavailable),
    _parse_message_from_event must raise BedrockError before touching the
    botocore parser — not an opaque AttributeError from inside botocore.
    """
    from unittest.mock import MagicMock, patch

    import litellm.llms.bedrock.common_utils as mod
    from litellm.llms.bedrock.common_utils import (
        BedrockError,
        BedrockEventStreamDecoderBase,
    )

    decoder = BedrockEventStreamDecoderBase.__new__(BedrockEventStreamDecoderBase)
    decoder.parser = MagicMock()
    mock_event = MagicMock()

    with patch.object(mod, "get_bedrock_response_stream_shape", return_value=None):
        with pytest.raises(BedrockError) as exc_info:
            decoder._parse_message_from_event(mock_event)

    assert exc_info.value.status_code == 500
    assert "botocore" in str(exc_info.value.message).lower()
    # The botocore parser must never have been called
    mock_event.to_response_dict.assert_not_called()


def test_deepseek_cris():
    """
    Test that DeepSeek models with cross-region inference prefix use converse route
    """
    bedrock_model_info = BedrockModelInfo
    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/us.deepseek.r1-v1:0"
    )
    assert bedrock_route == "converse"


def test_govcloud_cross_region_inference_prefix():
    """
    Test that GovCloud models with cross-region inference prefix (us-gov.) are parsed correctly
    """
    bedrock_model_info = BedrockModelInfo

    # Test us-gov prefix is stripped correctly for Claude models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert base_model == "anthropic.claude-haiku-4-5-20251001-v1:0"

    # Test us-gov prefix is stripped correctly for different Claude versions
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert base_model == "anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Test us-gov prefix is stripped correctly for Haiku models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert base_model == "anthropic.claude-3-haiku-20240307-v1:0"

    # Test us-gov prefix is stripped correctly for Meta models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.meta.llama3-8b-instruct-v1:0"
    )
    assert base_model == "meta.llama3-8b-instruct-v1:0"


def test_context_window_suffix_stripped_for_cost_lookup():
    """
    Test that [1m], [200k] etc. context window suffixes are stripped from
    Bedrock model names before cost lookup.

    Models configured like `bedrock/us.anthropic.claude-opus-4-6-v1[1m]`
    should resolve to the base model name so pricing can be found.
    """
    from litellm.llms.bedrock.common_utils import get_bedrock_base_model

    assert (
        get_bedrock_base_model("us.anthropic.claude-opus-4-6-v1[1m]")
        == "anthropic.claude-opus-4-6-v1"
    )
    assert (
        get_bedrock_base_model("us.anthropic.claude-sonnet-4-6[1m]")
        == "anthropic.claude-sonnet-4-6"
    )
    assert (
        get_bedrock_base_model("global.anthropic.claude-opus-4-5-20251101-v1:0[1m]")
        == "anthropic.claude-opus-4-5-20251101-v1:0"
    )
    # Ensure models without suffix are unaffected
    assert (
        get_bedrock_base_model("us.anthropic.claude-opus-4-6-v1")
        == "anthropic.claude-opus-4-6-v1"
    )
    # Ensure :51k throughput suffix still works
    assert (
        get_bedrock_base_model("anthropic.claude-3-5-sonnet-20241022-v2:0:51k")
        == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
