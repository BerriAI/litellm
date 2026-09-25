import pytest

from litellm.llms.bedrock.common_utils import BedrockError, stream_chunk_size_from


def test_stream_chunk_size_from_absent_is_none():
    assert stream_chunk_size_from({}) is None


def test_stream_chunk_size_from_int_is_returned():
    assert stream_chunk_size_from({"stream_chunk_size": 64}) == 64


@pytest.mark.parametrize("bad_value", ["64", 6.4, True])
def test_stream_chunk_size_from_rejects_non_int_with_400(bad_value):
    with pytest.raises(BedrockError) as excinfo:
        stream_chunk_size_from({"stream_chunk_size": bad_value})

    assert excinfo.value.status_code == 400
    assert repr(bad_value) in excinfo.value.message
