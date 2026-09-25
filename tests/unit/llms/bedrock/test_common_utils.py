from litellm.llms.bedrock.common_utils import stream_chunk_size_from
from litellm.types.litellm_params import CONTROL_PARAMS_KEY, LiteLLMControlParams


def test_stream_chunk_size_from_absent_control_params_is_none():
    assert stream_chunk_size_from({}) is None


def test_stream_chunk_size_from_reads_the_control_params():
    assert stream_chunk_size_from({CONTROL_PARAMS_KEY: LiteLLMControlParams(stream_chunk_size=64)}) == 64


def test_stream_chunk_size_from_ignores_a_flat_key_the_control_params_do_not_carry():
    assert stream_chunk_size_from({"stream_chunk_size": 64, CONTROL_PARAMS_KEY: LiteLLMControlParams()}) is None
