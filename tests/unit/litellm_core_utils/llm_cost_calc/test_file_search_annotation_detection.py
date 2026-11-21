"""
File search detection must find file_citation annotations after they are
moved to message.provider_specific_fields["annotations"].

See https://github.com/BerriAI/litellm/issues/49126.
"""

from typing import Final

import litellm
from litellm.constants import OPENAI_FILE_SEARCH_COST_PER_1K_CALLS
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import ModelResponse

_FILE_CITATION: Final = {
    "type": "file_citation",
    "file_id": "file-abc123",
    "filename": "doc.pdf",
    "index": 0,
}


def _streamed_response(annotations: tuple[object, ...]) -> ModelResponse:
    chunks: Final = [
        litellm.ModelResponseStream(
            id="chatcmpl-1",
            model="gpt-4o",
            choices=[{"index": 0, "delta": {"role": "assistant", "content": "See the doc."}}],
        ),
        litellm.ModelResponseStream(
            id="chatcmpl-1",
            model="gpt-4o",
            choices=[
                {
                    "index": 0,
                    "delta": {"annotations": list(annotations)},
                    "finish_reason": "stop",
                }
            ],
        ),
    ]
    response: Final = litellm.stream_chunk_builder(chunks=chunks)
    assert isinstance(response, ModelResponse)
    return response


def _non_streamed_response(annotations: tuple[object, ...]) -> ModelResponse:
    response: Final = convert_to_model_response_object(
        response_object={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "See the doc.",
                        "annotations": list(annotations),
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        model_response_object=ModelResponse(),
    )
    assert isinstance(response, ModelResponse)
    return response


def test_streamed_file_citation_is_detected_as_file_search() -> None:
    response: Final = _streamed_response((_FILE_CITATION,))
    assert StandardBuiltInToolCostTracking.response_object_includes_file_search_call(response_object=response)


def test_non_streamed_file_citation_is_detected_as_file_search() -> None:
    response: Final = _non_streamed_response((_FILE_CITATION,))
    assert StandardBuiltInToolCostTracking.response_object_includes_file_search_call(response_object=response)


def test_url_citation_only_is_not_file_search() -> None:
    url_citation: Final = {
        "type": "url_citation",
        "url_citation": {"start_index": 0, "end_index": 3, "title": "t", "url": "https://example.com"},
    }
    response: Final = _streamed_response((url_citation,))
    assert not StandardBuiltInToolCostTracking.response_object_includes_file_search_call(response_object=response)


def test_streamed_file_citation_bills_file_search_cost() -> None:
    response: Final = _streamed_response((_FILE_CITATION,))
    cost: Final = StandardBuiltInToolCostTracking.get_cost_for_built_in_tools(
        model="gpt-4o",
        response_object=response,
        custom_llm_provider="openai",
        standard_built_in_tools_params={"file_search": {"type": "file_search", "vector_store_ids": ["vs_1"]}},
    )
    assert cost == OPENAI_FILE_SEARCH_COST_PER_1K_CALLS
