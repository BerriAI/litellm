"""
Tests for Gemini code execution parts (executableCode / codeExecutionResult).

When the `codeExecution` tool is enabled, Gemini writes Python, Google runs it
server-side, and the response carries `executableCode` + `codeExecutionResult`
parts alongside the regular `text` parts.

Neither part type was handled by the response transformation, so both were
dropped: the console output - the only trustworthy source for the computed
values - never reached the caller.

They must now be surfaced in provider_specific_fields["code_execution"], in
wire order, for both streaming and non-streaming responses.
"""

from unittest.mock import MagicMock

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator,
    VertexGeminiConfig,
)
from litellm.types.llms.vertex_ai import HttpxPartType
from litellm.types.utils import ModelResponse

CODE = "import numpy as np\nprint(np.linalg.solve(A, B))"
OUTPUT = "x = 13.835714\ny = -6.000000\n"


def _make_logging_obj(**kwargs):
    """Create a minimal mock logging object for ModelResponseIterator."""
    logging_obj = MagicMock()
    logging_obj.optional_params = kwargs.get("optional_params", {})
    return logging_obj


def _code_execution_parts() -> list[HttpxPartType]:
    return [
        {"text": "Let me compute this with Python."},
        {"executableCode": {"language": "PYTHON", "code": CODE}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": OUTPUT}},
        {"text": "The exact solution is x = 13.84, y = -6.00."},
    ]


class TestExtractCodeExecutionParts:
    """Test _extract_code_execution_parts from response parts."""

    def test_extracts_code_and_result_in_wire_order(self):
        result = VertexGeminiConfig._extract_code_execution_parts(_code_execution_parts())

        # a tuple, not a list: the entries are handed to the caller and must not be
        # grown or rewritten downstream
        assert result == (
            {"type": "executable_code", "language": "PYTHON", "code": CODE},
            {
                "type": "code_execution_result",
                "outcome": "OUTCOME_OK",
                "output": OUTPUT,
            },
        )

    def test_returns_none_when_no_code_execution_parts(self):
        parts: list[HttpxPartType] = [
            {"text": "Hello world"},
            {"functionCall": {"name": "get_weather", "args": {"location": "Paris"}}},
        ]

        assert VertexGeminiConfig._extract_code_execution_parts(parts) is None

    def test_returns_none_for_empty_parts(self):
        assert VertexGeminiConfig._extract_code_execution_parts([]) is None

    def test_handles_failed_execution(self):
        """A failed run still carries its outcome and stderr - it must not be dropped."""
        parts: list[HttpxPartType] = [
            {
                "codeExecutionResult": {
                    "outcome": "OUTCOME_FAILED",
                    "output": "ZeroDivisionError: division by zero",
                }
            }
        ]

        result = VertexGeminiConfig._extract_code_execution_parts(parts)

        assert result is not None
        assert result[0]["outcome"] == "OUTCOME_FAILED"
        assert "ZeroDivisionError" in result[0]["output"]

    def test_handles_multiple_execution_rounds(self):
        """Gemini can iterate: code, result, code, result. Order must be preserved."""
        parts: list[HttpxPartType] = [
            {"executableCode": {"language": "PYTHON", "code": "print(1)"}},
            {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1\n"}},
            {"executableCode": {"language": "PYTHON", "code": "print(2)"}},
            {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "2\n"}},
        ]

        result = VertexGeminiConfig._extract_code_execution_parts(parts)

        assert result is not None
        assert [item["type"] for item in result] == [
            "executable_code",
            "code_execution_result",
            "executable_code",
            "code_execution_result",
        ]
        assert result[2]["code"] == "print(2)"

    def test_missing_fields_do_not_raise(self):
        """Vertex may omit `language`; extraction must stay defensive."""
        parts: list[HttpxPartType] = [
            {"executableCode": {"code": "print(1)"}},  # type: ignore[typeddict-item]
            {"codeExecutionResult": {"outcome": "OUTCOME_OK"}},  # type: ignore[typeddict-item]
        ]

        result = VertexGeminiConfig._extract_code_execution_parts(parts)

        assert result is not None
        assert result[0]["language"] is None
        assert result[1]["output"] is None


class TestCodeExecutionInResponse:
    """Test that code execution parts reach the transformed response."""

    def test_non_streaming_response_carries_code_execution(self):
        candidates = [
            {
                "content": {"role": "model", "parts": _code_execution_parts()},
                "finishReason": "STOP",
            }
        ]
        model_response = ModelResponse()

        VertexGeminiConfig._process_candidates(
            candidates,
            model_response,
            standard_optional_params={},
        )

        # _process_candidates appends to the (pre-seeded) choices list
        choice = model_response.choices[-1]
        message = choice.message
        code_execution = message.provider_specific_fields["code_execution"]

        assert code_execution[0]["code"] == CODE
        assert code_execution[1]["output"] == OUTPUT
        # text parts keep their existing behaviour: concatenated into content
        assert message.content == ("Let me compute this with Python.The exact solution is x = 13.84, y = -6.00.")
        assert choice.finish_reason == "stop"

    def test_streaming_chunk_carries_code_execution(self):
        iterator = ModelResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
            logging_obj=_make_logging_obj(),
        )

        chunk = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": _code_execution_parts()},
                    "index": 0,
                }
            ]
        }

        response = iterator.chunk_parser(chunk)

        assert response is not None
        code_execution = response.choices[0].delta.provider_specific_fields["code_execution"]
        assert code_execution[1]["output"] == OUTPUT

    def test_response_without_code_execution_is_unchanged(self):
        """Backwards compatibility: no code execution parts, no new field."""
        candidates = [
            {
                "content": {"role": "model", "parts": [{"text": "Hello world"}]},
                "finishReason": "STOP",
            }
        ]
        model_response = ModelResponse()

        VertexGeminiConfig._process_candidates(
            candidates,
            model_response,
            standard_optional_params={},
        )

        message = model_response.choices[-1].message
        assert message.content == "Hello world"
        provider_specific_fields = message.provider_specific_fields or {}
        assert "code_execution" not in provider_specific_fields
