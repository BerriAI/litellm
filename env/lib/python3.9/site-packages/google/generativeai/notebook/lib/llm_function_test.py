# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unittest for llm_function."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from absl.testing import absltest
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_inputs_source
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import model as model_lib


LLMCompareFunction = llm_function.LLMCompareFunction
LLMFunctionImpl = llm_function.LLMFunctionImpl
LLMFnOutputs = llmfn_outputs.LLMFnOutputs
LLMFnOutputRow = llmfn_output_row.LLMFnOutputRow
LLMFnOutputRowView = llmfn_output_row.LLMFnOutputRowView


class _MockModel(model_lib.AbstractModel):
    """Mock model that returns a caller-provided result."""

    def __init__(self, mock_results: Sequence[str]):
        self._mock_results = mock_results

    def call_model(
        self,
        model_input: str,
        model_args: model_lib.ModelArguments | None = None,
    ) -> model_lib.ModelResults:
        return model_lib.ModelResults(model_input=model_input, text_results=self._mock_results)


class _MockInputsSource(llmfn_inputs_source.LLMFnInputsSource):
    def _to_normalized_inputs_impl(
        self,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        return [
            {"word_one": "apple", "word_two": "banana"},
            {"word_one": "australia", "word_two": "brazil"},
        ], lambda: None


class LLMFunctionBasicTest(absltest.TestCase):
    """Test basic functionality such as execution and input-handling."""

    def _test_is_callable(
        self,
        llm_fn: Callable[[Sequence[tuple[str, str]] | None], LLMFnOutputs],
    ) -> LLMFnOutputs:
        return llm_fn(None)

    def test_run(self):
        llm_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["the opposite of hot is"],
        )
        results = self._test_is_callable(llm_fn)
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["the opposite of hot is"],
            "text_result": ["the opposite of hot is"],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_inputs(self):
        llm_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=[
                "A for {word_one}, B for {word_two}, C for",
                "if A is to {word_one} as B is to {word_two}, then C is",
            ],
        )
        results = llm_fn(
            inputs={
                "word_one": ["apple", "australia"],
                "word_two": ["banana", "brazil"],
            }
        )
        expected_results = {
            "Prompt Num": [0, 0, 1, 1],
            "Input Num": [0, 1, 0, 1],
            "Result Num": [0, 0, 0, 0],
            "Prompt": [
                "A for apple, B for banana, C for",
                "A for australia, B for brazil, C for",
                "if A is to apple as B is to banana, then C is",
                "if A is to australia as B is to brazil, then C is",
            ],
            "text_result": [
                "A for apple, B for banana, C for",
                "A for australia, B for brazil, C for",
                "if A is to apple as B is to banana, then C is",
                "if A is to australia as B is to brazil, then C is",
            ],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_inputs_source(self):
        llm_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=[
                "A for {word_one}, B for {word_two}, C for",
            ],
        )
        results = llm_fn(_MockInputsSource())
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt": [
                "A for apple, B for banana, C for",
                "A for australia, B for brazil, C for",
            ],
            "text_result": [
                "A for apple, B for banana, C for",
                "A for australia, B for brazil, C for",
            ],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_one_prompt_many_results(self):
        llm_fn = LLMFunctionImpl(
            model=_MockModel(mock_results=["cold", "cold", "cold"]),
            prompts=["The opposite of hot is"],
        )
        results = llm_fn()
        expected_results = {
            "Prompt Num": [0, 0, 0],
            "Input Num": [0, 0, 0],
            "Result Num": [0, 1, 2],
            "Prompt": [
                "The opposite of hot is",
                "The opposite of hot is",
                "The opposite of hot is",
            ],
            "text_result": ["cold", "cold", "cold"],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))


class LLMFunctionPostProcessTest(absltest.TestCase):
    """Test post-processing features."""

    def test_add_post_process_reorder_fn(self):
        llm_fn = LLMFunctionImpl(
            model=_MockModel(
                mock_results=["cold", "freezing", "chilly"],
            ),
            prompts=["The opposite of {word} is"],
        )

        # Reverse the order of rows.
        def reverse_fn(
            rows: Sequence[LLMFnOutputRowView],
        ) -> Sequence[int]:
            indices = list(range(0, len(rows)))
            indices.reverse()
            return indices

        results = llm_fn.add_post_process_reorder_fn(name="reverse_fn", fn=reverse_fn)(
            {"word": ["hot"]}
        )
        expected_results = {
            "Prompt Num": [0, 0, 0],
            "Input Num": [0, 0, 0],
            "Result Num": [2, 1, 0],
            "Prompt": [
                "The opposite of hot is",
                "The opposite of hot is",
                "The opposite of hot is",
            ],
            "text_result": ["chilly", "freezing", "cold"],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_add_post_process_add_fn(self):
        llm_fn = LLMFunctionImpl(
            model=_MockModel(
                mock_results=["cold", "freezing", "chilly"],
            ),
            prompts=["The opposite of {word} is"],
        )

        def add_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            return [len(row.result_value()) for row in rows]

        results = llm_fn.add_post_process_add_fn(name="length", fn=add_fn)({"word": ["hot"]})
        expected_results = {
            "Prompt Num": [0, 0, 0],
            "Input Num": [0, 0, 0],
            "Result Num": [0, 1, 2],
            "Prompt": [
                "The opposite of hot is",
                "The opposite of hot is",
                "The opposite of hot is",
            ],
            "length": [4, 8, 6],
            "text_result": ["cold", "freezing", "chilly"],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_add_post_process_replace_fn(self):
        llm_fn = LLMFunctionImpl(
            model=_MockModel(
                mock_results=["cold", "freezing", "chilly"],
            ),
            prompts=["The opposite of {word} is"],
        )

        def replace_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[str]:
            return [row.result_value().upper() for row in rows]

        results = llm_fn.add_post_process_replace_fn(name="replace_fn", fn=replace_fn)(
            {"word": ["hot"]}
        )
        expected_results = {
            "Prompt Num": [0, 0, 0],
            "Input Num": [0, 0, 0],
            "Result Num": [0, 1, 2],
            "Prompt": [
                "The opposite of hot is",
                "The opposite of hot is",
                "The opposite of hot is",
            ],
            "text_result": ["COLD", "FREEZING", "CHILLY"],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))


class LLMCompareFunctionTest(absltest.TestCase):
    """Test LLMCompareFunction."""

    def test_basic_run(self):
        """Basic comparison test."""
        lhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["lhs_{word}"],
        )
        rhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["rhs_{word}"],
        )
        compare_fn = LLMCompareFunction(
            lhs_name_and_fn=("lhs", lhs_fn), rhs_name_and_fn=("rhs", rhs_fn)
        )

        results = compare_fn(
            {
                "word": ["hello", "world"],
            }
        )
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "hello"}, {"word": "world"}],
            "lhs_text_result": ["lhs_hello", "lhs_world"],
            "rhs_text_result": ["rhs_hello", "rhs_world"],
            "is_equal": [False, False],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_run_with_post_process(self):
        """Comparison test with post-processing operations."""

        def length_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            return [len(str(row.result_value())) for row in rows]

        lhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["lhs_{word}"],
        ).add_post_process_add_fn(name="length", fn=length_fn)
        rhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["rhs_{word}"],
        ).add_post_process_add_fn(name="length", fn=length_fn)
        compare_fn = LLMCompareFunction(
            lhs_name_and_fn=("lhs", lhs_fn), rhs_name_and_fn=("rhs", rhs_fn)
        ).add_post_process_add_fn(name="length", fn=length_fn)

        results = compare_fn(
            {
                "word": ["hi", "world"],
            }
        )
        # Post-processing results from the LHS, RHS and compare functions are
        # all included in the results.
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "hi"}, {"word": "world"}],
            "lhs_length": [6, 9],
            "lhs_text_result": ["lhs_hi", "lhs_world"],
            "rhs_length": [6, 9],
            "rhs_text_result": ["rhs_hi", "rhs_world"],
            "length": [5, 5],
            "is_equal": [False, False],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_run_with_name_collisions(self):
        """Test with name collisions."""

        def length_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            return [len(str(row.result_value())) for row in rows]

        lhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["lhs_{word}"],
        ).add_post_process_add_fn(name="length", fn=length_fn)
        rhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["rhs_{word}"],
        ).add_post_process_add_fn(name="length", fn=length_fn)
        # Give left- and right functions the same names.
        compare_fn = LLMCompareFunction(
            lhs_name_and_fn=("fn", lhs_fn), rhs_name_and_fn=("fn", rhs_fn)
        ).add_post_process_add_fn(name="length", fn=length_fn)

        results = compare_fn(
            {
                "word": ["hey", "world"],
            }
        )
        # Name collisions are resolved.
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "hey"}, {"word": "world"}],
            "fn_length": [7, 9],
            "fn_text_result": ["lhs_hey", "lhs_world"],
            "fn_length_1": [7, 9],
            "fn_text_result_1": ["rhs_hey", "rhs_world"],
            "length": [5, 5],
            "is_equal": [False, False],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))

    def test_custom_compare(self):
        """Test custom comparison function."""

        def length_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            return [len(str(row.result_value())) for row in rows]

        lhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["{word}_{word}"],
        ).add_post_process_add_fn(name="length", fn=length_fn)
        rhs_fn = LLMFunctionImpl(
            model=model_lib.EchoModel(),
            prompts=["abcd_{word}"],
        ).add_post_process_add_fn(name="length", fn=length_fn)

        # We deliberately have our custom fn take Mapping[str, Any] instead
        # of LLMFnOutputRowView to make sure the typechecker allows this
        # as well.
        # Note that this function returns a non-string as well.
        def _is_length_less_than(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
            return lhs["length"] < rhs["length"]

        def _is_length_greater_than(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
            return lhs["length"] > rhs["length"]

        # Batch-based comparison function for post-processing.
        def _sum_of_lengths(
            rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
        ) -> Sequence[int]:
            return [lhs["length"] + rhs["length"] for lhs, rhs in rows]

        compare_fn = LLMCompareFunction(
            lhs_name_and_fn=("lhs", lhs_fn),
            rhs_name_and_fn=("rhs", rhs_fn),
            compare_name_and_fns=[
                ("is_shorter_than", _is_length_less_than),
                ("is_longer_than", _is_length_greater_than),
            ],
        ).add_compare_post_process_add_fn(name="sum_of_lengths", fn=_sum_of_lengths)

        results = compare_fn(
            {
                "word": ["hey", "world"],
            }
        )
        # Name collisions are resolved.
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "hey"}, {"word": "world"}],
            "lhs_length": [7, 11],
            "lhs_text_result": ["hey_hey", "world_world"],
            "rhs_length": [8, 10],
            "rhs_text_result": ["abcd_hey", "abcd_world"],
            "is_shorter_than": [True, False],
            "sum_of_lengths": [15, 21],
            "is_longer_than": [False, True],
        }
        self.assertEqual(expected_results, results.as_dict())
        self.assertEqual(list(expected_results.keys()), list(results.as_dict().keys()))


if __name__ == "__main__":
    absltest.main()
