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
"""Unittest for llmfn_outputs."""
from __future__ import annotations

from typing import Sequence

from absl.testing import absltest
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import model as model_lib

LLMFnOutputEntry = llmfn_outputs.LLMFnOutputEntry
LLMFnOutputs = llmfn_outputs.LLMFnOutputs
LLMFnOutputRow = llmfn_output_row.LLMFnOutputRow


def _get_empty_model_results() -> model_lib.ModelResults:
    return model_lib.ModelResults(model_input="", text_results=[])


class LLMFnOutputsTest(absltest.TestCase):
    def _test_is_sequence(self, outputs: Sequence[LLMFnOutputEntry]):
        # Make sure `outputs` is iterable.
        count = 0
        for _ in outputs:
            count = count + 1

        # Make sure len(outputs) works.
        self.assertLen(outputs, count)

        # Make sure regular integer indices are accepted.
        self.assertIsInstance(outputs[0], LLMFnOutputEntry)

        # Make sure slices are accepted.
        self.assertLen(outputs[:-1], count - 1)
        self.assertIsInstance(outputs[:-1][0], LLMFnOutputEntry)

    def test_is_sequence(self):
        tmp_entry_list = []
        for prompt_num in range(0, 3):
            prompt = ["one", "two", "three"][prompt_num]
            output_entry = LLMFnOutputEntry(
                prompt_num=prompt_num,
                input_num=0,
                prompt=prompt,
                prompt_vars={},
                model_input=prompt,
                model_results=_get_empty_model_results(),
                output_rows=[],
            )
            tmp_entry_list.append(output_entry)

        outputs = LLMFnOutputs(tmp_entry_list)
        self._test_is_sequence(outputs)

    def test_as_dict_basic(self):
        outputs_list = []
        for prompt_num in range(0, 3):
            prompt = ["one", "two", "three"][prompt_num]
            text_results = ["red", "green", "blue"][prompt_num]
            output_entry = LLMFnOutputEntry(
                prompt_num=prompt_num,
                input_num=0,
                prompt=prompt,
                prompt_vars={},
                model_input=prompt,
                model_results=_get_empty_model_results(),
                output_rows=[
                    LLMFnOutputRow(
                        data={
                            "Result Num": 0,
                            "text_results": "{}_one".format(text_results),
                        },
                        result_type=str,
                    ),
                    LLMFnOutputRow(
                        data={
                            "Result Num": 1,
                            "text_results": "{}_two".format(text_results),
                        },
                        result_type=str,
                    ),
                ],
            )
            outputs_list.append(output_entry)

        expected_dict = {
            "Prompt Num": [0, 0, 1, 1, 2, 2],
            "Input Num": [0, 0, 0, 0, 0, 0],
            "Result Num": [0, 1, 0, 1, 0, 1],
            "Prompt": ["one", "one", "two", "two", "three", "three"],
            "text_results": [
                "red_one",
                "red_two",
                "green_one",
                "green_two",
                "blue_one",
                "blue_two",
            ],
        }

        outputs = LLMFnOutputs(outputs_list)
        self.assertEqual(expected_dict, outputs.as_dict())
        # Keys must be in the same order as well.
        self.assertEqual(list(expected_dict.keys()), list(outputs.as_dict().keys()))

    def test_as_dict_with_holes(self):
        outputs_list = []
        for prompt_num in range(0, 3):
            prompt = ["one", "two", "three"][prompt_num]
            text_results = ["red", "green", "blue"][prompt_num]
            output_entry = LLMFnOutputEntry(
                prompt_num=prompt_num,
                input_num=0,
                prompt=prompt,
                prompt_vars={},
                model_input=prompt,
                model_results=_get_empty_model_results(),
                output_rows=[
                    LLMFnOutputRow(
                        data={
                            "Result Num": 0,
                            text_results: True,
                            "text_results": "{}_one".format(text_results),
                        },
                        result_type=str,
                    ),
                    LLMFnOutputRow(
                        data={
                            "Result Num": 1,
                            text_results: True,
                            "text_results": "{}_two".format(text_results),
                        },
                        result_type=str,
                    ),
                ],
            )
            outputs_list.append(output_entry)

        expected_dict = {
            "Prompt Num": [0, 0, 1, 1, 2, 2],
            "Input Num": [0, 0, 0, 0, 0, 0],
            "Result Num": [0, 1, 0, 1, 0, 1],
            "Prompt": ["one", "one", "two", "two", "three", "three"],
            "red": [True, True, None, None, None, None],
            "green": [None, None, True, True, None, None],
            "blue": [None, None, None, None, True, True],
            "text_results": [
                "red_one",
                "red_two",
                "green_one",
                "green_two",
                "blue_one",
                "blue_two",
            ],
        }

        outputs = LLMFnOutputs(outputs_list)
        self.assertEqual(expected_dict, outputs.as_dict())
        # Keys must be in the same order as well.
        self.assertEqual(list(expected_dict.keys()), list(outputs.as_dict().keys()))


if __name__ == "__main__":
    absltest.main()
