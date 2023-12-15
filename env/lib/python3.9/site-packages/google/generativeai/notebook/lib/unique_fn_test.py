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
from __future__ import annotations

from absl.testing import absltest
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import unique_fn


LLMFnOutputRow = llmfn_output_row.LLMFnOutputRow


class UniqueFntest(absltest.TestCase):
    def test_all_unique(self):
        rows = [
            LLMFnOutputRow(data={"text_result": "red"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "green"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "blue"}, result_type=str),
        ]
        self.assertEqual([0, 1, 2], unique_fn.unique_fn(rows))

    def test_some_dupes(self):
        rows = [
            LLMFnOutputRow(data={"text_result": "red"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "red"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "green"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "red"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "green"}, result_type=str),
            LLMFnOutputRow(data={"text_result": "blue"}, result_type=str),
        ]
        self.assertEqual([0, 2, 5], unique_fn.unique_fn(rows))


if __name__ == "__main__":
    absltest.main()
