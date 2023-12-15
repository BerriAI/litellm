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

from typing import Sequence

from absl.testing import absltest
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_post_process
from google.generativeai.notebook.lib import llmfn_post_process_cmds


LLMFnOutputRow = llmfn_output_row.LLMFnOutputRow
LLMFnOutputRowView = llmfn_output_row.LLMFnOutputRowView
PostProcessExecutionError = llmfn_post_process.PostProcessExecutionError
LLMFnPostProcessReorderCommand = llmfn_post_process_cmds.LLMFnPostProcessReorderCommand
LLMFnPostProcessAddCommand = llmfn_post_process_cmds.LLMFnPostProcessAddCommand
LLMFnPostProcessReplaceCommand = llmfn_post_process_cmds.LLMFnPostProcessReplaceCommand
LLMCompareFnPostProcessAddCommand = llmfn_post_process_cmds.LLMCompareFnPostProcessAddCommand


class LLMFnPostProcessCmdTest(absltest.TestCase):
    def test_post_process_reorder_cmd_bad_index_duplicate_indices(self):
        def bad_index_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            del rows
            return [0, 0]

        cmd = LLMFnPostProcessReorderCommand(name="test", fn=bad_index_fn)

        expected_msg = 'Error executing "test": returned indices should be unique'
        with self.assertRaisesRegex(PostProcessExecutionError, expected_msg):
            cmd.run([LLMFnOutputRow(data={"text_result": "hello"}, result_type=str)])

    def test_post_process_reorder_cmd_bad_index_less_than_zero(self):
        def bad_index_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            del rows
            return [-1]

        cmd = LLMFnPostProcessReorderCommand(name="test", fn=bad_index_fn)

        expected_msg = (
            'Error executing "test": returned indices must be greater than or equal'
            " to zero, got -1"
        )
        with self.assertRaisesRegex(PostProcessExecutionError, expected_msg):
            cmd.run([])

    def test_post_process_reorder_cmd_bad_index_greater_than_equal_to_len(self):
        def bad_index_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            del rows
            return [1]

        cmd = LLMFnPostProcessReorderCommand(name="test", fn=bad_index_fn)

        expected_msg = (
            'Error executing "test": returned indices must be less than length of'
            " rows \\(=1\\), got 1"
        )
        with self.assertRaisesRegex(PostProcessExecutionError, expected_msg):
            cmd.run([LLMFnOutputRow(data={"text_result": "hello"}, result_type=str)])

    def test_post_process_reorder_cmd_reverse(self):
        def reverse_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            del rows
            return [1, 0]

        cmd = LLMFnPostProcessReorderCommand(name="test", fn=reverse_fn)
        results = cmd.run(
            [
                LLMFnOutputRow(data={"text_result": "one"}, result_type=str),
                LLMFnOutputRow(data={"text_result": "two"}, result_type=str),
            ]
        )
        self.assertLen(results, 2)
        self.assertEqual({"text_result": "two"}, dict(results[0]))
        self.assertEqual({"text_result": "one"}, dict(results[1]))

    def test_post_process_reorder_cmd_filter(self):
        def filter_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            del rows
            return [1]

        cmd = LLMFnPostProcessReorderCommand(name="test", fn=filter_fn)
        results = cmd.run(
            [
                LLMFnOutputRow(data={"text_result": "one"}, result_type=str),
                LLMFnOutputRow(data={"text_result": "two"}, result_type=str),
            ]
        )
        self.assertLen(results, 1)
        self.assertEqual({"text_result": "two"}, dict(results[0]))

    def test_post_process_reorder_cmd_filter_to_empty(self):
        def filter_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            del rows
            return []

        cmd = LLMFnPostProcessReorderCommand(name="test", fn=filter_fn)
        results = cmd.run(
            [
                LLMFnOutputRow(data={"text_result": "one"}, result_type=str),
                LLMFnOutputRow(data={"text_result": "two"}, result_type=str),
            ]
        )
        self.assertEmpty(results)

    def test_post_process_add_cmd(self):
        def add_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[int]:
            return [len(row.result_value()) for row in rows]

        cmd = LLMFnPostProcessAddCommand(name="test", fn=add_fn)
        results = cmd.run(
            [
                LLMFnOutputRow(data={"text_result": "apple"}, result_type=str),
                LLMFnOutputRow(data={"text_result": "banana"}, result_type=str),
            ]
        )
        self.assertLen(results, 2)
        self.assertEqual({"test": 5, "text_result": "apple"}, dict(results[0]))
        self.assertEqual({"test": 6, "text_result": "banana"}, dict(results[1]))

    def test_post_process_replace_cmd(self):
        def replace_fn(rows: Sequence[LLMFnOutputRowView]) -> Sequence[str]:
            return [row.result_value().upper() for row in rows]

        cmd = LLMFnPostProcessReplaceCommand(name="test", fn=replace_fn)
        results = cmd.run(
            [
                LLMFnOutputRow(data={"text_result": "apple"}, result_type=str),
                LLMFnOutputRow(data={"text_result": "banana"}, result_type=str),
            ]
        )
        self.assertLen(results, 2)
        self.assertEqual({"text_result": "APPLE"}, dict(results[0]))
        self.assertEqual({"text_result": "BANANA"}, dict(results[1]))


class LLMCompareFnPostProcessTest(absltest.TestCase):
    def test_cmp_post_process_add_cmd(self):
        def add_fn(rows: Sequence[tuple[LLMFnOutputRowView, LLMFnOutputRowView]]) -> Sequence[int]:
            return [x.result_value() + y.result_value() for x, y in rows]

        cmd = LLMCompareFnPostProcessAddCommand(name="sum", fn=add_fn)
        results = cmd.run(
            [
                (
                    LLMFnOutputRow(data={"int_result": 1}, result_type=int),
                    LLMFnOutputRow(data={"int_result": 2}, result_type=int),
                    LLMFnOutputRow(data={"text_result": "ok"}, result_type=str),
                ),
                (
                    LLMFnOutputRow(data={"int_result": 3}, result_type=int),
                    LLMFnOutputRow(data={"int_result": 4}, result_type=int),
                    LLMFnOutputRow(data={"int_result": 5}, result_type=int),
                ),
            ]
        )
        self.assertLen(results, 2)
        self.assertEqual({"sum": 3, "text_result": "ok"}, dict(results[0]))
        self.assertEqual({"sum": 7, "int_result": 5}, dict(results[1]))


if __name__ == "__main__":
    absltest.main()
