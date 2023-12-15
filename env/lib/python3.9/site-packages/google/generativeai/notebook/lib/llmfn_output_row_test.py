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

from typing import Any, Mapping
from absl.testing import absltest
from google.generativeai.notebook.lib import llmfn_output_row


LLMFnOutputRow = llmfn_output_row.LLMFnOutputRow


class LLMFnOutputRowTest(absltest.TestCase):
    def _test_is_mapping_impl(self, row: Mapping[str, Any]) -> int:
        """Dummy function that asserts that a LLMFnOutputRow is a Mapping."""
        count = 0
        for _ in row:
            count = count + 1
        return count

    def test_is_mapping(self):
        row = LLMFnOutputRow(data={"result": "none"}, result_type=str)
        self.assertLen(row, self._test_is_mapping_impl(row))

    def _test_is_output_row_view_impl(self, view: llmfn_output_row.LLMFnOutputRowView) -> None:
        self.assertEqual("result", view.result_key())
        self.assertEqual("none", view.result_value())

    def test_is_output_row_view(self):
        row = LLMFnOutputRow(data={"result": "none"}, result_type=str)
        self._test_is_output_row_view_impl(row)

    def test_constructor(self):
        with self.assertRaisesRegex(ValueError, "Must provide non-empty data"):
            LLMFnOutputRow(data={}, result_type=str)

        with self.assertRaisesRegex(ValueError, 'Value of last entry must be of type "str"'):
            LLMFnOutputRow(data={"result": 42}, result_type=str)

        # Non-strings are accepted for non-rightmost cell.
        _ = LLMFnOutputRow(data={"int": 42, "result": "forty-two"}, result_type=str)

    def test_add(self):
        row = LLMFnOutputRow(data={"result": "none"}, result_type=str)
        row.add("score", 42)
        row.set_result_value("hello")
        self.assertEqual({"score": 42, "result": "hello"}, dict(row))
        self.assertEqual("result", row.result_key())
        self.assertEqual("hello", row.result_value())

    def test_add_with_collision(self):
        row = LLMFnOutputRow(data={"result": "none"}, result_type=str)
        row.add("score", 42)
        row.add("score", "forty-two")
        row.set_result_value("hello")
        self.assertEqual(
            {"score": 42, "score_1": "forty-two", "result": "hello"},
            dict(row.items()),
        )
        self.assertEqual("result", row.result_key())
        self.assertEqual("hello", row.result_value())

    def test_add_does_not_affect_result_cell(self):
        row = LLMFnOutputRow(data={"result": "hello"}, result_type=str)
        self.assertEqual("hello", row.result_value())
        row.add("column_one", 42)
        row.add("column_two", "forty-two")
        self.assertEqual("hello", row.result_value())
        self.assertEqual(
            {"column_one": 42, "column_two": "forty-two", "result": "hello"},
            dict(row),
        )
        self.assertEqual("result", row.result_key())
        self.assertEqual("hello", row.result_value())

    def test_set_result_value(self):
        row = LLMFnOutputRow(data={"result": "none"}, result_type=str)
        row.set_result_value("hello")
        self.assertEqual("result", row.result_key())
        self.assertEqual("hello", row.result_value())

        # Results should remain unaffected when a new column is added.
        row.add("column_one", 42)
        self.assertEqual("result", row.result_key())
        self.assertEqual("hello", row.result_value())

        self.assertEqual(
            {"column_one": 42, "result": "hello"},
            dict(row),
        )

    def test_get_item(self):
        row = LLMFnOutputRow(
            data={"one": "first", "two": "second", "three": "third"},
            result_type=str,
        )
        self.assertEqual("first", row["one"])
        self.assertEqual("second", row["two"])
        self.assertEqual("third", row["three"])

    def test_result_type(self):
        # Cannot construct the row if the result cell is of the wrong type.
        with self.assertRaisesRegex(
            ValueError, 'Value of last entry must be of type "int", got "str"'
        ):
            LLMFnOutputRow(
                data={"one": "first", "two": "second", "three": "third"},
                result_type=int,
            )

        row = LLMFnOutputRow(
            data={"one": "first", "two": "second", "three": 3},
            result_type=int,
        )
        # Cannot set the result value to the wrong type.
        with self.assertRaisesRegex(
            ValueError, 'Value of last entry must be of type "int", got "str"'
        ):
            row.set_result_value("third")

        # Can set the result value if it's the correct type.
        row.set_result_value(42)
        self.assertEqual(42, row.result_value())


if __name__ == "__main__":
    absltest.main()
