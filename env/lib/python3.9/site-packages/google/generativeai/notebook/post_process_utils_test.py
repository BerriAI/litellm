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
"""Unittests for post_process_utils."""
from __future__ import annotations

import sys
from unittest import mock

from absl.testing import absltest
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook import (
    post_process_utils_test_helper as helper,
)
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import model as model_lib

NOT_A_FUNCTION = "this is a string not a function"
LLMFnOutputRow = llmfn_output_row.LLMFnOutputRow
LLMFnOutputRowView = llmfn_output_row.LLMFnOutputRowView
PostProcessParseError = post_process_utils.PostProcessParseError


def add_length(x: str) -> int:
    return len(x)


@post_process_utils.post_process_add_fn
def add_length_decorated(x: str) -> int:
    return len(x)


@post_process_utils.post_process_replace_fn
def to_upper(x: str) -> str:
    return x.upper()


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class PostProcessUtilsResolveTest(absltest.TestCase):
    def test_cannot_resolve_empty_expression(self):
        with self.assertRaisesRegex(PostProcessParseError, "Cannot have empty"):
            post_process_utils._resolve_one_post_processing_expression([])

    def test_cannot_resolve_multiword_expression(self):
        with self.assertRaisesRegex(PostProcessParseError, "should be a single token"):
            post_process_utils._resolve_one_post_processing_expression(["hello", "world"])

    def test_cannot_resolve_invalid_module(self):
        with self.assertRaisesRegex(PostProcessParseError, 'Unable to resolve "invalid_module"'):
            post_process_utils._resolve_one_post_processing_expression(
                ["invalid_module.add_length"]
            )

    def test_cannot_resolve_invalid_function(self):
        with self.assertRaisesRegex(
            PostProcessParseError, 'Unable to resolve "helper.invalid_function"'
        ):
            post_process_utils._resolve_one_post_processing_expression(["helper.invalid_function"])

    def test_resolve_undecorated_function(self):
        name, expr = post_process_utils._resolve_one_post_processing_expression(["add_length"])
        self.assertEqual("add_length", name)
        self.assertEqual(add_length, expr)
        self.assertEqual(11, expr("hello_world"))

    def test_resolve_decorated_add_function(self):
        name, expr = post_process_utils._resolve_one_post_processing_expression(
            ["add_length_decorated"]
        )
        self.assertEqual("add_length_decorated", name)
        self.assertEqual(add_length_decorated, expr)
        self.assertIsInstance(expr, post_process_utils._ParsedPostProcessAddExpr)
        self.assertEqual(
            [11],
            expr([LLMFnOutputRow(data={"text_result": "hello_world"}, result_type=str)]),
        )

    def test_resolve_decorated_replace_function(self):
        # Test to_upper().
        name, expr = post_process_utils._resolve_one_post_processing_expression(["to_upper"])
        self.assertEqual("to_upper", name)
        self.assertEqual(to_upper, expr)
        self.assertIsInstance(expr, post_process_utils._ParsedPostProcessReplaceExpr)
        self.assertEqual(
            ["HELLO_WORLD"],
            expr([LLMFnOutputRow(data={"text_result": "hello_world"}, result_type=str)]),
        )

    def test_resolve_module_undecorated_function(self):
        name, expr = post_process_utils._resolve_one_post_processing_expression(
            ["helper.add_length"]
        )
        self.assertEqual("helper.add_length", name)
        self.assertEqual(helper.add_length, expr)
        self.assertEqual(11, expr("hello_world"))

    def test_resolve_module_decorated_add_function(self):
        name, expr = post_process_utils._resolve_one_post_processing_expression(
            ["helper.add_length_decorated"]
        )
        self.assertEqual("helper.add_length_decorated", name)
        self.assertEqual(helper.add_length_decorated, expr)
        self.assertIsInstance(expr, post_process_utils._ParsedPostProcessAddExpr)
        self.assertEqual(
            [11],
            expr([LLMFnOutputRow(data={"text_result": "hello_world"}, result_type=str)]),
        )

    def test_resolve_module_decorated_replace_function(self):
        name, expr = post_process_utils._resolve_one_post_processing_expression(["helper.to_upper"])
        self.assertEqual("helper.to_upper", name)
        self.assertEqual(helper.to_upper, expr)
        self.assertIsInstance(expr, post_process_utils._ParsedPostProcessReplaceExpr)
        self.assertEqual(
            ["HELLO_WORLD"],
            expr([LLMFnOutputRow(data={"text_result": "hello_world"}, result_type=str)]),
        )


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class PostProcessUtilsTest(absltest.TestCase):
    def test_must_be_callable(self):
        with self.assertRaisesRegex(PostProcessParseError, "NOT_A_FUNCTION is not callable"):
            post_process_utils.resolve_post_processing_tokens([["NOT_A_FUNCTION"]])

    def test_parsed_post_process_add_fn(self):
        """Test that from a post-processing token to an updated LLMFunction."""
        parsed_exprs = post_process_utils.resolve_post_processing_tokens(
            [
                ["add_length"],
            ]
        )
        self.assertLen(parsed_exprs, 1)
        self.assertIsInstance(parsed_exprs[0], post_process_utils._ParsedPostProcessAddExpr)
        llm_fn = llm_function.LLMFunctionImpl(model=model_lib.EchoModel(), prompts=["hello"])
        parsed_exprs[0].add_to_llm_function(llm_fn)
        results = llm_fn()
        self.assertEqual(
            {
                "Input Num": [0],
                "Prompt Num": [0],
                "Prompt": ["hello"],
                "Result Num": [0],
                "add_length": [5],
                "text_result": ["hello"],
            },
            results.as_dict(),
        )

    def test_parsed_post_process_replace_fn(self):
        parsed_exprs = post_process_utils.resolve_post_processing_tokens(
            [
                ["to_upper"],
            ]
        )
        self.assertLen(parsed_exprs, 1)
        self.assertIsInstance(parsed_exprs[0], post_process_utils._ParsedPostProcessReplaceExpr)
        llm_fn = llm_function.LLMFunctionImpl(model=model_lib.EchoModel(), prompts=["hello"])
        parsed_exprs[0].add_to_llm_function(llm_fn)
        results = llm_fn()
        self.assertEqual(
            {
                "Input Num": [0],
                "Prompt Num": [0],
                "Prompt": ["hello"],
                "Result Num": [0],
                "text_result": ["HELLO"],
            },
            results.as_dict(),
        )

    def test_resolve_post_processing_tokens(self):
        parsed_exprs = post_process_utils.resolve_post_processing_tokens(
            [
                ["add_length"],
                ["to_upper"],
                ["add_length_decorated"],
                ["helper.add_length"],
                ["helper.add_length_decorated"],
                ["helper.to_upper"],
            ]
        )

        for fn in parsed_exprs:
            self.assertIsInstance(fn, post_process_utils.ParsedPostProcessExpr)

        llm_fn = llm_function.LLMFunctionImpl(model=model_lib.EchoModel(), prompts=["hello"])
        for expr in parsed_exprs:
            expr.add_to_llm_function(llm_fn)

        results = llm_fn()
        self.assertEqual(
            {
                "Input Num": [0],
                "Prompt Num": [0],
                "Prompt": ["hello"],
                "Result Num": [0],
                "add_length": [5],
                "add_length_decorated": [5],
                "add_length_decorated_1": [5],
                "helper.add_length": [5],
                "text_result": ["HELLO"],
            },
            results.as_dict(),
        )


if __name__ == "__main__":
    absltest.main()
