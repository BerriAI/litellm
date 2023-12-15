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
"""Unittests for cmd_line_parser."""
from __future__ import annotations

import sys
from unittest import mock

from absl.testing import absltest
from google.generativeai.notebook import argument_parser
from google.generativeai.notebook import cmd_line_parser
from google.generativeai.notebook import model_registry
from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import model as model_lib


_INPUT_VAR_ONE = {"word": ["one"]}
_INPUT_VAR_TWO = {"word": ["two"]}
_INPUT_VAR_THREE = {"word": ["three"]}
_NOT_WORD_INPUT_VAR = {"not_word": ["hello", "world"]}
_OUTPUT_VAR_ONE: llmfn_outputs.LLMFnOutputs | None = None
_OUTPUT_VAR_TWO: llmfn_outputs.LLMFnOutputs | None = None
_GROUND_TRUTH_VAR = ["apple", "banana", "cantaloupe"]


def _set_output_sink(text_result: str, sink: llmfn_outputs.LLMFnOutputsSink) -> None:
    sink.write_outputs(
        llmfn_outputs.LLMFnOutputs(
            outputs=[
                llmfn_outputs.LLMFnOutputEntry(
                    prompt_num=0,
                    input_num=0,
                    prompt_vars={},
                    output_rows=[
                        llmfn_output_row.LLMFnOutputRow(
                            data={
                                llmfn_outputs.ColumnNames.RESULT_NUM: 0,
                                llmfn_outputs.ColumnNames.TEXT_RESULT: (text_result),
                            },
                            result_type=str,
                        )
                    ],
                ),
            ]
        )
    )


class CmdLineParserTestBase(absltest.TestCase):
    def setUp(self):
        super().setUp()

        # Reset variables.
        global _OUTPUT_VAR_ONE
        global _OUTPUT_VAR_TWO
        _OUTPUT_VAR_ONE = None
        _OUTPUT_VAR_TWO = None


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserCommonTest(CmdLineParserTestBase):
    """For tests that are not specific to any command."""

    def test_parse_args_help(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaises(argument_parser.ParserNormalExit):
            parser.parse_line("-h")
        with self.assertRaises(argument_parser.ParserNormalExit):
            parser.parse_line("--help")
        with self.assertRaises(argument_parser.ParserNormalExit):
            parser.parse_line("run -h")
        with self.assertRaises(argument_parser.ParserNormalExit):
            parser.parse_line("run --help")

    def test_parse_args_empty(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("")
        self.assertEqual(parsed_args_lib.CommandName.RUN_CMD, results.cmd)
        self.assertEqual(model_registry.ModelName.TEXT_MODEL, results.model_type)
        self.assertEqual([], results.inputs)
        self.assertEqual(
            model_lib.ModelArguments(),
            results.model_args,
        )

    def test_parse_args_no_reuse(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("--inputs _INPUT_VAR_ONE")
        self.assertLen(results.inputs, 1)
        self.assertEqual([{"word": "one"}], results.inputs[0].to_normalized_inputs())

        # Calling parse_line() again should return brand new results.
        results, _ = parser.parse_line("--inputs _INPUT_VAR_TWO _INPUT_VAR_THREE")
        self.assertLen(results.inputs, 2)
        self.assertEqual([{"word": "two"}], results.inputs[0].to_normalized_inputs())
        self.assertEqual([{"word": "three"}], results.inputs[1].to_normalized_inputs())


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserModelFlagsTest(CmdLineParserTestBase):
    def test_parse_args_sets_model_type(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("--model_type=echo")
        self.assertEqual(model_registry.ModelName.ECHO_MODEL, results.model_type)

        results, _ = parser.parse_line("--model_type=text")
        self.assertEqual(model_registry.ModelName.TEXT_MODEL, results.model_type)

    def test_parse_args_sets_model(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("--model=/ml/test")
        self.assertEqual(model_lib.ModelArguments(model="/ml/test"), results.model_args)

    def test_parse_args_sets_temperature(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("--temperature=0")
        self.assertEqual(model_lib.ModelArguments(temperature=0), results.model_args)

        results, _ = parser.parse_line("--temperature=0.5")
        self.assertEqual(model_lib.ModelArguments(temperature=0.5), results.model_args)

        with self.assertRaisesRegex(
            argument_parser.ParserError,
            (
                'Error with value "-1.0", got ValueError: Value should be greater'
                " than or equal to zero, got -1.0"
            ),
        ):
            parser.parse_line("--temperature=-1")

    def test_parse_args_sets_candidate_count(self):
        parser = cmd_line_parser.CmdLineParser()
        # Test that the min and max values are accepted.
        results, _ = parser.parse_line("--candidate_count=1")
        self.assertEqual(model_lib.ModelArguments(candidate_count=1), results.model_args)
        results, _ = parser.parse_line("--candidate_count=8")
        self.assertEqual(model_lib.ModelArguments(candidate_count=8), results.model_args)

        # Test that values outside the min and max are rejected.
        with self.assertRaisesRegex(
            argument_parser.ParserError,
            (
                r'Error with value "0", got ValueError: Value should be in the'
                r" range \[1, 8\], got 0"
            ),
        ):
            parser.parse_line("--candidate_count=0")

        with self.assertRaisesRegex(
            argument_parser.ParserError,
            (
                r'Error with value "9", got ValueError: Value should be in the'
                r" range \[1, 8\], got 9"
            ),
        ):
            parser.parse_line("--candidate_count=9")

    def test_parse_args_sets_unique(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("")
        self.assertFalse(results.unique)

        results, _ = parser.parse_line("--unique")
        self.assertTrue(results.unique)


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserRunTest(CmdLineParserTestBase):
    """For the "run" command."""

    def test_parse_args_run_is_default(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("--model_type=echo")
        self.assertEqual(parsed_args_lib.CommandName.RUN_CMD, results.cmd)
        self.assertEqual(model_registry.ModelName.ECHO_MODEL, results.model_type)

    def test_parse_input_and_output_args(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line(
            "run --model_type=echo --inputs _INPUT_VAR_ONE _INPUT_VAR_TWO --outputs"
            " _OUTPUT_VAR_ONE _OUTPUT_VAR_TWO _UNDECLARED_OUTPUT_VAR"
        )
        self.assertEqual(parsed_args_lib.CommandName.RUN_CMD, results.cmd)
        self.assertEqual(model_registry.ModelName.ECHO_MODEL, results.model_type)

        self.assertLen(results.inputs, 2)
        self.assertEqual([{"word": "one"}], results.inputs[0].to_normalized_inputs())
        self.assertEqual([{"word": "two"}], results.inputs[1].to_normalized_inputs())

        self.assertLen(results.outputs, 3)

        # Check that the output is going to the correct variable by writing a value
        # to the sink then reading it back.
        _set_output_sink(text_result="one", sink=results.outputs[0])
        self.assertIsInstance(_OUTPUT_VAR_ONE, llmfn_outputs.LLMFnOutputs)
        self.assertEqual(
            "one",
            _OUTPUT_VAR_ONE[0].output_rows[0][llmfn_outputs.ColumnNames.TEXT_RESULT],
        )
        _set_output_sink(text_result="two", sink=results.outputs[1])
        self.assertIsInstance(_OUTPUT_VAR_TWO, llmfn_outputs.LLMFnOutputs)
        self.assertEqual(
            "two",
            _OUTPUT_VAR_TWO[0].output_rows[0][llmfn_outputs.ColumnNames.TEXT_RESULT],
        )
        _set_output_sink(text_result="undeclared", sink=results.outputs[2])
        # pylint: disable-next=undefined-variable
        undeclared_var = _UNDECLARED_OUTPUT_VAR  # type: ignore
        self.assertIsInstance(undeclared_var, llmfn_outputs.LLMFnOutputs)
        self.assertEqual(
            "undeclared",
            undeclared_var[0].output_rows[0][llmfn_outputs.ColumnNames.TEXT_RESULT],
        )

    def test_placeholder_error(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(
            argument_parser.ParserError,
            (
                'argument --inputs/-i: Error with value "_NOT_WORD_INPUT_VAR", got'
                ' ValueError: Placeholder "word" not found in input'
            ),
        ):
            parser.parse_line(
                "run --inputs _NOT_WORD_INPUT_VAR",
                placeholders=frozenset({"word"}),
            )


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserCompileTest(CmdLineParserTestBase):
    """For the "compile" command."""

    def test_parse_args_needs_save_name(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(
            argument_parser.ParserError,
            "the following arguments are required: compile_save_name",
        ):
            parser.parse_line("compile")

    def test_parse_args_bad_save_name(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(argument_parser.ParserError, "Invalid Python variable name"):
            parser.parse_line("compile 1234")

    def test_parse_args_has_save_name(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("compile my_fn")
        self.assertEqual("my_fn", results.compile_save_name)


_test_lhs_fn = llm_function.LLMFunctionImpl(
    model=model_lib.EchoModel(), prompts=["dummy lhs prompt {word}"]
)

_test_rhs_fn = llm_function.LLMFunctionImpl(
    model=model_lib.EchoModel(), prompts=["dummy rhs prompt {word}"]
)


def _test_compare_fn(lhs: str, rhs: str) -> bool:
    return lhs == rhs


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserCompareTest(CmdLineParserTestBase):
    """For the "compare" command."""

    def test_compare(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line("compare _test_lhs_fn _test_rhs_fn --inputs _INPUT_VAR_ONE")
        self.assertEqual(("_test_lhs_fn", _test_lhs_fn), results.lhs_name_and_fn)
        self.assertEqual(("_test_rhs_fn", _test_rhs_fn), results.rhs_name_and_fn)
        self.assertEmpty(results.compare_fn)

    def test_compare_with_custom_compare_fn(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line(
            "compare _test_lhs_fn _test_rhs_fn --inputs _INPUT_VAR_ONE --compare_fn"
            " _test_compare_fn"
        )
        self.assertEqual(("_test_lhs_fn", _test_lhs_fn), results.lhs_name_and_fn)
        self.assertEqual(("_test_rhs_fn", _test_rhs_fn), results.rhs_name_and_fn)
        self.assertEqual([("_test_compare_fn", _test_compare_fn)], results.compare_fn)

    def test_placeholder_error(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(
            argument_parser.ParserError,
            (
                'argument --inputs/-i: Error with value "_NOT_WORD_INPUT_VAR", got'
                ' ValueError: Placeholder "word" not found in input'
            ),
        ):
            parser.parse_line("compare _test_lhs_fn _test_rhs_fn --inputs _NOT_WORD_INPUT_VAR")


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserEvalTest(CmdLineParserTestBase):
    """For the "eval" command."""

    def test_eval(self):
        parser = cmd_line_parser.CmdLineParser()
        results, _ = parser.parse_line(
            "eval --ground_truth _GROUND_TRUTH_VAR --inputs _INPUT_VAR_ONE"
        )
        self.assertEqual(["apple", "banana", "cantaloupe"], results.ground_truth)

    def test_placeholder_error(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(
            argument_parser.ParserError,
            (
                'argument --inputs/-i: Error with value "_NOT_WORD_INPUT_VAR", got'
                ' ValueError: Placeholder "word" not found in input'
            ),
        ):
            parser.parse_line(
                "eval --ground_truth _GROUND_TRUTH_VAR --inputs _NOT_WORD_INPUT_VAR",
                placeholders=frozenset({"word"}),
            )


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CmdLineParserPostProcessingTest(CmdLineParserTestBase):
    """For the "run" command."""

    def test_parse_tokens(self):
        parser = cmd_line_parser.CmdLineParser()
        _, post_process_exprs = parser.parse_line("| add_length | to_upper")
        self.assertLen(post_process_exprs, 2)
        self.assertEqual(["add_length"], post_process_exprs[0])
        self.assertEqual(["to_upper"], post_process_exprs[1])

    def test_illformed_expression(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(
            post_process_utils.PostProcessParseError,
            "Cannot have empty post-processing expression",
        ):
            parser.parse_line("| | to_upper")

        with self.assertRaisesRegex(
            post_process_utils.PostProcessParseError,
            "Cannot have empty post-processing expression",
        ):
            parser.parse_line("| ")

        with self.assertRaisesRegex(
            post_process_utils.PostProcessParseError,
            "Cannot have empty post-processing expression",
        ):
            parser.parse_line("| add_length |")

    def test_cannot_parse_multiple_tokens_in_one_expression(self):
        parser = cmd_line_parser.CmdLineParser()
        with self.assertRaisesRegex(
            post_process_utils.PostProcessParseError,
            "Post-processing expression should be a single token",
        ):
            parser.parse_line("| add_length to_upper")


if __name__ == "__main__":
    absltest.main()
