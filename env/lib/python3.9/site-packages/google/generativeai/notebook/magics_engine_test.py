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

import sys
from typing import Any, Callable, Mapping, Sequence
from unittest import mock

from absl import logging
from absl.testing import absltest
from google.generativeai.notebook import gspread_client
from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import magics_engine
from google.generativeai.notebook import model_registry
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook import sheets_id
from google.generativeai.notebook import sheets_utils
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_inputs_source
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import model
import pandas


def _fake_llm_function(x: Any) -> Any:
    assert False, "Should not be called"
    return x


# Used to store output of "compile" tests.
_compiled_function = _fake_llm_function
_compiled_lhs_function = _fake_llm_function
_compiled_rhs_function = _fake_llm_function


# Decorators for testing post-processing operations.
def add_length(result: str) -> int:
    return len(result)


@post_process_utils.post_process_add_fn
def add_length_decorated(result: str) -> int:
    return len(result)


@post_process_utils.post_process_replace_fn
def repeat(result: str) -> str:
    return result + result


@post_process_utils.post_process_replace_fn
def to_upper(result: str) -> str:
    return result.upper()


# Comparison functions for "compare" command.
def get_sum_of_lengths(lhs: str, rhs: str) -> int:
    return len(lhs) + len(rhs)


def concat(lhs: str, rhs: str) -> str:
    return lhs + " " + rhs


def my_is_equal_fn(lhs: str, rhs: str) -> bool:
    return lhs == rhs


class EchoModelRegistry(model_registry.ModelRegistry):
    """Fake model registry for testing."""

    def __init__(self, alt_model=None):
        self.model = alt_model or model.EchoModel()
        self.get_model_name: model_registry.ModelName | None = None

    def get_model(self, model_name: model_registry.ModelName) -> model.AbstractModel:
        self.get_model_name = model_name
        return self.model


class FakeIPythonEnv(ipython_env.IPythonEnv):
    """Fake IPythonEnv for testing."""

    def __init__(self):
        self.display_args: Any = None

    def clear(self) -> None:
        self.display_args = None

    def display(self, x: Any) -> None:
        self.display_args = x
        logging.info("IPythonEnv.display called with:\n%r", x)

    def display_html(self, x: Any) -> None:
        logging.info("IPythonEnv.display_html called with:\n%r", x)


class FakeInputsSource(llmfn_inputs_source.LLMFnInputsSource):
    def _to_normalized_inputs_impl(
        self,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        return [{"word": "quack3"}, {"word": "quack4"}], lambda: None


class FakeOutputsSink(llmfn_outputs.LLMFnOutputsSink):
    def __init__(self):
        self.outputs: llmfn_outputs.LLMFnOutputsBase | None = None

    def write_outputs(self, outputs: llmfn_outputs.LLMFnOutputsBase) -> None:
        self.outputs = outputs


class MockGSpreadClient(gspread_client.GSpreadClient):
    def __init__(self):
        self.get_all_records_name: str | None = None
        self.get_all_records_worksheet_id: int | None = None

        self.write_records_name: str | None = None
        self.write_records_rows: Sequence[Sequence[Any]] | None = None

    def validate(self, sid: sheets_id.SheetsIdentifier):
        if sid.name() is None:
            raise gspread_client.SpreadsheetNotFoundError("Sheets not found: {}".format(sid))
        pass

    def get_all_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        worksheet_id: int,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        self.get_all_records_name = sid.name()
        self.get_all_records_worksheet_id = worksheet_id
        return [{"word": "quack5"}, {"word": "quack6"}], lambda: None

    def write_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        rows: Sequence[Sequence[Any]],
    ) -> None:
        self.write_records_name = sid.name()
        self.write_records_rows = rows


# Fake input variables to test the --inputs flag.
_INPUT_VAR_ONE = {"word": ["quack1", "quack2"]}
_INPUT_VAR_TWO = FakeInputsSource()
_SHEETS_INPUT_VAR = None

# Variable to test the --ground_truth flag.
_GROUND_TRUTH_VAR = ["QUACK QUACK1", "NOT QUACK QUACK2"]

# Variables to test the --outputs flag.
_output_var: llmfn_outputs.LLMFnOutputs | None = None
_output_sink_var = FakeOutputsSink()


def _reset_globals() -> None:
    # Reset all our gloabls.
    global _compiled_function
    global _compiled_lhs_function
    global _compiled_rhs_function
    global _output_var
    global _output_sink_var
    global _SHEETS_INPUT_VAR

    _compiled_function = None
    _compiled_lhs_function = None
    _compiled_rhs_function = None
    _output_var = None
    _output_sink_var = FakeOutputsSink()

    # This should be done after the MockGSpreadClient has been set up.
    _SHEETS_INPUT_VAR = sheets_utils.SheetsInputs(
        sid=sheets_id.SheetsIdentifier(name="fake_sheets"),
        worksheet_id=42,
    )


class EndToEndTests(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self._mock_client = MockGSpreadClient()
        gspread_client.testonly_set_client(self._mock_client)
        _reset_globals()

    def _assert_is_expected_pandas_dataframe(
        self, results: pandas.DataFrame, expected_results: Mapping[str, Any]
    ) -> None:
        self.assertIsInstance(results, pandas.DataFrame)
        self.assertEqual(expected_results, results.to_dict(orient="list"))
        self.assertEqual(
            list(expected_results.keys()),
            list(results.to_dict(orient="list").keys()),
        )

    def _assert_output_var_is_expected_results(
        self,
        var: Any,
        expected_results: Mapping[str, Any],
        fake_env: FakeIPythonEnv,
    ) -> None:
        self.assertIsInstance(var, llmfn_outputs.LLMFnOutputs)

        # Make sure output vars are also populated.
        self.assertEqual(expected_results, var.as_dict())
        self.assertEqual(
            list(expected_results.keys()),
            list(var.as_dict().keys()),
        )

        # Make sure the output object is displayable in notebooks.
        self.assertTrue(hasattr(var, "_ipython_display_"))
        fake_env.clear()
        # The typechecker thinks LLMFnOutputs does not have _ipython_display_
        # because the method is conditionally added.
        var._ipython_display_()  # type: ignore
        self.assertIsInstance(fake_env.display_args, pandas.DataFrame)
        self.assertEqual(expected_results, fake_env.display_args.to_dict(orient="list"))


@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class HelpEndToEndTests(EndToEndTests):
    def test_help(self):
        mock_registry = EchoModelRegistry()
        magic_line = "--help"
        engine = magics_engine.MagicsEngine(registry=mock_registry)

        # Should not raise an exception.
        results = engine.execute_cell(magic_line, "ignored")
        self.assertRegex(str(results), "A system for interacting with LLMs")

    def test_run_help(self):
        mock_registry = EchoModelRegistry()
        magic_line = "run --help"
        engine = magics_engine.MagicsEngine(registry=mock_registry)

        # Should not raise an exception.
        results = engine.execute_cell(magic_line, "ignored")
        self.assertRegex(str(results), "usage: palm run")

    def test_error(self):
        mock_registry = EchoModelRegistry()
        magic_line = "run --this_is_an_invalid_flag"
        engine = magics_engine.MagicsEngine(registry=mock_registry)

        with self.assertRaisesRegex(
            SystemExit, "unrecognized arguments: --this_is_an_invalid_flag"
        ):
            engine.execute_cell(magic_line, "ignored")


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class RunCmdEndToEndTests(EndToEndTests):
    def test_run_cmd(self):
        """Smoke test for executing the run command."""
        mock_registry = EchoModelRegistry()
        magic_line = "run --model_type=echo"
        engine = magics_engine.MagicsEngine(registry=mock_registry)

        results = engine.execute_cell(magic_line, "quack quack")
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["quack quack"],
            "text_result": ["quack quack"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        results = engine.execute_cell(magic_line, "line 1\nline 2\n")
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["line 1\nline 2"],
            "text_result": ["line 1\nline 2"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        # Confirm trailing line breaks preserved (bar 1)
        results = engine.execute_cell(magic_line, "line 1\nline 2\n\n")
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["line 1\nline 2\n"],
            "text_result": ["line 1\nline 2\n"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        # --model_type should be parsed and passed to the ModelRegistry instance.
        self.assertEqual(model_registry.ModelName.ECHO_MODEL, mock_registry.get_model_name)

    def test_model_args_passed(self):
        mock_model = mock.create_autospec(model.EchoModel)
        reg = EchoModelRegistry(mock_model)

        engine = magics_engine.MagicsEngine(registry=reg)
        _ = engine.execute_cell(
            (
                "run --model_type=echo --model=the_best_model --temperature=0.25"
                " --candidate_count=3"
            ),
            "quack",
        )

        expected_model_args = model.ModelArguments(
            model="the_best_model", temperature=0.25, candidate_count=3
        )
        actual_model_args = mock_model.call_model.call_args.kwargs["model_args"]
        self.assertEqual(actual_model_args, expected_model_args)

    def test_candidate_count(self):
        mock_registry = EchoModelRegistry()
        engine = magics_engine.MagicsEngine(registry=mock_registry)
        results = engine.execute_cell(
            "run --model_type=echo --candidate=3",
            "quack",
        )
        expected_results = {
            "Prompt Num": [0, 0, 0],
            "Input Num": [0, 0, 0],
            "Result Num": [0, 1, 2],
            "Prompt": ["quack", "quack", "quack"],
            "text_result": ["quack", "quack", "quack"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

    def test_unique(self):
        mock_registry = EchoModelRegistry()
        engine = magics_engine.MagicsEngine(registry=mock_registry)
        results = engine.execute_cell(
            "run --model_type=echo --candidate=3 --unique",
            "quack",
        )
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["quack"],
            "text_result": ["quack"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

    def test_inputs_passed(self):
        magic_line = (
            "run --model_type=echo --inputs _INPUT_VAR_ONE _INPUT_VAR_TWO" " _SHEETS_INPUT_VAR"
        )
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        results = engine.execute_cell(magic_line, "quack {word}")
        self.assertIsInstance(results, pandas.DataFrame)
        expected_results = {
            "Prompt Num": [0, 0, 0, 0, 0, 0],
            "Input Num": [0, 1, 2, 3, 4, 5],
            "Result Num": [0, 0, 0, 0, 0, 0],
            "Prompt": [
                "quack quack1",
                "quack quack2",
                "quack quack3",
                "quack quack4",
                "quack quack5",
                "quack quack6",
            ],
            "text_result": [
                "quack quack1",
                "quack quack2",
                "quack quack3",
                "quack quack4",
                "quack quack5",
                "quack quack6",
            ],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self.assertEqual("fake_sheets", self._mock_client.get_all_records_name)
        self.assertEqual(42, self._mock_client.get_all_records_worksheet_id)

    def test_sheets_input_names_passed(self):
        """Test using --sheets_input_names."""

        magic_line = "run --model_type=echo --sheets_input_names my_fake_sheets"
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        results = engine.execute_cell(magic_line, "quack {word}")
        self.assertIsInstance(results, pandas.DataFrame)
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt": [
                "quack quack5",
                "quack quack6",
            ],
            "text_result": [
                "quack quack5",
                "quack quack6",
            ],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self.assertEqual("my_fake_sheets", self._mock_client.get_all_records_name)
        # The default worksheet_id should be used.
        self.assertEqual(0, self._mock_client.get_all_records_worksheet_id)

    def test_validate_inputs_against_placeholders(self):
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        with self.assertRaisesRegex(
            SystemExit,
            (
                'argument --inputs/-i: Error with value "_INPUT_VAR_ONE", got'
                ' ValueError: Placeholder "not_word" not found in input'
            ),
        ):
            engine.execute_cell(
                "run --model_type=echo --inputs _INPUT_VAR_ONE",
                "quack {not_word}",
            )

        with self.assertRaisesRegex(
            SystemExit,
            (
                'argument --inputs/-i: Error with value "_INPUT_VAR_TWO", got'
                ' ValueError: Placeholder "not_word" not found in input'
            ),
        ):
            engine.execute_cell(
                "run --model_type=echo --inputs _INPUT_VAR_TWO",
                "quack {not_word}",
            )

        with self.assertRaisesRegex(
            SystemExit,
            (
                'argument --inputs/-i: Error with value "_SHEETS_INPUT_VAR", got'
                ' ValueError: Placeholder "not_word" not found in input'
            ),
        ):
            engine.execute_cell(
                "run --model_type=echo --inputs _SHEETS_INPUT_VAR",
                "quack {not_word}",
            )

    def test_validate_sheets_inputs_against_placeholders(self):
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        with self.assertRaisesRegex(
            SystemExit,
            (
                "argument --sheets_input_names/-si: Error with value"
                ' "my_fake_sheets", got ValueError: Placeholder "not_word" not'
                " found in input"
            ),
        ):
            engine.execute_cell(
                "run --model_type=echo --sheets_input_names my_fake_sheets",
                "quack {not_word}",
            )

    def test_post_process(self):
        magic_line = "run --model_type=echo | add_length | repeat | add_length_decorated"
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        results = engine.execute_cell(magic_line, "quack")
        self.assertIsInstance(results, pandas.DataFrame)
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["quack"],
            "add_length": [5],
            "add_length_decorated": [10],
            "text_result": ["quackquack"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

    def test_outputs(self):
        # Include post-processing commands to make sure their results are exported
        # as well.
        magic_line = "run --model_type=echo --outputs _output_var | add_length | repeat"

        fake_env = FakeIPythonEnv()
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry(), env=fake_env)
        results = engine.execute_cell(magic_line, "quack")
        self.assertIsInstance(_output_var, llmfn_outputs.LLMFnOutputs)
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["quack"],
            "add_length": [5],
            "text_result": ["quackquack"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self._assert_output_var_is_expected_results(
            var=_output_var,
            expected_results=expected_results,
            fake_env=fake_env,
        )

    def test_outputs_sink(self):
        # Include post-processing commands to make sure their results are exported
        # as well.
        magic_line = "run --model_type=echo --outputs _output_sink_var | add_length | repeat"
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        results = engine.execute_cell(magic_line, "quack")
        self.assertIsNotNone(_output_sink_var.outputs)
        self.assertIsInstance(_output_sink_var.outputs, llmfn_outputs.LLMFnOutputs)
        expected_results = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["quack"],
            "add_length": [5],
            "text_result": ["quackquack"],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self.assertEqual(
            expected_results,
            _output_sink_var.outputs.as_pandas_dataframe().to_dict(orient="list"),
        )

    def test_sheets_outputs_names(self):
        # Include post-processing commands to make sure their results are exported
        # as well.
        magic_line = (
            "run --model_type=echo --sheets_output_names my_fake_output_sheets  |"
            " add_length | repeat"
        )
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry())
        _ = engine.execute_cell(magic_line, "quack")

        self.assertEqual("my_fake_output_sheets", self._mock_client.write_records_name)
        expected_rows = [
            [
                "Prompt Num",
                "Input Num",
                "Result Num",
                "Prompt",
                "add_length",
                "text_result",
            ],
            [0, 0, 0, "quack", 5, "quackquack"],
        ]
        self.assertEqual(expected_rows, self._mock_client.write_records_rows)


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CompileCmdEndToEndTests(EndToEndTests):
    def test_compile_cmd(self):
        fake_env = FakeIPythonEnv()
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry(), env=fake_env)

        _ = engine.execute_cell("compile _compiled_function --model_type=echo", "quack {word}")

        # The "compile" command produces a saved function.
        # Execute the saved function and check that it produces the expected output.
        self.assertIsInstance(_compiled_function, llm_function.LLMFunction)
        outputs = _compiled_function({"word": ["LOUD QUACK"]})
        expected_outputs = {
            "Prompt Num": [0],
            "Input Num": [0],
            "Result Num": [0],
            "Prompt": ["quack LOUD QUACK"],
            "text_result": ["quack LOUD QUACK"],
        }
        self._assert_output_var_is_expected_results(
            var=outputs, expected_results=expected_outputs, fake_env=fake_env
        )


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class CompareCmdEndToEndTests(EndToEndTests):
    def test_compare_cmd_with_default_compare_fn(self):
        fake_env = FakeIPythonEnv()
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry(), env=fake_env)

        # Create a pair of LLMFunctions to compare.
        _ = engine.execute_cell(
            "compile _compiled_lhs_function --model_type=echo",
            "left quack {word}",
        )
        _ = engine.execute_cell(
            "compile _compiled_rhs_function --model_type=echo",
            "right quack {word}",
        )

        # Run comparison.
        results = engine.execute_cell(
            (
                "compare _compiled_lhs_function _compiled_rhs_function --inputs"
                " _INPUT_VAR_ONE --outputs _output_var"
            ),
            "ignored",
        )

        # Check results.
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "quack1"}, {"word": "quack2"}],
            "_compiled_lhs_function_text_result": [
                "left quack quack1",
                "left quack quack2",
            ],
            "_compiled_rhs_function_text_result": [
                "right quack quack1",
                "right quack quack2",
            ],
            "is_equal": [False, False],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self._assert_output_var_is_expected_results(
            var=_output_var,
            expected_results=expected_results,
            fake_env=fake_env,
        )

    def test_compare_cmd_with_custom_compare_fn(self):
        fake_env = FakeIPythonEnv()
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry(), env=fake_env)

        # Create a pair of LLMFunctions to compare.
        _ = engine.execute_cell(
            "compile _compiled_lhs_function --model_type=echo",
            "left quack {word}",
        )
        _ = engine.execute_cell(
            "compile _compiled_rhs_function --model_type=echo",
            "right quack {word}",
        )

        # Run comparison.
        results = engine.execute_cell(
            (
                "compare _compiled_lhs_function _compiled_rhs_function --inputs"
                " _INPUT_VAR_ONE --outputs _output_var --compare_fn concat"
                " get_sum_of_lengths"
            ),
            "ignored",
        )

        # Check results.
        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "quack1"}, {"word": "quack2"}],
            "_compiled_lhs_function_text_result": [
                "left quack quack1",
                "left quack quack2",
            ],
            "_compiled_rhs_function_text_result": [
                "right quack quack1",
                "right quack quack2",
            ],
            "concat": [
                "left quack quack1 right quack quack1",
                "left quack quack2 right quack quack2",
            ],
            "get_sum_of_lengths": [35, 35],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self._assert_output_var_is_expected_results(
            var=_output_var,
            expected_results=expected_results,
            fake_env=fake_env,
        )


# `unittest discover` does not run via __main__, so patch this context in.
@mock.patch.dict(sys.modules, {"__main__": sys.modules[__name__]})
class EvalCmdEndToEndTests(EndToEndTests):
    def test_eval_cmd(self):
        fake_env = FakeIPythonEnv()
        engine = magics_engine.MagicsEngine(registry=EchoModelRegistry(), env=fake_env)

        # Run evaluation.
        #
        # Some of the features tested are:
        # 1. We evoke a model flag (to make sure model flags are parsed.)
        # 2. We add a post-processing function to the prompt results
        # 3. We add a few custom comparison functions.
        # 4. We write to an output variable.
        results = engine.execute_cell(
            (
                "eval --model_type=echo --ground_truth _GROUND_TRUTH_VAR --inputs"
                " _INPUT_VAR_ONE --outputs _output_var --compare_fn"
                " get_sum_of_lengths my_is_equal_fn | to_upper"
            ),
            "quack {word}",
        )

        expected_results = {
            "Prompt Num": [0, 0],
            "Input Num": [0, 1],
            "Result Num": [0, 0],
            "Prompt vars": [{"word": "quack1"}, {"word": "quack2"}],
            "actual_text_result": ["QUACK QUACK1", "QUACK QUACK2"],
            "ground_truth_text_result": ["QUACK QUACK1", "NOT QUACK QUACK2"],
            "get_sum_of_lengths": [24, 28],
            "my_is_equal_fn": [True, False],
        }
        self._assert_is_expected_pandas_dataframe(
            results=results, expected_results=expected_results
        )

        self._assert_output_var_is_expected_results(
            var=_output_var,
            expected_results=expected_results,
            fake_env=fake_env,
        )


if __name__ == "__main__":
    absltest.main()
