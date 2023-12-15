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
"""Parses an LLM command line."""
from __future__ import annotations

import argparse
import shlex
import sys
from typing import AbstractSet, Any, Callable, MutableMapping, Sequence

from google.generativeai.notebook import argument_parser
from google.generativeai.notebook import flag_def
from google.generativeai.notebook import input_utils
from google.generativeai.notebook import model_registry
from google.generativeai.notebook import output_utils
from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook import py_utils
from google.generativeai.notebook import sheets_utils
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_inputs_source
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import model as model_lib


_MIN_CANDIDATE_COUNT = 1
_MAX_CANDIDATE_COUNT = 8


def _validate_input_source_against_placeholders(
    source: llmfn_inputs_source.LLMFnInputsSource,
    placeholders: AbstractSet[str],
) -> None:
    for inputs in source.to_normalized_inputs():
        for keyword in placeholders:
            if keyword not in inputs:
                raise ValueError('Placeholder "{}" not found in input'.format(keyword))


def _get_resolve_input_from_py_var_fn(
    placeholders: AbstractSet[str] | None,
) -> Callable[[str], llmfn_inputs_source.LLMFnInputsSource]:
    def _fn(var_name: str) -> llmfn_inputs_source.LLMFnInputsSource:
        source = input_utils.get_inputs_source_from_py_var(var_name)
        if placeholders:
            _validate_input_source_against_placeholders(source, placeholders)
        return source

    return _fn


def _resolve_compare_fn_var(
    name: str,
) -> tuple[str, parsed_args_lib.TextResultCompareFn]:
    """Resolves a value passed into --compare_fn."""
    fn = py_utils.get_py_var(name)
    if not isinstance(fn, Callable):
        raise ValueError('Variable "{}" does not contain a Callable object'.format(name))

    return name, fn


def _resolve_ground_truth_var(name: str) -> Sequence[str]:
    """Resolves a value passed into --ground_truth."""
    value = py_utils.get_py_var(name)

    # "str" and "bytes" are also Sequences but we want an actual Sequence of
    # strings, like a list.
    if not isinstance(value, Sequence) or isinstance(value, str) or isinstance(value, bytes):
        raise ValueError('Variable "{}" does not contain a Sequence of strings'.format(name))
    for x in value:
        if not isinstance(x, str):
            raise ValueError('Variable "{}" does not contain a Sequence of strings'.format(name))
    return value


def _get_resolve_sheets_inputs_fn(
    placeholders: AbstractSet[str] | None,
) -> Callable[[str], llmfn_inputs_source.LLMFnInputsSource]:
    def _fn(value: str) -> llmfn_inputs_source.LLMFnInputsSource:
        sheets_id = sheets_utils.get_sheets_id_from_str(value)
        source = sheets_utils.SheetsInputs(sheets_id)
        if placeholders:
            _validate_input_source_against_placeholders(source, placeholders)
        return source

    return _fn


def _resolve_sheets_outputs(value: str) -> llmfn_outputs.LLMFnOutputsSink:
    sheets_id = sheets_utils.get_sheets_id_from_str(value)
    return sheets_utils.SheetsOutputs(sheets_id)


def _add_model_flags(
    parser: argparse.ArgumentParser,
) -> None:
    """Adds flags that are related to model selection and config."""
    flag_def.EnumFlagDef(
        name="model_type",
        short_name="mt",
        enum_type=model_registry.ModelName,
        default_value=model_registry.ModelRegistry.DEFAULT_MODEL,
        help_msg="The type of model to use.",
    ).add_argument_to_parser(parser)

    def _check_is_greater_than_or_equal_to_zero(x: float) -> float:
        if x < 0:
            raise ValueError("Value should be greater than or equal to zero, got {}".format(x))
        return x

    flag_def.SingleValueFlagDef(
        name="temperature",
        short_name="t",
        parse_type=float,
        # Use None for default value to indicate that this will use the default
        # value in Text service.
        default_value=None,
        parse_to_dest_type_fn=_check_is_greater_than_or_equal_to_zero,
        help_msg=(
            "Controls the randomness of the output. Must be positive. Typical"
            " values are in the range: [0.0, 1.0]. Higher values produce a more"
            " random and varied response. A temperature of zero will be"
            " deterministic."
        ),
    ).add_argument_to_parser(parser)

    flag_def.SingleValueFlagDef(
        name="model",
        short_name="m",
        default_value=None,
        help_msg=(
            "The name of the model to use. If not provided, a default model will" " be used."
        ),
    ).add_argument_to_parser(parser)

    def _check_candidate_count_range(x: Any) -> int:
        if x < _MIN_CANDIDATE_COUNT or x > _MAX_CANDIDATE_COUNT:
            raise ValueError(
                "Value should be in the range [{}, {}], got {}".format(
                    _MIN_CANDIDATE_COUNT, _MAX_CANDIDATE_COUNT, x
                )
            )
        return int(x)

    flag_def.SingleValueFlagDef(
        name="candidate_count",
        short_name="cc",
        parse_type=int,
        # Use None for default value to indicate that this will use the default
        # value in Text service.
        default_value=None,
        parse_to_dest_type_fn=_check_candidate_count_range,
        help_msg="The number of candidates to produce.",
    ).add_argument_to_parser(parser)

    flag_def.BooleanFlagDef(
        name="unique",
        help_msg="Whether to dedupe candidates returned by the model.",
    ).add_argument_to_parser(parser)


def _add_input_flags(
    parser: argparse.ArgumentParser,
    placeholders: AbstractSet[str] | None,
) -> None:
    """Adds flags to read inputs from a Python variable or Sheets."""
    flag_def.MultiValuesFlagDef(
        name="inputs",
        short_name="i",
        dest_type=llmfn_inputs_source.LLMFnInputsSource,
        parse_to_dest_type_fn=_get_resolve_input_from_py_var_fn(placeholders),
        help_msg=(
            "Optional names of Python variables containing inputs to use to"
            " instantiate a prompt. The variable must be either: a dictionary"
            " {'key1': ['val1', 'val2'] ...}, or an instance of LLMFnInputsSource"
            " such as SheetsInput."
        ),
    ).add_argument_to_parser(parser)

    flag_def.MultiValuesFlagDef(
        name="sheets_input_names",
        short_name="si",
        dest_type=llmfn_inputs_source.LLMFnInputsSource,
        parse_to_dest_type_fn=_get_resolve_sheets_inputs_fn(placeholders),
        help_msg=(
            "Optional names of Google Sheets to read inputs from. This is"
            " equivalent to using --inputs with the names of variables that are"
            " instances of SheetsInputs, just more convenient to use."
        ),
    ).add_argument_to_parser(parser)


def _add_output_flags(
    parser: argparse.ArgumentParser,
) -> None:
    """Adds flags to write outputs to a Python variable."""
    flag_def.MultiValuesFlagDef(
        name="outputs",
        short_name="o",
        dest_type=llmfn_outputs.LLMFnOutputsSink,
        parse_to_dest_type_fn=output_utils.get_outputs_sink_from_py_var,
        help_msg=(
            "Optional names of Python variables to output to. If the Python"
            " variable has not already been defined, it will be created. If the"
            " variable is defined and is an instance of LLMFnOutputsSink, the"
            " outputs will be written through the sink's write_outputs() method."
        ),
    ).add_argument_to_parser(parser)

    flag_def.MultiValuesFlagDef(
        name="sheets_output_names",
        short_name="so",
        dest_type=llmfn_outputs.LLMFnOutputsSink,
        parse_to_dest_type_fn=_resolve_sheets_outputs,
        help_msg=(
            "Optional names of Google Sheets to write inputs to. This is"
            " equivalent to using --outputs with the names of variables that are"
            " instances of SheetsOutputs, just more convenient to use."
        ),
    ).add_argument_to_parser(parser)


def _add_compare_flags(
    parser: argparse.ArgumentParser,
) -> None:
    flag_def.MultiValuesFlagDef(
        name="compare_fn",
        dest_type=tuple,
        parse_to_dest_type_fn=_resolve_compare_fn_var,
        help_msg=(
            "An optional function that takes two inputs: (lhs_result, rhs_result)"
            " which are the results of the left- and right-hand side functions. "
            "Multiple comparison functions can be provided."
        ),
    ).add_argument_to_parser(parser)


def _add_eval_flags(
    parser: argparse.ArgumentParser,
) -> None:
    flag_def.SingleValueFlagDef(
        name="ground_truth",
        required=True,
        dest_type=Sequence,
        parse_to_dest_type_fn=_resolve_ground_truth_var,
        help_msg=(
            "A variable containing a Sequence of strings representing the ground"
            " truth that the output of this cell will be compared against. It"
            " should have the same number of entries as inputs."
        ),
    ).add_argument_to_parser(parser)


def _create_run_parser(
    parser: argparse.ArgumentParser,
    placeholders: AbstractSet[str] | None,
) -> None:
    """Adds flags for the `run` command.

    `run` sends one or more prompts to a model.

    Args:
      parser: The parser to which flags will be added.
      placeholders: Placeholders from prompts in the cell contents.
    """
    _add_model_flags(parser)
    _add_input_flags(parser, placeholders)
    _add_output_flags(parser)


def _create_compile_parser(
    parser: argparse.ArgumentParser,
) -> None:
    """Adds flags for the compile command.

    `compile` "compiles" a prompt and model call into a callable function.

    Args:
      parser: The parser to which flags will be added.
    """

    # Add a positional argument for "compile_save_name".
    def _compile_save_name_fn(var_name: str) -> str:
        try:
            py_utils.validate_var_name(var_name)
        except ValueError as e:
            # Re-raise as ArgumentError to preserve the original error message.
            raise argparse.ArgumentError(None, "{}".format(e)) from e
        return var_name

    save_name_help = "The name of a Python variable to save the compiled function to."
    parser.add_argument("compile_save_name", help=save_name_help, type=_compile_save_name_fn)
    _add_model_flags(parser)


def _create_compare_parser(
    parser: argparse.ArgumentParser,
    placeholders: AbstractSet[str] | None,
) -> None:
    """Adds flags for the compare command.

    Args:
      parser: The parser to which flags will be added.
      placeholders: Placeholders from prompts in the compiled functions.
    """

    # Add positional arguments.
    def _resolve_llm_function_fn(
        var_name: str,
    ) -> tuple[str, llm_function.LLMFunction]:
        try:
            py_utils.validate_var_name(var_name)
        except ValueError as e:
            # Re-raise as ArgumentError to preserve the original error message.
            raise argparse.ArgumentError(None, "{}".format(e)) from e

        fn = py_utils.get_py_var(var_name)
        if not isinstance(fn, llm_function.LLMFunction):
            raise argparse.ArgumentError(
                None,
                '{} is not a function created with the "compile" command'.format(var_name),
            )
        return var_name, fn

    name_help = (
        "The name of a Python variable containing a function previously created"
        ' with the "compile" command.'
    )
    parser.add_argument("lhs_name_and_fn", help=name_help, type=_resolve_llm_function_fn)
    parser.add_argument("rhs_name_and_fn", help=name_help, type=_resolve_llm_function_fn)

    _add_input_flags(parser, placeholders)
    _add_output_flags(parser)
    _add_compare_flags(parser)


def _create_eval_parser(
    parser: argparse.ArgumentParser,
    placeholders: AbstractSet[str] | None,
) -> None:
    """Adds flags for the eval command.

    Args:
      parser: The parser to which flags will be added.
      placeholders: Placeholders from prompts in the cell contents.
    """
    _add_model_flags(parser)
    _add_input_flags(parser, placeholders)
    _add_output_flags(parser)
    _add_compare_flags(parser)
    _add_eval_flags(parser)


def _create_parser(
    placeholders: AbstractSet[str] | None,
) -> argparse.ArgumentParser:
    """Create the full parser."""
    system_name = "palm"
    description = "A system for interacting with LLMs."
    epilog = ""

    # Commands
    extra_args = {}
    if sys.version_info[0:2] >= (3, 9):
        extra_args["exit_on_error"] = False

    parser = argument_parser.ArgumentParser(
        prog=system_name,
        description=description,
        epilog=epilog,
        **extra_args,
    )
    subparsers = parser.add_subparsers(dest="cmd")
    _create_run_parser(
        subparsers.add_parser(parsed_args_lib.CommandName.RUN_CMD.value),
        placeholders,
    )
    _create_compile_parser(subparsers.add_parser(parsed_args_lib.CommandName.COMPILE_CMD.value))
    _create_compare_parser(
        subparsers.add_parser(parsed_args_lib.CommandName.COMPARE_CMD.value),
        placeholders,
    )
    _create_eval_parser(
        subparsers.add_parser(parsed_args_lib.CommandName.EVAL_CMD.value),
        placeholders,
    )
    return parser


def _validate_parsed_args(parsed_args: parsed_args_lib.ParsedArgs) -> None:
    # If candidate_count is not set (i.e. is None), assuming the default value
    # is 1.
    if parsed_args.unique and (
        parsed_args.model_args.candidate_count is None
        or parsed_args.model_args.candidate_count == 1
    ):
        print(
            '"--unique" works across candidates only: it should be used with'
            " --candidate_count set to a value greater-than one."
        )


class CmdLineParser:
    """Implementation of Magics command line parser."""

    # Commands
    DEFAULT_CMD = parsed_args_lib.CommandName.RUN_CMD

    # Post-processing operator.
    PIPE_OP = "|"

    @classmethod
    def _split_post_processing_tokens(
        cls,
        tokens: Sequence[str],
    ) -> tuple[Sequence[str], parsed_args_lib.PostProcessingTokens]:
        """Splits inputs into the command and post processing tokens.

        The command is represented as a sequence of tokens.
        See comments on the PostProcessingTokens type alias.

        E.g. Given: "run --temperature 0.5 | add_score | to_lower_case"
        The command will be: ["run", "--temperature", "0.5"].
        The post processing tokens will be: [["add_score"], ["to_lower_case"]]

        Args:
          tokens: The command line tokens.

        Returns:
          A tuple of (command line, post processing tokens).
        """
        split_tokens = []
        start_idx: int | None = None
        for token_num, token in enumerate(tokens):
            if start_idx is None:
                start_idx = token_num
            if token == CmdLineParser.PIPE_OP:
                split_tokens.append(tokens[start_idx:token_num] if start_idx is not None else [])
                start_idx = None

        # Add the remaining tokens after the last PIPE_OP.
        split_tokens.append(tokens[start_idx:] if start_idx is not None else [])

        return split_tokens[0], split_tokens[1:]

    @classmethod
    def _tokenize_line(
        cls, line: str
    ) -> tuple[Sequence[str], parsed_args_lib.PostProcessingTokens]:
        """Parses `line` and returns command line and post processing tokens."""
        # Check to make sure there is a command at the start. If not, add the
        # default command to the list of tokens.
        tokens = shlex.split(line)
        if not tokens:
            tokens = [CmdLineParser.DEFAULT_CMD.value]
        first_token = tokens[0]
        # Add default command if the first token is not the help token.
        if not first_token[0].isalpha() and first_token not in ["-h", "--help"]:
            tokens = [CmdLineParser.DEFAULT_CMD.value] + tokens
        # Split line into tokens and post-processing
        return CmdLineParser._split_post_processing_tokens(tokens)

    @classmethod
    def _get_model_args(
        cls, parsed_results: MutableMapping[str, Any]
    ) -> tuple[MutableMapping[str, Any], model_lib.ModelArguments]:
        """Extracts fields for model args from `parsed_results`.

        Keys specific to model arguments will be removed from `parsed_results`.

        Args:
          parsed_results: A dictionary of parsed arguments (from ArgumentParser). It
            will be modified in place.

        Returns:
          A tuple of (updated parsed_results, model arguments).
        """
        model = parsed_results.pop("model", None)
        temperature = parsed_results.pop("temperature", None)
        candidate_count = parsed_results.pop("candidate_count", None)

        model_args = model_lib.ModelArguments(
            model=model,
            temperature=temperature,
            candidate_count=candidate_count,
        )
        return parsed_results, model_args

    def parse_line(
        self,
        line: str,
        placeholders: AbstractSet[str] | None = None,
    ) -> tuple[parsed_args_lib.ParsedArgs, parsed_args_lib.PostProcessingTokens]:
        """Parses the commandline and returns ParsedArgs and post-processing tokens.

        Args:
          line: The line to parse (usually contents from cell Magics).
          placeholders: Placeholders from prompts in the cell contents.

        Returns:
          A tuple of (parsed_args, post_processing_tokens).
        """
        tokens, post_processing_tokens = CmdLineParser._tokenize_line(line)

        parsed_args = self._get_parsed_args_from_cmd_line_tokens(
            tokens=tokens, placeholders=placeholders
        )

        # Special-case for "compare" command: because the prompts are compiled into
        # the left- and right-hand side functions rather than in the cell body, we
        # cannot examine the cell body to get the placeholders.
        #
        # Instead we parse the command line twice: once to get the left- and right-
        # functions, then we query the functions for their placeholders, then
        # parse the commandline again to validate the inputs.
        if parsed_args.cmd == parsed_args_lib.CommandName.COMPARE_CMD:
            assert parsed_args.lhs_name_and_fn is not None
            assert parsed_args.rhs_name_and_fn is not None
            _, lhs_fn = parsed_args.lhs_name_and_fn
            _, rhs_fn = parsed_args.rhs_name_and_fn
            parsed_args = self._get_parsed_args_from_cmd_line_tokens(
                tokens=tokens,
                placeholders=frozenset(lhs_fn.get_placeholders()).union(rhs_fn.get_placeholders()),
            )

        _validate_parsed_args(parsed_args)

        for expr in post_processing_tokens:
            post_process_utils.validate_one_post_processing_expression(expr)

        return parsed_args, post_processing_tokens

    def _get_parsed_args_from_cmd_line_tokens(
        self,
        tokens: Sequence[str],
        placeholders: AbstractSet[str] | None,
    ) -> parsed_args_lib.ParsedArgs:
        """Returns ParsedArgs from a tokenized command line."""
        # Create a new parser to avoid reusing the temporary argparse.Namespace
        # object.
        results = _create_parser(placeholders).parse_args(tokens)

        results_dict = vars(results)
        results_dict["cmd"] = parsed_args_lib.CommandName(results_dict["cmd"])

        results_dict, model_args = CmdLineParser._get_model_args(results_dict)
        results_dict["model_args"] = model_args

        return parsed_args_lib.ParsedArgs(**results_dict)
