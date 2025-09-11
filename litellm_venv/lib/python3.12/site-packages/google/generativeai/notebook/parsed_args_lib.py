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
"""Results from parsing the commandline.

This module separates the results from commandline parsing from the parser
itself so that classes that operate on the results (e.g. the subclasses of
Command) do not have to depend on the commandline parser as well.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Sequence

from google.generativeai.notebook import model_registry
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_inputs_source
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import model as model_lib

# Post processing tokens are represented as a sequence of sequence of tokens,
# because the pipe operator could be used more than once.
PostProcessingTokens = Sequence[Sequence[str]]

# The type of function taken by the "compare_fn" flag.
# It takes the text_results of the left- and right-hand side functions as
# inputs and returns a comparison result.
TextResultCompareFn = Callable[[str, str], Any]


class CommandName(enum.Enum):
    RUN_CMD = "run"
    COMPILE_CMD = "compile"
    COMPARE_CMD = "compare"
    EVAL_CMD = "eval"


@dataclasses.dataclass(frozen=True)
class ParsedArgs:
    """The results of parsing the command line."""

    cmd: CommandName

    # For run, compile and eval commands.
    model_args: model_lib.ModelArguments
    model_type: model_registry.ModelName | None = None
    unique: bool = False

    # For run, compare and eval commands.
    inputs: Sequence[llmfn_inputs_source.LLMFnInputsSource] = dataclasses.field(
        default_factory=list
    )
    sheets_input_names: Sequence[
        llmfn_inputs_source.LLMFnInputsSource
    ] = dataclasses.field(default_factory=list)

    outputs: Sequence[llmfn_outputs.LLMFnOutputsSink] = dataclasses.field(
        default_factory=list
    )
    sheets_output_names: Sequence[llmfn_outputs.LLMFnOutputsSink] = dataclasses.field(
        default_factory=list
    )

    # For compile command.
    compile_save_name: str | None = None

    # For compare command.
    lhs_name_and_fn: tuple[str, llm_function.LLMFunction] | None = None
    rhs_name_and_fn: tuple[str, llm_function.LLMFunction] | None = None

    # For compare and eval commands.
    compare_fn: Sequence[tuple[str, TextResultCompareFn]] = dataclasses.field(
        default_factory=list
    )

    # For eval command.
    ground_truth: Sequence[str] = dataclasses.field(default_factory=list)
