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
"""Utilities for Commands.

Common methods for Commands such as RunCommand and CompileCommand.
"""
from __future__ import annotations

from typing import AbstractSet, Any, Callable, Sequence

from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import model_registry
from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_input_utils
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import unique_fn


class _GroundTruthLLMFunction(llm_function.LLMFunction):
    """LLMFunction that returns pre-generated ground truth data."""

    def __init__(self, data: Sequence[str]):
        super().__init__(outputs_ipython_display_fn=None)
        self._data = data

    def get_placeholders(self) -> AbstractSet[str]:
        # Ground truth is fixed and thus has no placeholders.
        return frozenset({})

    def _call_impl(
        self, inputs: llmfn_input_utils.LLMFunctionInputs | None
    ) -> Sequence[llmfn_outputs.LLMFnOutputEntry]:
        normalized_inputs = llmfn_input_utils.to_normalized_inputs(inputs)
        if len(self._data) != len(normalized_inputs):
            raise RuntimeError(
                "Ground truth should have same number of entries as inputs: {} vs {}".format(
                    len(self._data), len(normalized_inputs)
                )
            )

        outputs: list[llmfn_outputs.LLMFnOutputEntry] = []
        for idx, (value, prompt_vars) in enumerate(zip(self._data, normalized_inputs)):
            output_row = llmfn_output_row.LLMFnOutputRow(
                data={
                    llmfn_outputs.ColumnNames.RESULT_NUM: 0,
                    llmfn_outputs.ColumnNames.TEXT_RESULT: value,
                },
                result_type=str,
            )
            outputs.append(
                llmfn_outputs.LLMFnOutputEntry(
                    prompt_num=0,
                    input_num=idx,
                    prompt_vars=prompt_vars,
                    output_rows=[output_row],
                )
            )
        return outputs


def _get_ipython_display_fn(
    env: ipython_env.IPythonEnv,
) -> Callable[[llmfn_outputs.LLMFnOutputs], None]:
    return lambda x: env.display(x.as_pandas_dataframe())


def create_llm_function(
    models: model_registry.ModelRegistry,
    env: ipython_env.IPythonEnv | None,
    parsed_args: parsed_args_lib.ParsedArgs,
    cell_content: str,
    post_processing_fns: Sequence[post_process_utils.ParsedPostProcessExpr],
) -> llm_function.LLMFunction:
    """Creates an LLMFunction from Command.execute() arguments."""
    prompts: list[str] = [cell_content]

    llmfn_outputs_display_fn = _get_ipython_display_fn(env) if env else None

    llm_fn = llm_function.LLMFunctionImpl(
        model=models.get_model(parsed_args.model_type),
        model_args=parsed_args.model_args,
        prompts=prompts,
        outputs_ipython_display_fn=llmfn_outputs_display_fn,
    )
    if parsed_args.unique:
        llm_fn = llm_fn.add_post_process_reorder_fn(
            name="unique", fn=unique_fn.unique_fn
        )
    for fn in post_processing_fns:
        llm_fn = fn.add_to_llm_function(llm_fn)

    return llm_fn


def _convert_simple_compare_fn(
    name_and_simple_fn: tuple[str, Callable[[str, str], Any]]
) -> tuple[str, llm_function.CompareFn]:
    simple_fn = name_and_simple_fn[1]
    new_fn = lambda x, y: simple_fn(x.result_value(), y.result_value())
    return name_and_simple_fn[0], new_fn


def create_llm_compare_function(
    env: ipython_env.IPythonEnv | None,
    parsed_args: parsed_args_lib.ParsedArgs,
    post_processing_fns: Sequence[post_process_utils.ParsedPostProcessExpr],
) -> llm_function.LLMFunction:
    """Creates an LLMCompareFunction from Command.execute() arguments."""
    llmfn_outputs_display_fn = _get_ipython_display_fn(env) if env else None

    llm_cmp_fn = llm_function.LLMCompareFunction(
        lhs_name_and_fn=parsed_args.lhs_name_and_fn,
        rhs_name_and_fn=parsed_args.rhs_name_and_fn,
        compare_name_and_fns=[
            _convert_simple_compare_fn(x) for x in parsed_args.compare_fn
        ],
        outputs_ipython_display_fn=llmfn_outputs_display_fn,
    )
    for fn in post_processing_fns:
        llm_cmp_fn = fn.add_to_llm_function(llm_cmp_fn)

    return llm_cmp_fn


def create_llm_eval_function(
    models: model_registry.ModelRegistry,
    env: ipython_env.IPythonEnv | None,
    parsed_args: parsed_args_lib.ParsedArgs,
    cell_content: str,
    post_processing_fns: Sequence[post_process_utils.ParsedPostProcessExpr],
) -> llm_function.LLMFunction:
    """Creates an LLMCompareFunction from Command.execute() arguments."""
    llmfn_outputs_display_fn = _get_ipython_display_fn(env) if env else None

    # First construct a regular LLMFunction from the cell contents.
    llm_fn = create_llm_function(
        models=models,
        env=env,
        parsed_args=parsed_args,
        cell_content=cell_content,
        post_processing_fns=post_processing_fns,
    )

    # Next create a LLMCompareFunction.
    ground_truth_fn = _GroundTruthLLMFunction(data=parsed_args.ground_truth)
    llm_cmp_fn = llm_function.LLMCompareFunction(
        lhs_name_and_fn=("actual", llm_fn),
        rhs_name_and_fn=("ground_truth", ground_truth_fn),
        compare_name_and_fns=[
            _convert_simple_compare_fn(x) for x in parsed_args.compare_fn
        ],
        outputs_ipython_display_fn=llmfn_outputs_display_fn,
    )

    return llm_cmp_fn
