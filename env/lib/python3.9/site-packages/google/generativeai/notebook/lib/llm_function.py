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
"""LLMFunction."""
from __future__ import annotations

import abc
import dataclasses
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from google.generativeai.notebook.lib import llmfn_input_utils
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_outputs
from google.generativeai.notebook.lib import llmfn_post_process
from google.generativeai.notebook.lib import llmfn_post_process_cmds
from google.generativeai.notebook.lib import model as model_lib
from google.generativeai.notebook.lib import prompt_utils


# In the same spirit as post-processing functions (see: llmfn_post_process.py),
# we keep the LLM functions more flexible by providing the entire left- and
# right-hand side rows to the user-defined comparison function.
#
# Possible use-cases include adding a scoring function as a post-process
# command, then comparing the scores.
CompareFn = Callable[
    [llmfn_output_row.LLMFnOutputRowView, llmfn_output_row.LLMFnOutputRowView],
    Any,
]


def _is_equal_fn(
    lhs: llmfn_output_row.LLMFnOutputRowView,
    rhs: llmfn_output_row.LLMFnOutputRowView,
) -> bool:
    """Default function used when comparing outputs."""
    return lhs.result_value() == rhs.result_value()


def _convert_compare_fn_to_batch_add_fn(
    fn: Callable[
        [
            llmfn_output_row.LLMFnOutputRowView,
            llmfn_output_row.LLMFnOutputRowView,
        ],
        Any,
    ]
) -> llmfn_post_process.LLMCompareFnPostProcessBatchAddFn:
    """Vectorize a single-row-based comparison function."""

    def _fn(
        lhs_and_rhs_rows: Sequence[
            tuple[
                llmfn_output_row.LLMFnOutputRowView,
                llmfn_output_row.LLMFnOutputRowView,
            ]
        ]
    ) -> Sequence[Any]:
        return [fn(lhs, rhs) for lhs, rhs in lhs_and_rhs_rows]

    return _fn


@dataclasses.dataclass
class _PromptInfo:
    prompt_num: int
    prompt: str
    input_num: int
    prompt_vars: Mapping[str, str]
    model_input: str


def _generate_prompts(
    prompts: Sequence[str], inputs: llmfn_input_utils.LLMFunctionInputs | None
) -> Iterable[_PromptInfo]:
    """Generate a tuple of fields needed for processing prompts.

    Args:
      prompts: A list of prompts, with optional keyword placeholders.
      inputs: A list of key/value pairs to substitute into placeholders in
        `prompts`.

    Yields:
      A _PromptInfo instance.
    """
    normalized_inputs: Sequence[Mapping[str, str]] = []
    if inputs is not None:
        normalized_inputs = llmfn_input_utils.to_normalized_inputs(inputs)

    # Must have at least one entry so that we execute the prompt at least once.
    if not normalized_inputs:
        normalized_inputs = [{}]

    for prompt_num, prompt in enumerate(prompts):
        for input_num, prompt_vars in enumerate(normalized_inputs):
            # Perform keyword substitution on the prompt based on `prompt_vars`.
            model_input = prompt.format(**prompt_vars)
            yield _PromptInfo(
                prompt_num=prompt_num,
                prompt=prompt,
                input_num=input_num,
                prompt_vars=prompt_vars,
                model_input=model_input,
            )


class LLMFunction(
    Callable[
        [Union[llmfn_input_utils.LLMFunctionInputs, None]],
        llmfn_outputs.LLMFnOutputs,
    ],
    metaclass=abc.ABCMeta,
):
    """Base class for LLMFunctionImpl and LLMCompareFunction."""

    def __init__(
        self,
        outputs_ipython_display_fn: Callable[[llmfn_outputs.LLMFnOutputs], None] | None = None,
    ):
        """Constructor.

        Args:
          outputs_ipython_display_fn: Optional function that will be used to
            override how the outputs of this LLMFunction will be displayed in a
            notebook (See further documentation in LLMFnOutputs.__init__().)
        """
        self._post_process_cmds: list[llmfn_post_process_cmds.LLMFnPostProcessCommand] = []
        self._outputs_ipython_display_fn = outputs_ipython_display_fn

    @abc.abstractmethod
    def get_placeholders(self) -> AbstractSet[str]:
        """Returns the placeholders that should be present in inputs for this function."""

    @abc.abstractmethod
    def _call_impl(
        self, inputs: llmfn_input_utils.LLMFunctionInputs | None
    ) -> Sequence[llmfn_outputs.LLMFnOutputEntry]:
        """Concrete implementation of __call__()."""

    def __call__(
        self, inputs: llmfn_input_utils.LLMFunctionInputs | None = None
    ) -> llmfn_outputs.LLMFnOutputs:
        """Runs and returns results based on `inputs`."""
        outputs = self._call_impl(inputs)

        return llmfn_outputs.LLMFnOutputs(
            outputs=outputs, ipython_display_fn=self._outputs_ipython_display_fn
        )

    def add_post_process_reorder_fn(
        self, name: str, fn: llmfn_post_process.LLMFnPostProcessBatchReorderFn
    ) -> LLMFunction:
        self._post_process_cmds.append(
            llmfn_post_process_cmds.LLMFnPostProcessReorderCommand(name=name, fn=fn)
        )
        return self

    def add_post_process_add_fn(
        self,
        name: str,
        fn: llmfn_post_process.LLMFnPostProcessBatchAddFn,
    ) -> LLMFunction:
        self._post_process_cmds.append(
            llmfn_post_process_cmds.LLMFnPostProcessAddCommand(name=name, fn=fn)
        )
        return self

    def add_post_process_replace_fn(
        self,
        name: str,
        fn: llmfn_post_process.LLMFnPostProcessBatchReplaceFn,
    ) -> LLMFunction:
        self._post_process_cmds.append(
            llmfn_post_process_cmds.LLMFnPostProcessReplaceCommand(name=name, fn=fn)
        )
        return self


class LLMFunctionImpl(LLMFunction):
    """Callable class that executes the contents of a Magics cell.

    An LLMFunction is constructed from the Magics command line and cell contents
    specified by the user. It is defined by:
    - A model instance,
    - Model arguments
    - A prompt template (e.g. "the opposite of hot is {word}") with an optional
      keyword placeholder.

    The LLMFunction takes as its input a sequence of dictionaries containing
    values for keyword replacement, e.g. [{"word": "hot"}, {"word": "tall"}].

    This will cause the model to be executed with the following prompts:
      "The opposite of hot is"
      "The opposite of tall is"

    The results will be returned in a LLMFnOutputs instance.
    """

    def __init__(
        self,
        model: model_lib.AbstractModel,
        prompts: Sequence[str],
        model_args: model_lib.ModelArguments | None = None,
        outputs_ipython_display_fn: Callable[[llmfn_outputs.LLMFnOutputs], None] | None = None,
    ):
        """Constructor.

        Args:
          model: The model that the prompts will execute on.
          prompts: A sequence of prompt templates with optional placeholders. The
            placeholders will be replaced by the inputs passed into this function.
          model_args: Optional set of model arguments to configure how the model
            executes the prompts.
          outputs_ipython_display_fn: See documentation in LLMFunction.__init__().
        """
        super().__init__(outputs_ipython_display_fn=outputs_ipython_display_fn)
        self._model = model
        self._prompts = prompts
        self._model_args = model_lib.ModelArguments() if model_args is None else model_args

        # Compute placeholders.
        self._placeholders = frozenset({})
        for prompt in self._prompts:
            self._placeholders = self._placeholders.union(prompt_utils.get_placeholders(prompt))

    def _run_post_processing_cmds(
        self, results: Sequence[llmfn_output_row.LLMFnOutputRow]
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        """Runs post-processing commands over `results`."""
        for cmd in self._post_process_cmds:
            try:
                if isinstance(cmd, llmfn_post_process_cmds.LLMFnImplPostProcessCommand):
                    results = cmd.run(results)
                else:
                    raise llmfn_post_process.PostProcessExecutionError(
                        "Unsupported post-process command type: {}".format(type(cmd))
                    )
            except llmfn_post_process.PostProcessExecutionError:
                raise
            except RuntimeError as e:
                raise llmfn_post_process.PostProcessExecutionError(
                    'Error executing "{}", got {}: {}'.format(cmd.name(), type(e).__name__, e)
                )
        return results

    def get_placeholders(self) -> AbstractSet[str]:
        return self._placeholders

    def _call_impl(
        self, inputs: llmfn_input_utils.LLMFunctionInputs | None
    ) -> Sequence[llmfn_outputs.LLMFnOutputEntry]:
        results: list[llmfn_outputs.LLMFnOutputEntry] = []
        for info in _generate_prompts(prompts=self._prompts, inputs=inputs):
            model_results = self._model.call_model(
                model_input=info.model_input, model_args=self._model_args
            )
            output_rows: list[llmfn_output_row.LLMFnOutputRow] = []
            for result_num, text_result in enumerate(model_results.text_results):
                output_rows.append(
                    llmfn_output_row.LLMFnOutputRow(
                        data={
                            llmfn_outputs.ColumnNames.RESULT_NUM: result_num,
                            llmfn_outputs.ColumnNames.TEXT_RESULT: text_result,
                        },
                        result_type=str,
                    )
                )
            results.append(
                llmfn_outputs.LLMFnOutputEntry(
                    prompt_num=info.prompt_num,
                    input_num=info.input_num,
                    prompt=info.prompt,
                    prompt_vars=info.prompt_vars,
                    model_input=info.model_input,
                    model_results=model_results,
                    output_rows=self._run_post_processing_cmds(output_rows),
                )
            )
        return results


class LLMCompareFunction(LLMFunction):
    """LLMFunction for comparisons.

    LLMCompareFunction runs an input over a pair of LLMFunctions and compares the
    result.
    """

    def __init__(
        self,
        lhs_name_and_fn: tuple[str, LLMFunction],
        rhs_name_and_fn: tuple[str, LLMFunction],
        compare_name_and_fns: Sequence[tuple[str, CompareFn]] | None = None,
        outputs_ipython_display_fn: Callable[[llmfn_outputs.LLMFnOutputs], None] | None = None,
    ):
        """Constructor.

        Args:
          lhs_name_and_fn: Name and function for the left-hand side of the
            comparison.
          rhs_name_and_fn: Name and function for the right-hand side of the
            comparison.
          compare_name_and_fns: Optional names and functions for comparing the
            results of the left- and right-hand sides.
          outputs_ipython_display_fn: See documentation in LLMFunction.__init__().
        """
        super().__init__(outputs_ipython_display_fn=outputs_ipython_display_fn)
        self._lhs_name: str = lhs_name_and_fn[0]
        self._lhs_fn: LLMFunction = lhs_name_and_fn[1]
        self._rhs_name: str = rhs_name_and_fn[0]
        self._rhs_fn: LLMFunction = rhs_name_and_fn[1]
        self._placeholders = frozenset(self._lhs_fn.get_placeholders()).union(
            self._rhs_fn.get_placeholders()
        )

        if not compare_name_and_fns:
            self._result_name = "is_equal"
            self._result_compare_fn = _is_equal_fn
        else:
            # Assume the last entry in `compare_name_and_fns` is the one that
            # produces value for the result cell.
            name, fn = compare_name_and_fns[-1]
            self._result_name = name
            self._result_compare_fn = fn

            # Treat the other compare_fns as post-processing operators.
            for name, cmp_fn in compare_name_and_fns[:-1]:
                self.add_compare_post_process_add_fn(
                    name=name, fn=_convert_compare_fn_to_batch_add_fn(cmp_fn)
                )

    def _run_post_processing_cmds(
        self,
        lhs_output_rows: Sequence[llmfn_output_row.LLMFnOutputRow],
        rhs_output_rows: Sequence[llmfn_output_row.LLMFnOutputRow],
        results: Sequence[llmfn_output_row.LLMFnOutputRow],
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        """Runs post-processing commands over `results`."""
        for cmd in self._post_process_cmds:
            try:
                if isinstance(cmd, llmfn_post_process_cmds.LLMFnImplPostProcessCommand):
                    results = cmd.run(results)
                elif isinstance(cmd, llmfn_post_process_cmds.LLMCompareFnPostProcessCommand):
                    results = cmd.run(list(zip(lhs_output_rows, rhs_output_rows, results)))
                else:
                    raise RuntimeError(
                        "Unsupported post-process command type: {}".format(type(cmd))
                    )
            except llmfn_post_process.PostProcessExecutionError:
                raise
            except RuntimeError as e:
                raise llmfn_post_process.PostProcessExecutionError(
                    'Error executing "{}", got {}: {}'.format(cmd.name(), type(e).__name__, e)
                )
        return results

    def get_placeholders(self) -> AbstractSet[str]:
        return self._placeholders

    def _call_impl(
        self, inputs: llmfn_input_utils.LLMFunctionInputs | None
    ) -> Sequence[llmfn_outputs.LLMFnOutputEntry]:
        lhs_results = self._lhs_fn(inputs)
        rhs_results = self._rhs_fn(inputs)

        # Combine the results.
        outputs: list[llmfn_outputs.LLMFnOutputEntry] = []
        for lhs_entry, rhs_entry in zip(lhs_results, rhs_results):
            if lhs_entry.prompt_num != rhs_entry.prompt_num:
                raise RuntimeError(
                    "Prompt num mismatch: {} vs {}".format(
                        lhs_entry.prompt_num, rhs_entry.prompt_num
                    )
                )
            if lhs_entry.input_num != rhs_entry.input_num:
                raise RuntimeError(
                    "Input num mismatch: {} vs {}".format(lhs_entry.input_num, rhs_entry.input_num)
                )
            if lhs_entry.prompt_vars != rhs_entry.prompt_vars:
                raise RuntimeError(
                    "Prompt vars mismatch: {} vs {}".format(
                        lhs_entry.prompt_vars, rhs_entry.prompt_vars
                    )
                )

            # The two functions may have different numbers of results due to
            # options like candidate_count, so we can only compare up to the
            # minimum of the two.
            num_output_rows = min(len(lhs_entry.output_rows), len(rhs_entry.output_rows))
            lhs_output_rows = lhs_entry.output_rows[:num_output_rows]
            rhs_output_rows = rhs_entry.output_rows[:num_output_rows]
            output_rows: list[llmfn_output_row.LLMFnOutputRow] = []
            for result_num, lhs_and_rhs_output_row in enumerate(
                zip(lhs_output_rows, rhs_output_rows)
            ):
                lhs_output_row, rhs_output_row = lhs_and_rhs_output_row

                # Combine cells from lhs_output_row and rhs_output_row into a
                # single row.
                # Although it is possible for RESULT_NUM (the index of each
                # text_result if a prompt produces multiple text_results) to be
                # different between the left and right sides, we ignore their
                # RESULT_NUM entries and write our own.
                row_data: dict[str, Any] = {
                    llmfn_outputs.ColumnNames.RESULT_NUM: result_num,
                    self._result_name: self._result_compare_fn(lhs_output_row, rhs_output_row),
                }
                output_row = llmfn_output_row.LLMFnOutputRow(data=row_data, result_type=Any)

                # Add the prompt vars.
                output_row.add(llmfn_outputs.ColumnNames.PROMPT_VARS, lhs_entry.prompt_vars)

                # Add the results from the left-hand side and right-hand side.
                for name, row in [
                    (self._lhs_name, lhs_output_row),
                    (self._rhs_name, rhs_output_row),
                ]:
                    for k, v in row.items():
                        if k != llmfn_outputs.ColumnNames.RESULT_NUM:
                            # We use LLMFnOutputRow.add() because it handles column
                            # name collisions.
                            output_row.add("{}_{}".format(name, k), v)

                output_rows.append(output_row)

            outputs.append(
                llmfn_outputs.LLMFnOutputEntry(
                    prompt_num=lhs_entry.prompt_num,
                    input_num=lhs_entry.input_num,
                    prompt_vars=lhs_entry.prompt_vars,
                    output_rows=self._run_post_processing_cmds(
                        lhs_output_rows=lhs_output_rows,
                        rhs_output_rows=rhs_output_rows,
                        results=output_rows,
                    ),
                )
            )
        return outputs

    def add_compare_post_process_add_fn(
        self,
        name: str,
        fn: llmfn_post_process.LLMCompareFnPostProcessBatchAddFn,
    ) -> LLMFunction:
        self._post_process_cmds.append(
            llmfn_post_process_cmds.LLMCompareFnPostProcessAddCommand(name=name, fn=fn)
        )
        return self
