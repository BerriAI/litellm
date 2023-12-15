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
"""Output of LLMFunction."""
from __future__ import annotations

import abc
import dataclasses
from typing import (
    overload,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)

from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import model as model_lib
import pandas


class ColumnNames:
    """Names of columns that are used to represent output."""

    PROMPT_NUM = "Prompt Num"
    INPUT_NUM = "Input Num"
    RESULT_NUM = "Result Num"
    # In the code we refer to "model_input" as the full keyword-substituted prompt
    # and "prompt" as the template with placeholders.
    # When displaying the results however we use "prompt" since "model_input" is
    # an internal name.
    MODEL_INPUT = "Prompt"
    PROMPT_VARS = "Prompt vars"
    TEXT_RESULT = "text_result"


@dataclasses.dataclass
class LLMFnOutputEntry:
    """The output of a single model input from LLMFunction.

    A model input is a prompt where the keyword placeholders have been
    substituted (by `prompt_vars`).

    E.g. If we have:
      prompt: "the opposite of {word} is"
      prompt_vars: {"word", "hot"}
    Then we will have the following model input:
      model_input: "the opposite of hot is"

    Note: The model may produce one-or-more results for a given model_input.
    This is represented by the sequence `output_rows`.
    """

    prompt_num: int
    input_num: int
    prompt_vars: Mapping[str, str]
    output_rows: Sequence[llmfn_output_row.LLMFnOutputRow]

    prompt: str | None = None
    model_input: str | None = None
    model_results: model_lib.ModelResults | None = None


def _has_model_input_field(outputs: Iterable[LLMFnOutputEntry]):
    for entry in outputs:
        if entry.model_input is not None:
            return True
    return False


class LLMFnOutputsBase(Sequence[LLMFnOutputEntry]):
    """Parent class for LLMFnOutputs.

    This class exists mainly to avoid a circular dependency between LLMFnOutputs
    and LLMFnOutputsSink. Most users should use LLMFnOutputs directly instead.
    """

    def __init__(
        self,
        outputs: Iterable[LLMFnOutputEntry] | None = None,
    ):
        """Constructor.

        Args:
          outputs: The contents of this LLMFnOutputs instance.
        """
        self._outputs: list[LLMFnOutputEntry] = list(outputs) if outputs is not None else []

    # Needed for Iterable[LLMFnOutputEntry].
    def __iter__(self) -> Iterator[LLMFnOutputEntry]:
        return self._outputs.__iter__()

    # Needed for Sequence[LLMFnOutputEntry].
    def __len__(self) -> int:
        return self._outputs.__len__()

    # Needed for Sequence[LLMFnOutputEntry].
    @overload
    def __getitem__(self, x: int) -> LLMFnOutputEntry:
        ...

    @overload
    def __getitem__(self, x: slice) -> Sequence[LLMFnOutputEntry]:
        ...

    def __getitem__(self, x: int | slice) -> LLMFnOutputEntry | Sequence[LLMFnOutputEntry]:
        return self._outputs.__getitem__(x)

    # Convenience methods.
    def __bool__(self) -> bool:
        return bool(self._outputs)

    def __str__(self) -> str:
        return self.as_pandas_dataframe().__str__()

    # Own methods
    def as_dict(self) -> Mapping[str, Sequence[Any]]:
        """Formats returned results as dictionary."""

        # `data` is a table in column order, with the columns listed from left to
        # right.
        data = {
            ColumnNames.PROMPT_NUM: [],
            ColumnNames.INPUT_NUM: [],
            # RESULT_NUM is special: each LLMFnOutputRow in self._outputs is
            # expected to have a RESULT_NUM key.
            ColumnNames.RESULT_NUM: [],
        }
        if _has_model_input_field(self._outputs):
            data[ColumnNames.MODEL_INPUT] = []

        if not self._outputs:
            return data

        # Add column names of added data.
        # The last key in LLMFnOutputRow is special as it is considered
        # the result. To preserve order in the (unlikely) event of inconsistent
        # keys across rows, we first add all-but-the-last key to `total_keys_set`,
        # then the last key.
        # Note: `total_keys_set` is a Python dictionary instead of a Python set
        # because Python dictionaries preserve the order in which entries are
        # added, whereas Python sets do not.
        total_keys_set: dict[str, None] = {k: None for k in data.keys()}
        for output in self._outputs:
            for result in output.output_rows:
                for key in list(result.keys())[:-1]:
                    total_keys_set[key] = None
        for output in self._outputs:
            for result in output.output_rows:
                total_keys_set[list(result.keys())[-1]] = None

        # `data` represents the table as a dictionary of:
        #   column names -> list of values
        for key in total_keys_set:
            data[key] = []

        next_num_rows = 1
        for output in self._outputs:
            for result in output.output_rows:
                data[ColumnNames.PROMPT_NUM].append(output.prompt_num)
                data[ColumnNames.INPUT_NUM].append(output.input_num)
                if ColumnNames.MODEL_INPUT in data:
                    data[ColumnNames.MODEL_INPUT].append(output.model_input)

                for key, value in result.items():
                    data[key].append(value)

                # Look for empty cells and pad them with None.
                for column in data.values():
                    if len(column) < next_num_rows:
                        column.append(None)

                next_num_rows += 1

        return data

    def as_pandas_dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.as_dict())


class LLMFnOutputsSink(abc.ABC):
    """Abstract class representing an exporter for the output of LLMFunction.

    This class could be extended to write to external documents, such as
    Google Sheets.
    """

    def write_outputs(self, outputs: LLMFnOutputsBase) -> None:
        """Writes `outputs` to some destination."""


class LLMFnOutputs(LLMFnOutputsBase):
    """A sequence of LLMFnOutputEntry instances.

    Notes:
    - Each LLMFnOutputEntry represents the results of running one model
      input (see documentation for LLMFnOutputEntry for what "model input"
      means.)
    - A single model input may produce more-than-one text results.
    """

    def __init__(
        self,
        outputs: Iterable[LLMFnOutputEntry] | None = None,
        ipython_display_fn: Callable[[LLMFnOutputs], None] | None = None,
    ):
        """Constructor.

        Args:
          outputs: The contents of this LLMFnOutputs instance.
          ipython_display_fn: An optional function for pretty-printing this instance
            when it is the output of a cell in a notebook. If this argument is not
            None, the _ipython_display_ method will be defined which will in turn
            invoke this function.
        """
        super().__init__(outputs=outputs)

        if ipython_display_fn:
            self._ipython_display_fn = ipython_display_fn
            # We define the _ipython_display_ method only when `ipython_display_fn`
            # is set. This lets us fall back to a default implementation defined by
            # the notebook when `ipython_display_fn` is not set, instead of having to
            # provide our own default implementation.
            setattr(
                self,
                "_ipython_display_",
                getattr(self, "_ipython_display_impl"),
            )

    def _ipython_display_impl(self):
        """Actual implementation of _ipython_display_.

        This method should only be used invoked if self._ipython_display_fn is set.
        """
        self._ipython_display_fn(self)

    def export(self, sink: LLMFnOutputsSink) -> None:
        """Export contents to `sink`."""
        sink.write_outputs(self)
