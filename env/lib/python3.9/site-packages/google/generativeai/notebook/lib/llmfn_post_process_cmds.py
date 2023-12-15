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
"""Internal representation of post-process commands for LLMFunction.

This module is internal to LLMFunction and should only be used by
llm_function.py.
"""
from __future__ import annotations

import abc
from typing import Sequence

from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_post_process


def _convert_view_to_output_row(
    row: llmfn_output_row.LLMFnOutputRowView,
) -> llmfn_output_row.LLMFnOutputRow:
    """Convenience method to conert a LLMFnOutputRowView to LLMFnOutputRow.

    If `row` is already a LLMFnOutputRow, return as-is for efficiency.
    This could potentially break encapsulation as it could let code to modify
    a LLMFnOutputRowView that was intended to be immutable, so it should be
    used with care.

    Args:
      row: An instance of LLMFnOutputRowView.

    Returns:
      An instance of LLMFnOutputRow. May be the same instance as `row` if
      `row` is already an instance of LLMFnOutputRow.
    """
    if isinstance(row, llmfn_output_row.LLMFnOutputRow):
        return row
    return llmfn_output_row.LLMFnOutputRow(data=row, result_type=row.result_type())


class LLMFnPostProcessCommand(abc.ABC):
    """Abstract class representing post-processing commands."""

    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of this post-processing command."""


class LLMFnImplPostProcessCommand(LLMFnPostProcessCommand):
    """Post-processing commands for LLMFunctionImpl."""

    @abc.abstractmethod
    def run(
        self, rows: Sequence[llmfn_output_row.LLMFnOutputRowView]
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        """Processes a batch of results and returns a new batch.

        Args:
          rows: The rows in a batch. Note that `rows` are not guaranteed to be
            remain unmodified.

        Returns:
          A new set of rows that should replace the batch.
        """


class LLMFnPostProcessReorderCommand(LLMFnImplPostProcessCommand):
    """A batch command processes a set of results at once.

    Note that a "batch" represents a set of results coming from a single prompt,
    as the model may produce more-than-one result for a prompt.
    """

    def __init__(self, name: str, fn: llmfn_post_process.LLMFnPostProcessBatchReorderFn):
        self._name = name
        self._fn = fn

    def name(self) -> str:
        return self._name

    def run(
        self,
        rows: Sequence[llmfn_output_row.LLMFnOutputRowView],
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        new_row_indices = self._fn(rows)
        if len(set(new_row_indices)) != len(new_row_indices):
            raise llmfn_post_process.PostProcessExecutionError(
                'Error executing "{}": returned indices should be unique'.format(self._name)
            )

        new_rows: list[llmfn_output_row.LLMFnOutputRow] = []
        for idx in new_row_indices:
            if idx < 0:
                raise llmfn_post_process.PostProcessExecutionError(
                    'Error executing "{}": returned indices must be greater than or'
                    " equal to zero, got {}".format(self._name, idx)
                )
            if idx >= len(rows):
                raise llmfn_post_process.PostProcessExecutionError(
                    'Error executing "{}": returned indices must be less than length of'
                    " rows (={}), got {}".format(self._name, len(rows), idx)
                )
            new_rows.append(_convert_view_to_output_row(rows[idx]))
        return new_rows


class LLMFnPostProcessAddCommand(LLMFnImplPostProcessCommand):
    """A command that adds each row with a new column.

    This does not change the value of the results cell.
    """

    def __init__(self, name: str, fn: llmfn_post_process.LLMFnPostProcessBatchAddFn):
        self._name = name
        self._fn = fn

    def name(self) -> str:
        return self._name

    def run(
        self,
        rows: Sequence[llmfn_output_row.LLMFnOutputRowView],
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        new_values = self._fn(rows)
        if len(new_values) != len(rows):
            raise llmfn_post_process.PostProcessExecutionError(
                'Error executing "{}": returned length ({}) != number of input rows'
                " ({})".format(self._name, len(new_values), len(rows))
            )

        new_rows: list[llmfn_output_row.LLMFnOutputRow] = []
        for new_value, row in zip(new_values, rows):
            new_row = _convert_view_to_output_row(row)
            new_row.add(key=self._name, value=new_value)
            new_rows.append(new_row)

        return new_rows


class LLMFnPostProcessReplaceCommand(LLMFnImplPostProcessCommand):
    """A command that modifies the results in each row."""

    def __init__(self, name: str, fn: llmfn_post_process.LLMFnPostProcessBatchReplaceFn):
        self._name = name
        self._fn = fn

    def name(self) -> str:
        return self._name

    def run(
        self,
        rows: Sequence[llmfn_output_row.LLMFnOutputRowView],
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        new_values = self._fn(rows)
        if len(new_values) != len(rows):
            raise llmfn_post_process.PostProcessExecutionError(
                'Error executing "{}": returned length ({}) != number of input rows'
                " ({})".format(self._name, len(new_values), len(rows))
            )

        new_rows: list[llmfn_output_row.LLMFnOutputRow] = []
        for new_value, row in zip(new_values, rows):
            new_row = _convert_view_to_output_row(row)
            new_row.set_result_value(value=new_value)
            new_rows.append(new_row)

        return new_rows


class LLMCompareFnPostProcessCommand(LLMFnPostProcessCommand):
    """Post-processing commands for LLMCompareFunction."""

    @abc.abstractmethod
    def run(
        self,
        rows: Sequence[
            tuple[
                llmfn_output_row.LLMFnOutputRowView,
                llmfn_output_row.LLMFnOutputRowView,
                llmfn_output_row.LLMFnOutputRowView,
            ]
        ],
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        """Processes a batch of left- and right-hand side results.

        Args:
          rows: The rows in a batch. Each row is a three-tuple containing: - The
            left-hand side results, - The right-hand side results, and - The current
            combined results

        Returns:
          A new set of rows that should replace the combined results.
        """


class LLMCompareFnPostProcessAddCommand(LLMCompareFnPostProcessCommand):
    """A command that adds each row with a new column.

    This does not change the value of the results cell.
    """

    def __init__(
        self,
        name: str,
        fn: llmfn_post_process.LLMCompareFnPostProcessBatchAddFn,
    ):
        self._name = name
        self._fn = fn

    def name(self) -> str:
        return self._name

    def run(
        self,
        rows: Sequence[
            tuple[
                llmfn_output_row.LLMFnOutputRowView,
                llmfn_output_row.LLMFnOutputRowView,
                llmfn_output_row.LLMFnOutputRowView,
            ]
        ],
    ) -> Sequence[llmfn_output_row.LLMFnOutputRow]:
        new_values = self._fn([(lhs, rhs) for lhs, rhs, _ in rows])
        if len(new_values) != len(rows):
            raise llmfn_post_process.PostProcessExecutionError(
                'Error executing "{}": returned length ({}) != number of input rows'
                " ({})".format(self._name, len(new_values), len(rows))
            )

        new_rows: list[llmfn_output_row.LLMFnOutputRow] = []
        for new_value, row in zip(new_values, [combined for _, _, combined in rows]):
            new_row = _convert_view_to_output_row(row)
            new_row.add(key=self._name, value=new_value)
            new_rows.append(new_row)

        return new_rows
