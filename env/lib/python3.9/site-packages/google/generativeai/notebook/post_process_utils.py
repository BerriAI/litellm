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
"""Utilities for working with post-processing tokens."""
from __future__ import annotations

import abc
from typing import Any, Callable, Sequence

from google.generativeai.notebook import py_utils
from google.generativeai.notebook.lib import llm_function
from google.generativeai.notebook.lib import llmfn_output_row
from google.generativeai.notebook.lib import llmfn_post_process


class PostProcessParseError(RuntimeError):
    """An error parsing the post-processing tokens."""


class ParsedPostProcessExpr(abc.ABC):
    """A post-processing expression parsed from the command line."""

    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of this expression."""

    @abc.abstractmethod
    def add_to_llm_function(self, llm_fn: llm_function.LLMFunction) -> llm_function.LLMFunction:
        """Adds this parsed expression to `llm_fn` as a post-processing command."""


class _ParsedPostProcessAddExpr(
    ParsedPostProcessExpr, llmfn_post_process.LLMFnPostProcessBatchAddFn
):
    """An expression that returns the value of a new column to add to a row."""

    def __init__(self, name: str, fn: Callable[[str], Any]):
        """Constructor.

        Args:
          name: The name of the expression. The name of the new column will be
            derived from this.
          fn: A function that takes the result of a row and returns a new value to
            add as a new column in the row.
        """
        self._name = name
        self._fn = fn

    def name(self) -> str:
        return self._name

    def __call__(self, rows: Sequence[llmfn_output_row.LLMFnOutputRowView]) -> Sequence[Any]:
        return [self._fn(row.result_value()) for row in rows]

    def add_to_llm_function(self, llm_fn: llm_function.LLMFunction) -> llm_function.LLMFunction:
        return llm_fn.add_post_process_add_fn(name=self._name, fn=self)


class _ParsedPostProcessReplaceExpr(
    ParsedPostProcessExpr, llmfn_post_process.LLMFnPostProcessBatchReplaceFn
):
    """An expression that returns the new result value for a row."""

    def __init__(self, name: str, fn: Callable[[str], str]):
        """Constructor.

        Args:
          name: The name of the expression.
          fn: A function that takes the result of a row and returns the new result.
        """
        self._name = name
        self._fn = fn

    def name(self) -> str:
        return self._name

    def __call__(self, rows: Sequence[llmfn_output_row.LLMFnOutputRowView]) -> Sequence[str]:
        return [self._fn(row.result_value()) for row in rows]

    def add_to_llm_function(self, llm_fn: llm_function.LLMFunction) -> llm_function.LLMFunction:
        return llm_fn.add_post_process_replace_fn(name=self._name, fn=self)


# Decorator functions.
def post_process_add_fn(fn: Callable[[str], Any]):
    return _ParsedPostProcessAddExpr(name=fn.__name__, fn=fn)


def post_process_replace_fn(fn: Callable[[str], str]):
    return _ParsedPostProcessReplaceExpr(name=fn.__name__, fn=fn)


def validate_one_post_processing_expression(
    tokens: Sequence[str],
) -> None:
    if not tokens:
        raise PostProcessParseError("Cannot have empty post-processing expression")
    if len(tokens) > 1:
        raise PostProcessParseError("Post-processing expression should be a single token")


def _resolve_one_post_processing_expression(
    tokens: Sequence[str],
) -> tuple[str, Any]:
    """Returns name and the resolved expression."""
    validate_one_post_processing_expression(tokens)

    token_parts = tokens[0].split(".")

    current_module = py_utils.get_main_module()
    for part_num, part in enumerate(token_parts):
        current_module_vars = vars(current_module)
        if part not in current_module_vars:
            raise PostProcessParseError(
                'Unable to resolve "{}"'.format(".".join(token_parts[: part_num + 1]))
            )

        current_module = current_module_vars[part]

    return (" ".join(tokens), current_module)


def resolve_post_processing_tokens(
    tokens: Sequence[Sequence[str]],
) -> Sequence[ParsedPostProcessExpr]:
    """Resolves post-processing tokens into ParsedPostProcessExprs.

    E.g. Given [["add_length"], ["to_upper"]] as input, this function will return
    a sequence of ParsedPostProcessExprs that will execute add_length() and
    to_upper() on each entry of the LLM output as post-processing operations.

    Raises:
      PostProcessParseError: An error parsing or resolving the tokens.

    Args:
      tokens: A sequence of post-processing tokens after splitting.

    Returns:
      A sequence of ParsedPostProcessExprs.
    """
    results: list[ParsedPostProcessExpr] = []
    for expression in tokens:
        expr_name, expr_value = _resolve_one_post_processing_expression(expression)
        if isinstance(expr_value, ParsedPostProcessExpr):
            results.append(expr_value)
        elif isinstance(expr_value, Callable):
            # By default, assume that an undecorated function is an "add" function.
            results.append(_ParsedPostProcessAddExpr(name=expr_name, fn=expr_value))
        else:
            raise PostProcessParseError("{} is not callable".format(expr_name))

    return results
