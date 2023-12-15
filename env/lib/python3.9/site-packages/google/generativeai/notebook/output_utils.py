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
"""Utilities for exporting outputs from LLMFunctions."""
from __future__ import annotations

import copy

from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import py_utils
from google.generativeai.notebook.lib import llmfn_outputs


class _PyVarOutputsSink(llmfn_outputs.LLMFnOutputsSink):
    """Sink that writes results to a Python variable."""

    def __init__(self, var_name: str):
        self._var_name = var_name

    def write_outputs(self, outputs: llmfn_outputs.LLMFnOutputsBase) -> None:
        # Clone our results so that they are all independent.
        py_utils.set_py_var(self._var_name, copy.deepcopy(outputs))


def get_outputs_sink_from_py_var(
    var_name: str,
) -> llmfn_outputs.LLMFnOutputsSink:
    # The output variable `var_name` will be created if it does not already
    # exist.
    if py_utils.has_py_var(var_name):
        data = py_utils.get_py_var(var_name)
        if isinstance(data, llmfn_outputs.LLMFnOutputsSink):
            return data
    return _PyVarOutputsSink(var_name)


def write_to_outputs(
    results: llmfn_outputs.LLMFnOutputs,
    parsed_args: parsed_args_lib.ParsedArgs,
) -> None:
    """Writes `results` to the sinks provided.

    Args:
      results: The results to export.
      parsed_args: Arguments parsed from the command line.
    """
    for sink in parsed_args.outputs:
        results.export(sink)
    for sink in parsed_args.sheets_output_names:
        results.export(sink)
