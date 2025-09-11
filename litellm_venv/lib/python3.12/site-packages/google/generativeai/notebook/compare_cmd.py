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
"""The compare command."""
from __future__ import annotations

from typing import Sequence

from google.generativeai.notebook import command
from google.generativeai.notebook import command_utils
from google.generativeai.notebook import input_utils
from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import output_utils
from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils
import pandas


class CompareCommand(command.Command):
    """Implementation of "compare" command."""

    def __init__(
        self,
        env: ipython_env.IPythonEnv | None = None,
    ):
        """Constructor.

        Args:
          env: The IPythonEnv environment.
        """
        super().__init__()
        self._ipython_env = env

    def execute(
        self,
        parsed_args: parsed_args_lib.ParsedArgs,
        cell_content: str,
        post_processing_fns: Sequence[post_process_utils.ParsedPostProcessExpr],
    ) -> pandas.DataFrame:
        # We expect CmdLineParser to have already read the inputs once to validate
        # that the placeholders in the prompt are present in the inputs, so we can
        # suppress the status messages here.
        inputs = input_utils.join_inputs_sources(parsed_args, suppress_status_msgs=True)

        llm_cmp_fn = command_utils.create_llm_compare_function(
            env=self._ipython_env,
            parsed_args=parsed_args,
            post_processing_fns=post_processing_fns,
        )

        results = llm_cmp_fn(inputs=inputs)
        output_utils.write_to_outputs(results=results, parsed_args=parsed_args)
        return results.as_pandas_dataframe()

    def parse_post_processing_tokens(
        self, tokens: Sequence[Sequence[str]]
    ) -> Sequence[post_process_utils.ParsedPostProcessExpr]:
        if tokens:
            raise RuntimeError('Post-processing is not supported by "compare"')
        return []
