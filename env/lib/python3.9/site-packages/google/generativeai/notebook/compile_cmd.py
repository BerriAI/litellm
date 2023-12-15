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
"""The compile command."""
from __future__ import annotations

from typing import Sequence
from google.generativeai.notebook import command
from google.generativeai.notebook import command_utils
from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import model_registry
from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook import py_utils


class CompileCommand(command.Command):
    """Implementation of the "compile" command."""

    def __init__(
        self,
        models: model_registry.ModelRegistry,
        env: ipython_env.IPythonEnv | None = None,
    ):
        """Constructor.

        Args:
          models: ModelRegistry instance.
          env: The IPythonEnv environment.
        """
        super().__init__()
        self._models = models
        self._ipython_env = env

    def execute(
        self,
        parsed_args: parsed_args_lib.ParsedArgs,
        cell_content: str,
        post_processing_fns: Sequence[post_process_utils.ParsedPostProcessExpr],
    ) -> str:
        llm_fn = command_utils.create_llm_function(
            models=self._models,
            env=self._ipython_env,
            parsed_args=parsed_args,
            cell_content=cell_content,
            post_processing_fns=post_processing_fns,
        )

        py_utils.set_py_var(parsed_args.compile_save_name, llm_fn)
        return "Saved function to Python variable: {}".format(parsed_args.compile_save_name)

    def parse_post_processing_tokens(
        self, tokens: Sequence[Sequence[str]]
    ) -> Sequence[post_process_utils.ParsedPostProcessExpr]:
        return post_process_utils.resolve_post_processing_tokens(tokens)
