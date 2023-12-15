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
"""MagicsEngine class."""
from __future__ import annotations

from typing import AbstractSet, Sequence

from google.generativeai.notebook import argument_parser
from google.generativeai.notebook import cmd_line_parser
from google.generativeai.notebook import command
from google.generativeai.notebook import compare_cmd
from google.generativeai.notebook import compile_cmd
from google.generativeai.notebook import eval_cmd
from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import model_registry
from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook import run_cmd
from google.generativeai.notebook.lib import prompt_utils


class MagicsEngine:
    """Implementation of functionality used by Magics.

    This class provides the implementation for Magics, decoupled from the
    details of integrating with Colab Magics such as registration.
    """

    def __init__(
        self,
        registry: model_registry.ModelRegistry | None = None,
        env: ipython_env.IPythonEnv | None = None,
    ):
        self._ipython_env = env
        models = registry or model_registry.ModelRegistry()
        self._cmd_handlers: dict[parsed_args_lib.CommandName, command.Command] = {
            parsed_args_lib.CommandName.RUN_CMD: run_cmd.RunCommand(models=models, env=env),
            parsed_args_lib.CommandName.COMPILE_CMD: compile_cmd.CompileCommand(
                models=models, env=env
            ),
            parsed_args_lib.CommandName.COMPARE_CMD: compare_cmd.CompareCommand(env=env),
            parsed_args_lib.CommandName.EVAL_CMD: eval_cmd.EvalCommand(models=models, env=env),
        }

    def parse_line(
        self,
        line: str,
        placeholders: AbstractSet[str],
    ) -> tuple[parsed_args_lib.ParsedArgs, parsed_args_lib.PostProcessingTokens]:
        return cmd_line_parser.CmdLineParser().parse_line(line, placeholders)

    def _get_handler(
        self, line: str, placeholders: AbstractSet[str]
    ) -> tuple[
        command.Command,
        parsed_args_lib.ParsedArgs,
        Sequence[post_process_utils.ParsedPostProcessExpr],
    ]:
        """Given the command line, parse and return all components.

        Args:
          line: The LLM Magics command line.
          placeholders: Placeholders from prompts in the cell contents.

        Returns:
          A three-tuple containing:
          - The command (e.g. "run")
          - Parsed arguments for the command,
          - Parsed post-processing expressions
        """
        parsed_args, post_processing_tokens = self.parse_line(line, placeholders)
        cmd_name = parsed_args.cmd
        handler = self._cmd_handlers[cmd_name]
        post_processing_fns = handler.parse_post_processing_tokens(post_processing_tokens)
        return handler, parsed_args, post_processing_fns

    def execute_cell(self, line: str, cell_content: str):
        """Executes the supplied magic line and cell payload."""
        cell = _clean_cell(cell_content)
        placeholders = prompt_utils.get_placeholders(cell)

        try:
            handler, parsed_args, post_processing_fns = self._get_handler(line, placeholders)
            return handler.execute(parsed_args, cell, post_processing_fns)
        except argument_parser.ParserNormalExit as e:
            if self._ipython_env is not None:
                e.set_ipython_env(self._ipython_env)
            # ParserNormalExit implements the _ipython_display_ method so it can
            # be returned as the output of this cell for display.
            return e
        except argument_parser.ParserError as e:
            e.display(self._ipython_env)
            # Raise an exception to indicate that execution for this cell has
            # failed.
            # The exception is re-raised as SystemExit because Colab automatically
            # suppresses traceback for SystemExit but not other exceptions. Because
            # ParserErrors are usually due to user error (e.g. a missing required
            # flag or an invalid flag value), we want to hide the traceback to
            # avoid detracting the user from the error message, and we want to
            # reserve exceptions-with-traceback for actual bugs and unexpected
            # errors.
            error_msg = "Got parser error: {}".format(e.msgs()[-1]) if e.msgs() else ""
            raise SystemExit(error_msg) from e


def _clean_cell(cell_content: str) -> str:
    # Colab includes a trailing newline in cell_content. Remove only the last
    # line break from cell contents (i.e. not rstrip), so that multi-line and
    # intentional line breaks are preserved, but single-line prompts don't have
    # a trailing line break.
    cell = cell_content
    if cell.endswith("\n"):
        cell = cell[:-1]
    return cell
