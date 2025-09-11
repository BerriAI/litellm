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
"""Command."""
from __future__ import annotations

import abc
import collections
from typing import Sequence

from google.generativeai.notebook import parsed_args_lib
from google.generativeai.notebook import post_process_utils


ProcessingCommand = collections.namedtuple("ProcessingCommand", ["name", "fn"])


class Command(abc.ABC):
    """Base class for implementation of Magics commands like "run"."""

    @abc.abstractmethod
    def execute(
        self,
        parsed_args: parsed_args_lib.ParsedArgs,
        cell_content: str,
        post_processing_fns: Sequence[post_process_utils.ParsedPostProcessExpr],
    ):
        """Executes the command given `parsed_args` and the `cell_content`."""

    @abc.abstractmethod
    def parse_post_processing_tokens(
        self, tokens: Sequence[Sequence[str]]
    ) -> Sequence[post_process_utils.ParsedPostProcessExpr]:
        """Parses post-processing tokens for this command."""
