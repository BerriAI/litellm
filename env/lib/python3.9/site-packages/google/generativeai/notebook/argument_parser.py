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
"""Customized ArgumentParser.

The default behvaior of argparse.ArgumentParser's parse_args() method  is to
exit with a SystemExit exception in the following cases:
1. When the user requests a help message (with the --help or -h flags), or
2. When there's a parsing error (e.g. missing required flags or mistyped flags)

To make the errors more user-friendly, this class customizes
argparse.ArgumentParser and raises either ParserNormalExit for (1) or
ParserError for (2); this way the caller has control over how to display them
to the user.
"""
from __future__ import annotations

import abc
import argparse
from typing import Sequence
from google.generativeai.notebook import ipython_env


# pylint: disable-next=g-bad-exception-name
class _ParserBaseException(RuntimeError, metaclass=abc.ABCMeta):
    """Base class for parser exceptions including normal exit."""

    def __init__(self, msgs: Sequence[str], *args, **kwargs):
        super().__init__("".join(msgs), *args, **kwargs)
        self._msgs = msgs
        self._ipython_env: ipython_env.IPythonEnv | None = None

    def set_ipython_env(self, env: ipython_env.IPythonEnv) -> None:
        self._ipython_env = env

    def _ipython_display_(self):
        self.display(self._ipython_env)

    def msgs(self) -> Sequence[str]:
        return self._msgs

    @abc.abstractmethod
    def display(self, env: ipython_env.IPythonEnv | None) -> None:
        """Display this exception on an IPython console."""


# ParserNormalExit is not an error: it's a way for ArgumentParser to indicate
# that the user has entered a special request (e.g. "--help") instead of a
# runnable command.
# pylint: disable-next=g-bad-exception-name
class ParserNormalExit(_ParserBaseException):
    """Exception thrown when the parser exits normally.

    This is usually thrown when the user requests the help message.
    """

    def display(self, env: ipython_env.IPythonEnv | None) -> None:
        for msg in self._msgs:
            print(msg)


class ParserError(_ParserBaseException):
    """Exception thrown when there is an error."""

    def display(self, env: ipython_env.IPythonEnv | None) -> None:
        for msg in self._msgs:
            print(msg)
        if env is not None:
            # Highlight to the user that an error has occurred.
            env.display_html("<b style='font-family:courier new'>ERROR</b>")


class ArgumentParser(argparse.ArgumentParser):
    """Customized ArgumentParser for LLM Magics.

    This class overrides the parent argparse.ArgumentParser's error-handling
    methods to avoid side-effects like printing to stderr. The messages are
    accumulated and passed into the raised exceptions for the caller to
    handle them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._messages: list[str] = []

    def _print_message(self, message, file=None):
        """Override ArgumentParser's _print_message() method."""
        del file
        self._messages.append(message)

    def exit(self, status=0, message=None):
        """Override ArgumentParser's exit() method."""
        if message:
            self._print_message(message)

        msgs = self._messages
        self._messages = []
        if status == 0:
            raise ParserNormalExit(msgs=msgs)
        else:
            raise ParserError(msgs=msgs)
