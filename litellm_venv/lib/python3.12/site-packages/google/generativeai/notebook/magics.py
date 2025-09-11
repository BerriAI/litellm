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
"""Colab Magics class.

Installs %%llm magics.
"""
from __future__ import annotations

import abc

from google.auth import credentials
from google.generativeai import client as genai
from google.generativeai.notebook import gspread_client
from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import ipython_env_impl
from google.generativeai.notebook import magics_engine
from google.generativeai.notebook import post_process_utils
from google.generativeai.notebook import sheets_utils

import IPython
from IPython.core import magic


# Set the UA to distinguish the magic from the client. Do this at import-time
# so that a user can still call `genai.configure()`, and both their settings
# and this are honored.
genai.USER_AGENT = "genai-py-magic"

SheetsInputs = sheets_utils.SheetsInputs
SheetsOutputs = sheets_utils.SheetsOutputs

# Decorator functions for post-processing.
post_process_add_fn = post_process_utils.post_process_add_fn
post_process_replace_fn = post_process_utils.post_process_replace_fn

# Globals.
_ipython_env: ipython_env.IPythonEnv | None = None


def _get_ipython_env() -> ipython_env.IPythonEnv:
    """Lazily constructs and returns a global IPythonEnv instance."""
    global _ipython_env
    if _ipython_env is None:
        _ipython_env = ipython_env_impl.IPythonEnvImpl()
    return _ipython_env


def authorize(creds: credentials.Credentials) -> None:
    """Sets up credentials.

    This is used for interacting Google APIs, such as Google Sheets.

    Args:
      creds: The credentials that will be used (e.g. to read from Google Sheets.)
    """
    gspread_client.authorize(creds=creds, env=_get_ipython_env())


class AbstractMagics(abc.ABC):
    """Defines interface to Magics class."""

    @abc.abstractmethod
    def llm(self, cell_line: str | None, cell_body: str | None):
        """Perform various LLM-related operations.

        Args:
          cell_line: String to pass to the MagicsEngine.
          cell_body: Contents of the cell body.
        """
        raise NotImplementedError()


class MagicsImpl(AbstractMagics):
    """Actual class implementing the magics functionality.

    We use a separate class to ensure a single, global instance
    of the magics class.
    """

    def __init__(self):
        self._engine = magics_engine.MagicsEngine(env=_get_ipython_env())

    def llm(self, cell_line: str | None, cell_body: str | None):
        """Perform various LLM-related operations.

        Args:
          cell_line: String to pass to the MagicsEngine.
          cell_body: Contents of the cell body.

        Returns:
          Results from running MagicsEngine.
        """
        cell_line = cell_line or ""
        cell_body = cell_body or ""
        return self._engine.execute_cell(cell_line, cell_body)


@magic.magics_class
class Magics(magic.Magics):
    """Class to register the magic with Colab.

    Objects of this class delegate all calls to a single,
    global instance.
    """

    # Global instance
    _instance = None

    @classmethod
    def get_instance(cls) -> AbstractMagics:
        """Retrieve global instance of the Magics object."""
        if cls._instance is None:
            cls._instance = MagicsImpl()
        return cls._instance

    @magic.line_cell_magic
    def llm(self, cell_line: str | None, cell_body: str | None):
        """Perform various LLM-related operations.

        Args:
          cell_line: String to pass to the MagicsEngine.
          cell_body: Contents of the cell body.

        Returns:
          Results from running MagicsEngine.
        """
        return Magics.get_instance().llm(cell_line=cell_line, cell_body=cell_body)

    @magic.line_cell_magic
    def palm(self, cell_line: str | None, cell_body: str | None):
        return self.llm(cell_line, cell_body)

    @magic.line_cell_magic
    def gemini(self, cell_line: str | None, cell_body: str | None):
        return self.llm(cell_line, cell_body)


IPython.get_ipython().register_magics(Magics)
