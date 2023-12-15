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
"""Module that holds a global gspread.client.Client."""
from __future__ import annotations

import abc
import datetime
from typing import Any, Callable, Mapping, Sequence
from google.auth import credentials
from google.generativeai.notebook import html_utils
from google.generativeai.notebook import ipython_env
from google.generativeai.notebook import sheets_id


# The code may be running in an environment where the gspread library has not
# been installed.
_gspread_import_error: Exception | None = None
try:
    # pylint: disable-next=g-import-not-at-top
    import gspread
except ImportError as e:
    _gspread_import_error = e
    gspread = None

# Base class of exceptions that  gspread.open(), open_by_url() and open_by_key()
# may throw.
GSpreadException = Exception if gspread is None else gspread.exceptions.GSpreadException  # type: ignore


class SpreadsheetNotFoundError(RuntimeError):
    pass


def _get_import_error() -> Exception:
    return RuntimeError('"gspread" module not imported, got: {}'.format(_gspread_import_error))


class GSpreadClient(abc.ABC):
    """Wrapper around gspread.client.Client.

    This adds a layer of indirection for us to inject mocks for testing.
    """

    @abc.abstractmethod
    def validate(self, sid: sheets_id.SheetsIdentifier) -> None:
        """Validates that `name` is the name of a Google Sheets document.

        Raises an exception if false.

        Args:
          sid: The identifier for the document.
        """

    @abc.abstractmethod
    def get_all_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        worksheet_id: int,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        """Returns all records for a Google Sheets worksheet."""

    @abc.abstractmethod
    def write_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        rows: Sequence[Sequence[Any]],
    ) -> None:
        """Writes results to a new worksheet to the Google Sheets document."""


class GSpreadClientImpl(GSpreadClient):
    """Concrete implementation of GSpreadClient."""

    def __init__(self, client: Any, env: ipython_env.IPythonEnv | None):
        """Constructor.

        Args:
          client: Instance of gspread.client.Client.
          env: Optional instance of IPythonEnv. This is used to display messages
            such as the URL of the output Worksheet.
        """
        self._client = client
        self._ipython_env = env

    def _open(self, sid: sheets_id.SheetsIdentifier):
        """Opens a Sheets document from `sid`.

        Args:
          sid: The identifier for the Sheets document.

        Raises:
          SpreadsheetNotFoundError: If the Sheets document cannot be found or
            cannot be opened.

        Returns:
          A gspread.Worksheet instance representing the worksheet referred to by
          `sid`.
        """
        try:
            if sid.name():
                return self._client.open(sid.name())
            if sid.key():
                return self._client.open_by_key(str(sid.key()))
            if sid.url():
                return self._client.open_by_url(str(sid.url()))
        except GSpreadException as exc:
            raise SpreadsheetNotFoundError("Unable to find Sheets with {}".format(sid)) from exc
        raise SpreadsheetNotFoundError("Invalid sheets_id.SheetsIdentifier")

    def validate(self, sid: sheets_id.SheetsIdentifier) -> None:
        self._open(sid)

    def get_all_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        worksheet_id: int,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        sheet = self._open(sid)
        worksheet = sheet.get_worksheet(worksheet_id)

        if self._ipython_env is not None:
            env = self._ipython_env

            def _display_fn():
                env.display_html(
                    "Reading inputs from worksheet {}".format(
                        html_utils.get_anchor_tag(
                            url=sheets_id.SheetsURL(worksheet.url),
                            text="{} in {}".format(worksheet.title, sheet.title),
                        )
                    )
                )

        else:

            def _display_fn():
                print("Reading inputs from worksheet {} in {}".format(worksheet.title, sheet.title))

        return worksheet.get_all_records(), _display_fn

    def write_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        rows: Sequence[Sequence[Any]],
    ) -> None:
        sheet = self._open(sid)

        # Create a new Worksheet.
        # `title` has to be carefully constructed: some characters like colon ":"
        # will not work with gspread in Worksheet.append_rows().
        current_datetime = datetime.datetime.now()
        title = f"Results {current_datetime:%Y_%m_%d} ({current_datetime:%s})"

        # append_rows() will resize the worksheet as needed, so `rows` and `cols`
        # can be set to 1 to create a worksheet with only a single cell.
        worksheet = sheet.add_worksheet(title=title, rows=1, cols=1)
        worksheet.append_rows(values=rows)

        if self._ipython_env is not None:
            self._ipython_env.display_html(
                "Results written to new worksheet {}".format(
                    html_utils.get_anchor_tag(
                        url=sheets_id.SheetsURL(worksheet.url),
                        text="{} in {}".format(worksheet.title, sheet.title),
                    )
                )
            )
        else:
            print("Results written to new worksheet {} in {}".format(worksheet.title, sheet.title))


class NullGSpreadClient(GSpreadClient):
    """Null-object implementation of GSpreadClient.

    This class raises an error if any of its methods are called. It is used when
    the gspread library is not available.
    """

    def validate(self, sid: sheets_id.SheetsIdentifier) -> None:
        raise _get_import_error()

    def get_all_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        worksheet_id: int,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        raise _get_import_error()

    def write_records(
        self,
        sid: sheets_id.SheetsIdentifier,
        rows: Sequence[Sequence[Any]],
    ) -> None:
        raise _get_import_error()


# Global instance of gspread client.
_gspread_client: GSpreadClient | None = None


def authorize(creds: credentials.Credentials, env: ipython_env.IPythonEnv | None) -> None:
    """Sets up credential for gspreads."""
    global _gspread_client
    if gspread is not None:
        client = gspread.authorize(creds)  # type: ignore
        _gspread_client = GSpreadClientImpl(client=client, env=env)
    else:
        _gspread_client = NullGSpreadClient()


def get_client() -> GSpreadClient:
    if not _gspread_client:
        raise RuntimeError("Must call authorize() first")
    return _gspread_client


def testonly_set_client(client: GSpreadClient) -> None:
    """Overrides the global client for testing."""
    global _gspread_client
    _gspread_client = client
