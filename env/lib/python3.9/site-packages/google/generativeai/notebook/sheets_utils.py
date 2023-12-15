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
"""SheetsInputs."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
from urllib import parse
from google.generativeai.notebook import gspread_client
from google.generativeai.notebook import sheets_id
from google.generativeai.notebook.lib import llmfn_inputs_source
from google.generativeai.notebook.lib import llmfn_outputs


def _try_sheet_id_as_url(value: str) -> sheets_id.SheetsIdentifier | None:
    """Try to open a Sheets document with `value` as a URL."""
    try:
        parse_result = parse.urlparse(value)
    except ValueError:
        # If there's a URL parsing error, then it's not a URL.
        return None

    if parse_result.scheme:
        # If it looks like a URL, try to open the document as a URL but don't fall
        # back to trying as key or name since it's very unlikely that a key or name
        # looks like a URL.
        sid = sheets_id.SheetsIdentifier(url=sheets_id.SheetsURL(value))
        gspread_client.get_client().validate(sid)
        return sid

    return None


def _try_sheet_id_as_key(value: str) -> sheets_id.SheetsIdentifier | None:
    """Try to open a Sheets document with `value` as a key."""
    try:
        sid = sheets_id.SheetsIdentifier(key=sheets_id.SheetsKey(value))
    except ValueError:
        # `value` is not a well-formed Sheets key.
        return None

    try:
        gspread_client.get_client().validate(sid)
    except gspread_client.SpreadsheetNotFoundError:
        return None
    return sid


def _try_sheet_id_as_name(value: str) -> sheets_id.SheetsIdentifier | None:
    """Try to open a Sheets document with `value` as a name."""
    sid = sheets_id.SheetsIdentifier(name=value)
    try:
        gspread_client.get_client().validate(sid)
    except gspread_client.SpreadsheetNotFoundError:
        return None
    return sid


def get_sheets_id_from_str(value: str) -> sheets_id.SheetsIdentifier:
    if sid := _try_sheet_id_as_url(value):
        return sid
    if sid := _try_sheet_id_as_key(value):
        return sid
    if sid := _try_sheet_id_as_name(value):
        return sid
    raise RuntimeError('No Sheets found with "{}" as URL, key or name'.format(value))


class SheetsInputs(llmfn_inputs_source.LLMFnInputsSource):
    """Inputs to an LLMFunction from Google Sheets."""

    def __init__(self, sid: sheets_id.SheetsIdentifier, worksheet_id: int = 0):
        super().__init__()
        self._sid = sid
        self._worksheet_id = worksheet_id

    def _to_normalized_inputs_impl(
        self,
    ) -> tuple[Sequence[Mapping[str, str]], Callable[[], None]]:
        return gspread_client.get_client().get_all_records(
            sid=self._sid, worksheet_id=self._worksheet_id
        )


class SheetsOutputs(llmfn_outputs.LLMFnOutputsSink):
    """Writes outputs from an LLMFunction to Google Sheets."""

    def __init__(self, sid: sheets_id.SheetsIdentifier):
        self._sid = sid

    def write_outputs(self, outputs: llmfn_outputs.LLMFnOutputsBase) -> None:
        # Transpose `outputs` into a list of rows.
        outputs_dict = outputs.as_dict()
        outputs_rows: list[Sequence[Any]] = []
        outputs_rows.append(list(outputs_dict.keys()))
        outputs_rows.extend([list(x) for x in zip(*outputs_dict.values())])

        gspread_client.get_client().write_records(
            sid=self._sid,
            rows=outputs_rows,
        )
