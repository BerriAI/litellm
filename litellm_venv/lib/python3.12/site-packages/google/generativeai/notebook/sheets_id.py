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
"""Module for classes related to identifying a Sheets document."""
from __future__ import annotations

import re
from google.generativeai.notebook import sheets_sanitize_url


def _sanitize_key(key: str) -> str:
    if not re.fullmatch("[a-zA-Z0-9_-]+", key):
        raise ValueError('"{}" is not a valid Sheets key'.format(key))
    return key


class SheetsURL:
    """Class that enforces safety by ensuring that URLs are sanitized."""

    def __init__(self, url: str):
        self._url: str = sheets_sanitize_url.sanitize_sheets_url(url)

    def __str__(self) -> str:
        return self._url


class SheetsKey:
    """Class that enforces safety by ensuring that keys are sanitized."""

    def __init__(self, key: str):
        self._key: str = _sanitize_key(key)

    def __str__(self) -> str:
        return self._key


class SheetsIdentifier:
    """Encapsulates a means to identify a Sheets document.

    The gspread library provides three ways to look up a Sheets document: by name,
    by url and by key. An instance of this class represents exactly one of the
    methods.
    """

    def __init__(
        self,
        name: str | None = None,
        key: SheetsKey | None = None,
        url: SheetsURL | None = None,
    ):
        """Constructor.

        Exactly one of the arguments should be provided.

        Args:
          name: The name of the Sheets document. More-than-one Sheets documents can
            have the same name, so this is the least precise method of identifying
            the document.
          key: The key of the Sheets document
          url: The url to the Sheets document

        Raises:
          ValueError: If the caller does not specify exactly one of name, url or
          key.
        """
        self._name = name
        self._key = key
        self._url = url

        # There should be exactly one.
        num_inputs = int(bool(self._name)) + int(bool(self._key)) + int(bool(self._url))
        if num_inputs != 1:
            raise ValueError("Must set exactly one of name, key or url")

    def name(self) -> str | None:
        return self._name

    def key(self) -> SheetsKey | None:
        return self._key

    def url(self) -> SheetsURL | None:
        return self._url

    def __str__(self):
        if self._name:
            return "name={}".format(self._name)
        elif self._key:
            return "key={}".format(self._key)
        else:
            return "url={}".format(self._url)
