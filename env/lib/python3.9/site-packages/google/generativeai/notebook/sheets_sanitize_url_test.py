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
"""Unittest for sheets_sanitize_url."""
from __future__ import annotations

from absl.testing import absltest
from google.generativeai.notebook import sheets_sanitize_url


sanitize_sheets_url = sheets_sanitize_url.sanitize_sheets_url


class SheetsSanitizeURLTest(absltest.TestCase):
    def test_scheme_must_be_https(self):
        """The URL must be https://."""
        with self.assertRaisesRegex(
            ValueError, 'Scheme for Sheets url must be "https", got "http"'
        ):
            sanitize_sheets_url("http://docs.google.com")

        # HTTPS goes through.
        url = sanitize_sheets_url("https://docs.google.com")
        self.assertEqual("https://docs.google.com", str(url))

    def test_domain_must_be_docs_google_com(self):
        """Domain must be docs.google.com."""
        with self.assertRaisesRegex(
            ValueError,
            ('Domain for Sheets url must be "docs.google.com", got' ' "sheets.google.com"'),
        ):
            sanitize_sheets_url("https://sheets.google.com")

        # docs.google.com goes through.
        url = sanitize_sheets_url("https://docs.google.com")
        self.assertEqual("https://docs.google.com", str(url))

    def test_params_must_be_docs_google_com(self):
        """Params component must be empty."""
        with self.assertRaisesRegex(ValueError, 'Params component must be empty, got "hello"'):
            sanitize_sheets_url("https://docs.google.com/;hello")

        # URL without params goes through.
        url = sanitize_sheets_url("https://docs.google.com")
        self.assertEqual("https://docs.google.com", str(url))

    def test_path_must_be_limited_character_set(self):
        """Path can only contain a limited character set."""
        with self.assertRaisesRegex(
            ValueError, 'Invalid path for Sheets url, got "/abc/def/sheets.php'
        ):
            sanitize_sheets_url("https://docs.google.com/abc/def/sheets.php")

        # Valid path goes through.
        url = sanitize_sheets_url("https://docs.google.com/abc/DEF/123/-_-")
        self.assertEqual("https://docs.google.com/abc/DEF/123/-_-", str(url))

    def test_query_must_be_limited_character_set(self):
        """Query can only contain a limited character set."""
        with self.assertRaisesRegex(
            ValueError, 'Invalid query for Sheets url, got "a=b&key=sheets.php"'
        ):
            sanitize_sheets_url("https://docs.google.com/?a=b&key=sheets.php")

        # Valid query goes through.
        url = sanitize_sheets_url("https://docs.google.com/?k1=abc&k2=DEF&k3=123&k4=-_-")
        self.assertEqual("https://docs.google.com/?k1=abc&k2=DEF&k3=123&k4=-_-", str(url))

    def test_fragment_must_be_limited_character_set(self):
        """Fragment can only contain a limited character set."""
        with self.assertRaisesRegex(
            ValueError,
            'Invalid fragment for Sheets url, got "a=b&key=sheets.php"',
        ):
            sanitize_sheets_url("https://docs.google.com/#a=b&key=sheets.php")

        # Valid fragment goes through.
        url = sanitize_sheets_url("https://docs.google.com/#k1=abc&k2=DEF&k3=123&k4=-_-")
        self.assertEqual("https://docs.google.com/#k1=abc&k2=DEF&k3=123&k4=-_-", str(url))


if __name__ == "__main__":
    absltest.main()
