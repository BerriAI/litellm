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
"""Unittest for html_utils."""
from __future__ import annotations

from absl.testing import absltest
from google.generativeai.notebook import html_utils
from google.generativeai.notebook import sheets_id


class HtmlUtilsTest(absltest.TestCase):
    def test_get_anchor_tag_text_is_escaped(self):
        html = html_utils.get_anchor_tag(
            url=sheets_id.SheetsURL("https://docs.google.com/?a=b#hello"),
            text="hello<evil_tag/>world",
        )
        self.assertEqual(
            (
                '<a target="_blank" rel="noopener"'
                ' href="https://docs.google.com/?a=b#hello">hello&lt;evil_tag/&gt;world</a>'
            ),
            html,
        )

    def test_get_anchor_tag_url_is_escaped(self):
        url = sheets_id.SheetsURL("https://docs.google.com/")
        # Break encapsulation to modify the URL.
        url._url = 'https://docs.google.com/"evil_string"'
        html = html_utils.get_anchor_tag(
            url=url,
            text="hello world",
        )
        self.assertEqual(
            (
                '<a target="_blank" rel="noopener"'
                ' href="https://docs.google.com/&quot;evil_string&quot;">hello'
                " world</a>"
            ),
            html,
        )


if __name__ == "__main__":
    absltest.main()
