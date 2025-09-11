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
"""Utilities for generating HTML."""
from __future__ import annotations

from xml.etree import ElementTree
from google.generativeai.notebook import sheets_id


def get_anchor_tag(url: sheets_id.SheetsURL, text: str) -> str:
    """Returns a HTML string representing an anchor tag.

    This class uses the xml.etree library to handle HTML escaping.

    Args:
      url: The Sheets URL to link to.
      text: The text body of the link.

    Returns:
      A string representing a HTML fragment.
    """
    tag = ElementTree.Element(
        "a",
        attrib={
            # Open in a new window/tab
            "target": "_blank",
            # See:
            # https://developer.chrome.com/en/docs/lighthouse/best-practices/external-anchors-use-rel-noopener/
            "rel": "noopener",
            "href": str(url),
        },
    )
    tag.text = text if text else "link"
    return ElementTree.tostring(tag, encoding="unicode", method="html")
