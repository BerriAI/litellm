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
"""IPythonEnvImpl."""
from __future__ import annotations

from typing import Any
from google.generativeai.notebook import ipython_env
from IPython.core import display as ipython_display


class IPythonEnvImpl(ipython_env.IPythonEnv):
    """Concrete implementation of IPythonEnv."""

    def display(self, x: Any) -> None:
        ipython_display.display(x)

    def display_html(self, x: str) -> None:
        ipython_display.display(ipython_display.HTML(x))
