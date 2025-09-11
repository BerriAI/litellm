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
"""Abstract IPythonEnv base class.

This module provides a layer of abstraction to address the following problems:
1. Sometimes the code needs to run in an environment where IPython is not
available, e.g. inside a unittest.
2. We want to limit dependencies on IPython to code that deals directly with
the notebook environment.
"""
from __future__ import annotations

import abc
from typing import Any


class IPythonEnv(abc.ABC):
    """Abstract base class that provides a wrapper around IPython methods."""

    @abc.abstractmethod
    def display(self, x: Any) -> None:
        """Wrapper around IPython.core.display.display()."""

    @abc.abstractmethod
    def display_html(self, x: str) -> None:
        """Wrapper to display HTML.

        This method is equivalent to calling:
          display.display(display.HTML(x))

        display() and HTML() are combined into a single method because
        display.HTML() returns an object, which would be complicated to model with
        this abstract interface.

        Args:
          x: An HTML string to be displayed.
        """
