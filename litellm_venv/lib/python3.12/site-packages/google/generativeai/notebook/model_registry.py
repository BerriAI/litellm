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
"""Maintains set of LLM models that can be instantiated by name."""
from __future__ import annotations

import enum
from typing import Callable

from google.generativeai.notebook import text_model
from google.generativeai.notebook.lib import model as model_lib


class ModelName(enum.Enum):
    ECHO_MODEL = "echo"
    TEXT_MODEL = "text"


class ModelRegistry:
    """Registry that instantiates and caches models."""

    DEFAULT_MODEL = ModelName.TEXT_MODEL

    def __init__(self):
        self._model_cache: dict[ModelName, model_lib.AbstractModel] = {}
        self._model_constructors: dict[
            ModelName, Callable[[], model_lib.AbstractModel]
        ] = {
            ModelName.ECHO_MODEL: model_lib.EchoModel,
            ModelName.TEXT_MODEL: text_model.TextModel,
        }

    def get_model(self, model_name: ModelName) -> model_lib.AbstractModel:
        """Given `model_name`, return the corresponding Model instance.

        Model instances are cached and reused for the same `model_name`.

        Args:
          model_name: The name of the model.

        Returns:
          The corresponding model instance for `model_name`.
        """
        if model_name not in self._model_cache:
            self._model_cache[model_name] = self._model_constructors[model_name]()
        return self._model_cache[model_name]
