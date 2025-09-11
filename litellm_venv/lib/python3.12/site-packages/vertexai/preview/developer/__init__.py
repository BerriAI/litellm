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
#

from vertexai.preview._workflow.serialization_engine import (
    any_serializer,
)
from vertexai.preview._workflow.serialization_engine import (
    serializers_base,
)
from vertexai.preview._workflow.shared import configs
from vertexai.preview.developer import mark
from vertexai.preview.developer import remote_specs


PersistentResourceConfig = configs.PersistentResourceConfig
Serializer = serializers_base.Serializer
SerializationMetadata = serializers_base.SerializationMetadata
SerializerArgs = serializers_base.SerializerArgs
RemoteConfig = configs.RemoteConfig
WorkerPoolSpec = remote_specs.WorkerPoolSpec
WorkerPoolSepcs = remote_specs.WorkerPoolSpecs

register_serializer = any_serializer.register_serializer


__all__ = (
    "mark",
    "PersistentResourceConfig",
    "register_serializer",
    "Serializer",
    "SerializerArgs",
    "SerializationMetadata",
    "RemoteConfig",
    "WorkerPoolSpec",
    "WorkerPoolSepcs",
)
