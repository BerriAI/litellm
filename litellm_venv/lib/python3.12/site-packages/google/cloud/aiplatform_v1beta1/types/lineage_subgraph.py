# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.cloud.aiplatform_v1beta1.types import artifact
from google.cloud.aiplatform_v1beta1.types import event
from google.cloud.aiplatform_v1beta1.types import execution


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "LineageSubgraph",
    },
)


class LineageSubgraph(proto.Message):
    r"""A subgraph of the overall lineage graph. Event edges connect
    Artifact and Execution nodes.

    Attributes:
        artifacts (MutableSequence[google.cloud.aiplatform_v1beta1.types.Artifact]):
            The Artifact nodes in the subgraph.
        executions (MutableSequence[google.cloud.aiplatform_v1beta1.types.Execution]):
            The Execution nodes in the subgraph.
        events (MutableSequence[google.cloud.aiplatform_v1beta1.types.Event]):
            The Event edges between Artifacts and
            Executions in the subgraph.
    """

    artifacts: MutableSequence[artifact.Artifact] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=artifact.Artifact,
    )
    executions: MutableSequence[execution.Execution] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=execution.Execution,
    )
    events: MutableSequence[event.Event] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=event.Event,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
