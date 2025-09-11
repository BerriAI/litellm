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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "NetworkSpec",
    },
)


class NetworkSpec(proto.Message):
    r"""Network spec.

    Attributes:
        enable_internet_access (bool):
            Whether to enable public internet access.
            Default false.
        network (str):
            The full name of the Google Compute Engine
            `network <https://cloud.google.com//compute/docs/networks-and-firewalls#networks>`__
        subnetwork (str):
            The name of the subnet that this instance is in. Format:
            ``projects/{project_id_or_number}/regions/{region}/subnetworks/{subnetwork_id}``
    """

    enable_internet_access: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    network: str = proto.Field(
        proto.STRING,
        number=2,
    )
    subnetwork: str = proto.Field(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
