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
        "NotebookEucConfig",
    },
)


class NotebookEucConfig(proto.Message):
    r"""The euc configuration of NotebookRuntimeTemplate.

    Attributes:
        euc_disabled (bool):
            Input only. Whether EUC is disabled in this
            NotebookRuntimeTemplate. In proto3, the default
            value of a boolean is false. In this way, by
            default EUC will be enabled for
            NotebookRuntimeTemplate.
        bypass_actas_check (bool):
            Output only. Whether ActAs check is bypassed
            for service account attached to the VM. If
            false, we need ActAs check for the default
            Compute Engine Service account. When a Runtime
            is created, a VM is allocated using Default
            Compute Engine Service Account. Any user
            requesting to use this Runtime requires Service
            Account User (ActAs) permission over this SA. If
            true, Runtime owner is using EUC and does not
            require the above permission as VM no longer use
            default Compute Engine SA, but a P4SA.
    """

    euc_disabled: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    bypass_actas_check: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
