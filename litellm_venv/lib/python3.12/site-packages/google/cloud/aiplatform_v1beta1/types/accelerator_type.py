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
        "AcceleratorType",
    },
)


class AcceleratorType(proto.Enum):
    r"""Represents a hardware accelerator type.

    Values:
        ACCELERATOR_TYPE_UNSPECIFIED (0):
            Unspecified accelerator type, which means no
            accelerator.
        NVIDIA_TESLA_K80 (1):
            Nvidia Tesla K80 GPU.
        NVIDIA_TESLA_P100 (2):
            Nvidia Tesla P100 GPU.
        NVIDIA_TESLA_V100 (3):
            Nvidia Tesla V100 GPU.
        NVIDIA_TESLA_P4 (4):
            Nvidia Tesla P4 GPU.
        NVIDIA_TESLA_T4 (5):
            Nvidia Tesla T4 GPU.
        NVIDIA_TESLA_A100 (8):
            Nvidia Tesla A100 GPU.
        NVIDIA_A100_80GB (9):
            Nvidia A100 80GB GPU.
        NVIDIA_L4 (11):
            Nvidia L4 GPU.
        NVIDIA_H100_80GB (13):
            Nvidia H100 80Gb GPU.
        TPU_V2 (6):
            TPU v2.
        TPU_V3 (7):
            TPU v3.
        TPU_V4_POD (10):
            TPU v4.
        TPU_V5_LITEPOD (12):
            TPU v5.
    """
    ACCELERATOR_TYPE_UNSPECIFIED = 0
    NVIDIA_TESLA_K80 = 1
    NVIDIA_TESLA_P100 = 2
    NVIDIA_TESLA_V100 = 3
    NVIDIA_TESLA_P4 = 4
    NVIDIA_TESLA_T4 = 5
    NVIDIA_TESLA_A100 = 8
    NVIDIA_A100_80GB = 9
    NVIDIA_L4 = 11
    NVIDIA_H100_80GB = 13
    TPU_V2 = 6
    TPU_V3 = 7
    TPU_V4_POD = 10
    TPU_V5_LITEPOD = 12


__all__ = tuple(sorted(__protobuf__.manifest))
