# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class SpaceStage(str, Enum):
    """
    Enumeration of possible stage of a Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceStage.BUILDING == "BUILDING"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceInfo.ts#L61 (private url).
    """

    # Copied from moon-landing > server > repo_types > SpaceInfo.ts (private repo)
    NO_APP_FILE = "NO_APP_FILE"
    CONFIG_ERROR = "CONFIG_ERROR"
    BUILDING = "BUILDING"
    BUILD_ERROR = "BUILD_ERROR"
    RUNNING = "RUNNING"
    RUNNING_BUILDING = "RUNNING_BUILDING"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    DELETING = "DELETING"
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"


class SpaceHardware(str, Enum):
    """
    Enumeration of hardwares available to run your Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceHardware.CPU_BASIC == "cpu-basic"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceInfo.ts#L73 (private url).
    """

    CPU_BASIC = "cpu-basic"
    CPU_UPGRADE = "cpu-upgrade"
    T4_SMALL = "t4-small"
    T4_MEDIUM = "t4-medium"
    A10G_SMALL = "a10g-small"
    A10G_LARGE = "a10g-large"
    A100_LARGE = "a100-large"


@dataclass
class SpaceRuntime:
    """
    Contains information about the current runtime of a Space.

    Args:
        stage (`str`):
            Current stage of the space. Example: RUNNING.
        hardware (`str` or `None`):
            Current hardware of the space. Example: "cpu-basic". Can be `None` if Space
            is `BUILDING` for the first time.
        requested_hardware (`str` or `None`):
            Requested hardware. Can be different than `hardware` especially if the request
            has just been made. Example: "t4-medium". Can be `None` if no hardware has
            been requested yet.
        sleep_time (`int` or `None`):
            Number of seconds the Space will be kept alive after the last request. By default (if value is `None`), the
            Space will never go to sleep if it's running on an upgraded hardware, while it will go to sleep after 48
            hours on a free 'cpu-basic' hardware. For more details, see https://huggingface.co/docs/hub/spaces-gpus#sleep-time.
        raw (`dict`):
            Raw response from the server. Contains more information about the Space
            runtime like number of replicas, number of cpu, memory size,...
    """

    stage: SpaceStage
    hardware: Optional[SpaceHardware]
    requested_hardware: Optional[SpaceHardware]
    sleep_time: Optional[int]
    raw: Dict

    def __init__(self, data: Dict) -> None:
        self.stage = data["stage"]
        self.hardware = data["hardware"]["current"]
        self.requested_hardware = data["hardware"]["requested"]
        self.sleep_time = data["gcTimeout"]
        self.raw = data
