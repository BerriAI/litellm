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

from typing import NamedTuple, Optional, Dict, Union

from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    accelerator_type_v1beta1 as gca_accelerator_type_compat,
)


class _ResourcePool(NamedTuple):
    """Specification container for Worker Pool specs used for distributed training.

    Usage:

    resource_pool = _ResourcePool(
                replica_count=1,
                machine_type='n1-standard-4',
                accelerator_count=1,
                accelerator_type='NVIDIA_TESLA_K80',
                boot_disk_type='pd-ssd',
                boot_disk_size_gb=100,
            )

    Note that container and python package specs are not stored with this spec.
    """

    replica_count: int = 1
    machine_type: str = "n1-standard-4"
    accelerator_count: int = 0
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED"
    boot_disk_type: str = "pd-ssd"
    boot_disk_size_gb: int = 100

    def _get_accelerator_type(self) -> Optional[str]:
        """Validates accelerator_type and returns the name of the accelerator.

        Returns:
            None if no accelerator or valid accelerator name.

        Raise:
            ValueError if accelerator type is invalid.
        """

        # Raises ValueError if invalid accelerator_type
        utils.validate_accelerator_type(self.accelerator_type)

        accelerator_enum = getattr(
            gca_accelerator_type_compat.AcceleratorType, self.accelerator_type
        )

        if (
            accelerator_enum
            != gca_accelerator_type_compat.AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED
        ):
            return self.accelerator_type

    @property
    def spec_dict(self) -> Dict[str, Union[int, str, Dict[str, Union[int, str]]]]:
        """Return specification as a Dict."""
        spec = {
            "machine_spec": {"machine_type": self.machine_type},
            "replica_count": self.replica_count,
            "disk_spec": {
                "boot_disk_type": self.boot_disk_type,
                "boot_disk_size_gb": self.boot_disk_size_gb,
            },
        }

        accelerator_type = self._get_accelerator_type()
        if accelerator_type and self.accelerator_count:
            spec["machine_spec"]["accelerator_type"] = accelerator_type
            spec["machine_spec"]["accelerator_count"] = self.accelerator_count

        return spec

    @property
    def is_empty(self) -> bool:
        """Returns True is replica_count > 0 False otherwise."""
        return self.replica_count <= 0
