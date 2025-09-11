# -*- coding: utf-8 -*-
# Copyright 2020 Google LLC
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

from typing import NamedTuple, Optional, Dict, Union, List

from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    accelerator_type as gca_accelerator_type_compat,
)

# `_SPEC_ORDERS` contains the worker pool spec type and its order in the `_WorkerPoolSpec`.
# The `server_spec` supports either reduction server or parameter server, each
# with different configuration for its `container_spec`. This mapping will be
# used during configuration of `container_spec` for all worker pool specs.
_SPEC_ORDERS = {
    "chief_spec": 0,
    "worker_spec": 1,
    "server_spec": 2,
    "evaluator_spec": 3,
}


class _WorkerPoolSpec(NamedTuple):
    """Specification container for Worker Pool specs used for distributed training.

    Usage:

    spec = _WorkerPoolSpec(
                replica_count=10,
                machine_type='n1-standard-4',
                accelerator_count=2,
                accelerator_type='NVIDIA_TESLA_K80',
                boot_disk_type='pd-ssd',
                boot_disk_size_gb=100,
            )

    Note that container and python package specs are not stored with this spec.
    """

    replica_count: int = 0
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


class _DistributedTrainingSpec(NamedTuple):
    """Configuration for distributed training worker pool specs.

    Vertex AI Training expects configuration in this order:
    [
        chief spec, # can only have one replica
        worker spec,
        parameter server spec,
        evaluator spec
    ]

    Usage:

    dist_training_spec = _DistributedTrainingSpec(
        chief_spec = _WorkerPoolSpec(
                replica_count=1,
                machine_type='n1-standard-4',
                accelerator_count=2,
                accelerator_type='NVIDIA_TESLA_K80',
                boot_disk_type='pd-ssd',
                boot_disk_size_gb=100,
            ),
        worker_spec = _WorkerPoolSpec(
                replica_count=10,
                machine_type='n1-standard-4',
                accelerator_count=2,
                accelerator_type='NVIDIA_TESLA_K80',
                boot_disk_type='pd-ssd',
                boot_disk_size_gb=100,
            ),
    )
    """

    chief_spec: _WorkerPoolSpec = _WorkerPoolSpec()
    worker_spec: _WorkerPoolSpec = _WorkerPoolSpec()
    server_spec: _WorkerPoolSpec = _WorkerPoolSpec()
    evaluator_spec: _WorkerPoolSpec = _WorkerPoolSpec()

    @property
    def pool_specs(
        self,
    ) -> List[Dict[str, Union[int, str, Dict[str, Union[int, str]]]]]:
        """Return each pools spec in correct order for Vertex AI as a list of
        dicts.

        Also removes specs if they are empty but leaves specs in if there unusual
        specifications to not break the ordering in Vertex AI Training.
        ie. 0 chief replica, 10 worker replica, 3 ps replica

        Returns:
            Order list of worker pool specs suitable for Vertex AI Training.
        """
        if self.chief_spec.replica_count > 1:
            raise ValueError("Chief spec replica count cannot be greater than 1.")

        spec_order = [
            self.chief_spec,
            self.worker_spec,
            self.server_spec,
            self.evaluator_spec,
        ]
        specs = [{} if s.is_empty else s.spec_dict for s in spec_order]
        for i in reversed(range(len(spec_order))):
            if spec_order[i].is_empty:
                specs.pop()
            else:
                break
        return specs

    @classmethod
    def chief_worker_pool(
        cls,
        replica_count: int = 0,
        machine_type: str = "n1-standard-4",
        accelerator_count: int = 0,
        accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
        boot_disk_type: str = "pd-ssd",
        boot_disk_size_gb: int = 100,
        reduction_server_replica_count: int = 0,
        reduction_server_machine_type: str = None,
    ) -> "_DistributedTrainingSpec":
        """Parametrizes Config to support only chief with worker replicas.

        For replica is assigned to chief and the remainder to workers. All spec have the
        same machine type, accelerator count, and accelerator type.

        Args:
            replica_count (int):
                The number of worker replicas. Assigns 1 chief replica and
                replica_count - 1 worker replicas.
            machine_type (str):
                The type of machine to use for training.
            accelerator_type (str):
                Hardware accelerator type. One of ACCELERATOR_TYPE_UNSPECIFIED,
                NVIDIA_TESLA_K80, NVIDIA_TESLA_P100, NVIDIA_TESLA_V100, NVIDIA_TESLA_P4,
                NVIDIA_TESLA_T4
            accelerator_count (int):
                The number of accelerators to attach to a worker replica.
            boot_disk_type (str):
                Type of the boot disk (default is `pd-ssd`).
                Valid values: `pd-ssd` (Persistent Disk Solid State Drive) or
                `pd-standard` (Persistent Disk Hard Disk Drive).
            boot_disk_size_gb (int):
                Size in GB of the boot disk (default is 100GB).
                boot disk size must be within the range of [100, 64000].
            reduction_server_replica_count (int):
                The number of reduction server replicas, default is 0.
            reduction_server_machine_type (str):
                The type of machine to use for reduction server, default is `n1-highcpu-16`.

        Returns:
            _DistributedTrainingSpec representing one chief and n workers all of
            same type, optional with reduction server(s). If replica_count <= 0
            then an empty spec is returned.
        """
        if replica_count <= 0:
            return cls()

        chief_spec = _WorkerPoolSpec(
            replica_count=1,
            machine_type=machine_type,
            accelerator_count=accelerator_count,
            accelerator_type=accelerator_type,
            boot_disk_type=boot_disk_type,
            boot_disk_size_gb=boot_disk_size_gb,
        )

        worker_spec = _WorkerPoolSpec(
            replica_count=replica_count - 1,
            machine_type=machine_type,
            accelerator_count=accelerator_count,
            accelerator_type=accelerator_type,
            boot_disk_type=boot_disk_type,
            boot_disk_size_gb=boot_disk_size_gb,
        )

        reduction_server_spec = _WorkerPoolSpec(
            replica_count=reduction_server_replica_count,
            machine_type=reduction_server_machine_type,
        )

        return cls(
            chief_spec=chief_spec,
            worker_spec=worker_spec,
            server_spec=reduction_server_spec,
        )
