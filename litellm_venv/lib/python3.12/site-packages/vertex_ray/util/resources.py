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
import dataclasses
from typing import Dict, List, Optional
from google.cloud.aiplatform_v1beta1.types import PersistentResource


@dataclasses.dataclass
class Resources:
    """Resources for a ray cluster node.

    Attributes:
        machine_type: See the list of machine types:
            https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types
        node_count: This argument represents how many nodes to start for the
            ray cluster.
        accelerator_type: e.g. "NVIDIA_TESLA_P4".
            Vertex AI supports the following types of GPU:
            https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus
        accelerator_count: The number of accelerators to attach to the machine.
        boot_disk_type: Type of the boot disk (default is "pd-ssd").
            Valid values: "pd-ssd" (Persistent Disk Solid State Drive) or
            "pd-standard" (Persistent Disk Hard Disk Drive).
        boot_disk_size_gb: Size in GB of the boot disk (default is 100GB). Must
            be either unspecified or within the range of [100, 64000].
        custom_image: Custom image for this resource (e.g.
            us-docker.pkg.dev/my-project/ray-gpu.2-9.py310-tf:latest).
    """

    machine_type: Optional[str] = "n1-standard-8"
    node_count: Optional[int] = 1
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = 0
    boot_disk_type: Optional[str] = "pd-ssd"
    boot_disk_size_gb: Optional[int] = 100
    custom_image: Optional[str] = None


@dataclasses.dataclass
class NodeImages:
    """
    Custom images for a ray cluster. We currently support Ray v2.4 and python v3.10.
    The custom images must be extended from the following base images:
    "{region}-docker.pkg.dev/vertex-ai/training/ray-cpu.2-4.py310:latest" or
    "{region}-docker.pkg.dev/vertex-ai/training/ray-gpu.2-4.py310:latest". In
    order to use custom images, need to specify both head and worker images.

    Attributes:
        head: image for head node (eg. us-docker.pkg.dev/my-project/ray-cpu.2-9.py310-tf:latest).
        worker: image for all worker nodes (eg. us-docker.pkg.dev/my-project/ray-gpu.2-9.py310-tf:latest).
    """

    head: str = None
    worker: str = None


@dataclasses.dataclass
class Cluster:
    """Ray cluster (output only).

    Attributes:
        cluster_resource_name: It has a format:
            "projects/<project_num>/locations/<region>/persistentResources/<pr_id>".
        network: Virtual private cloud (VPC) network. It has a format:
            "projects/<project_num>/global/networks/<network_name>".
            For Ray Client, VPC peering is required to connect to the cluster
            managed in the Vertex API service. For Ray Job API, VPC network is
            not required because cluster connection can be accessed through
            dashboard address.
        state: Describes the cluster state (defined in PersistentResource.State).
        python_version: Python version for the ray cluster (e.g. "3.10").
        ray_version: Ray version for the ray cluster (e.g. "2.4").
        head_node_type: The head node resource. Resources.node_count must be 1.
            If not set, by default it is a CPU node with machine_type of n1-standard-8.
        worker_node_types: The list of Resources of the worker nodes. Should not
            duplicate the elements in the list.
        dashboard_address: For Ray Job API (JobSubmissionClient), with this
           cluster connection doesn't require VPC peering.
        labels:
            The labels with user-defined metadata to organize Ray cluster.

            Label keys and values can be no longer than 64 characters (Unicode
            codepoints), can only contain lowercase letters, numeric characters,
            underscores and dashes. International characters are allowed.

            See https://goo.gl/xmQnxf for more information and examples of labels.
    """

    cluster_resource_name: str = None
    network: str = None
    state: PersistentResource.State = None
    python_version: str = None
    ray_version: str = None
    head_node_type: Resources = None
    worker_node_types: List[Resources] = None
    dashboard_address: str = None
    labels: Dict[str, str] = None


def _check_machine_spec_identical(
    node_type_1: Resources,
    node_type_2: Resources,
) -> int:
    """Check if node_type_1 and node_type_2 have the same machine_spec.
    If they are identical, return additional_replica_count."""
    additional_replica_count = 0

    # Check if machine_spec are the same
    if (
        node_type_1.machine_type == node_type_2.machine_type
        and node_type_1.accelerator_type == node_type_2.accelerator_type
        and node_type_1.accelerator_count == node_type_2.accelerator_count
    ):
        additional_replica_count = node_type_2.node_count
        return additional_replica_count

    return additional_replica_count
