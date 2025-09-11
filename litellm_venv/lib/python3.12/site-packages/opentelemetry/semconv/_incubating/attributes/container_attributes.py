# Copyright The OpenTelemetry Authors
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


from enum import Enum

CONTAINER_COMMAND = "container.command"
"""
The command used to run the container (i.e. the command name).
Note: If using embedded credentials or sensitive data, it is recommended to remove them to prevent potential leakage.
"""

CONTAINER_COMMAND_ARGS = "container.command_args"
"""
All the command arguments (including the command/executable itself) run by the container. [2].
"""

CONTAINER_COMMAND_LINE = "container.command_line"
"""
The full command run by the container as a single string representing the full command. [2].
"""

CONTAINER_CPU_STATE = "container.cpu.state"
"""
The CPU state for this data point.
"""

CONTAINER_ID = "container.id"
"""
Container ID. Usually a UUID, as for example used to [identify Docker containers](https://docs.docker.com/engine/reference/run/#container-identification). The UUID might be abbreviated.
"""

CONTAINER_IMAGE_ID = "container.image.id"
"""
Runtime specific image identifier. Usually a hash algorithm followed by a UUID.
Note: Docker defines a sha256 of the image id; `container.image.id` corresponds to the `Image` field from the Docker container inspect [API](https://docs.docker.com/engine/api/v1.43/#tag/Container/operation/ContainerInspect) endpoint.
    K8s defines a link to the container registry repository with digest `"imageID": "registry.azurecr.io /namespace/service/dockerfile@sha256:bdeabd40c3a8a492eaf9e8e44d0ebbb84bac7ee25ac0cf8a7159d25f62555625"`.
    The ID is assinged by the container runtime and can vary in different environments. Consider using `oci.manifest.digest` if it is important to identify the same image in different environments/runtimes.
"""

CONTAINER_IMAGE_NAME = "container.image.name"
"""
Name of the image the container was built on.
"""

CONTAINER_IMAGE_REPO_DIGESTS = "container.image.repo_digests"
"""
Repo digests of the container image as provided by the container runtime.
Note: [Docker](https://docs.docker.com/engine/api/v1.43/#tag/Image/operation/ImageInspect) and [CRI](https://github.com/kubernetes/cri-api/blob/c75ef5b473bbe2d0a4fc92f82235efd665ea8e9f/pkg/apis/runtime/v1/api.proto#L1237-L1238) report those under the `RepoDigests` field.
"""

CONTAINER_IMAGE_TAGS = "container.image.tags"
"""
Container image tags. An example can be found in [Docker Image Inspect](https://docs.docker.com/engine/api/v1.43/#tag/Image/operation/ImageInspect). Should be only the `<tag>` section of the full name for example from `registry.example.com/my-org/my-image:<tag>`.
"""

CONTAINER_LABEL_TEMPLATE = "container.label"
"""
Container labels, `<key>` being the label name, the value being the label value.
"""

CONTAINER_LABELS_TEMPLATE = "container.labels"
"""
Deprecated: Replaced by `container.label`.
"""

CONTAINER_NAME = "container.name"
"""
Container name used by container runtime.
"""

CONTAINER_RUNTIME = "container.runtime"
"""
The container runtime managing this container.
"""


class ContainerCpuStateValues(Enum):
    USER = "user"
    """When tasks of the cgroup are in user mode (Linux). When all container processes are in user mode (Windows)."""
    SYSTEM = "system"
    """When CPU is used by the system (host OS)."""
    KERNEL = "kernel"
    """When tasks of the cgroup are in kernel mode (Linux). When all container processes are in kernel mode (Windows)."""
