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

import datetime
import logging
import time
from typing import Optional

from google.api_core import exceptions
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.utils import (
    PersistentResourceClientWithOverride,
)
from google.cloud.aiplatform.preview.vertex_ray.util import _validation_utils
from google.cloud.aiplatform.preview.vertex_ray.util.resources import (
    Cluster,
    Resources,
)
from google.cloud.aiplatform_v1beta1.types.persistent_resource import (
    PersistentResource,
)
from google.cloud.aiplatform_v1beta1.types.persistent_resource_service import (
    GetPersistentResourceRequest,
)


_PRIVATE_PREVIEW_IMAGE = "-docker.pkg.dev/vertex-ai/training/tf-"
_OFFICIAL_IMAGE = "-docker.pkg.dev/vertex-ai/training/ray-"


def create_persistent_resource_client():
    # location is inhereted from the global configuration at aiplatform.init().
    return initializer.global_config.create_client(
        client_class=PersistentResourceClientWithOverride,
        appended_gapic_version="vertex_ray",
    ).select_version("v1beta1")


def polling_delay(num_attempts: int, time_scale: float) -> datetime.timedelta:
    """Computes a delay to the next attempt to poll the Vertex service.

    This does bounded exponential backoff, starting with $time_scale.
    If $time_scale == 0, it starts with a small time interval, less than
    1 second.

    Args:
      num_attempts: The number of times have we polled and found that the
        desired result was not yet available.
      time_scale: The shortest polling interval, in seconds, or zero. Zero is
        treated as a small interval, less than 1 second.

    Returns:
      A recommended delay interval, in seconds.
    """
    #  The polling schedule is slow initially , and then gets faster until 6
    #  attempts (after that the sleeping time remains the same).
    small_interval = 30.0  # Seconds
    interval = max(time_scale, small_interval) * 0.765 ** min(num_attempts, 6)
    return datetime.timedelta(seconds=interval)


def get_persistent_resource(
    persistent_resource_name: str, tolerance: Optional[int] = 0
):
    """Get persistent resource.

    Args:
      persistent_resource_name:
          "projects/<project_num>/locations/<region>/persistentResources/<pr_id>".
      tolerance: number of attemps to get persistent resource.

    Returns:
      aiplatform_v1beta1.PersistentResource if state is RUNNING.

    Raises:
      ValueError: Invalid cluster resource name.
      RuntimeError: Service returns error.
      RuntimeError: Cluster resource state is STOPPING.
      RuntimeError: Cluster resource state is ERROR.
    """

    client = create_persistent_resource_client()
    request = GetPersistentResourceRequest(name=persistent_resource_name)

    # TODO(b/277117901): Add test cases for polling and error handling
    num_attempts = 0
    while True:
        try:
            response = client.get_persistent_resource(request)
        except exceptions.NotFound:
            response = None
            if num_attempts >= tolerance:
                raise ValueError(
                    "[Ray on Vertex AI]: Invalid cluster_resource_name (404 not found)."
                )
        if response:
            if response.error.message:
                logging.error("[Ray on Vertex AI]: %s" % response.error.message)
                raise RuntimeError("[Ray on Vertex AI]: Cluster returned an error.")

            print("[Ray on Vertex AI]: Cluster State =", response.state)
            if response.state == PersistentResource.State.RUNNING:
                return response
            elif response.state == PersistentResource.State.STOPPING:
                raise RuntimeError("[Ray on Vertex AI]: The cluster is stopping.")
            elif response.state == PersistentResource.State.ERROR:
                raise RuntimeError(
                    "[Ray on Vertex AI]: The cluster encountered an error."
                )
        # Polling decay
        sleep_time = polling_delay(num_attempts=num_attempts, time_scale=150.0)
        num_attempts += 1
        print(
            "Waiting for cluster provisioning; attempt {}; sleeping for {} seconds".format(
                num_attempts, sleep_time
            )
        )
        time.sleep(sleep_time.total_seconds())


def persistent_resource_to_cluster(
    persistent_resource: PersistentResource,
) -> Optional[Cluster]:
    """Format a PersistentResource to a dictionary.

    Args:
        persistent_resource: PersistentResource.
    Returns:
        Cluster.
    """
    dashboard_address = persistent_resource.resource_runtime.access_uris.get(
        "RAY_DASHBOARD_URI"
    )
    cluster = Cluster(
        cluster_resource_name=persistent_resource.name,
        network=persistent_resource.network,
        state=persistent_resource.state.name,
        labels=persistent_resource.labels,
        dashboard_address=dashboard_address,
    )
    if not persistent_resource.resource_runtime_spec.ray_spec:
        # skip PersistentResource without RaySpec
        logging.info(
            "[Ray on Vertex AI]: Cluster %s does not have Ray installed."
            % persistent_resource.name,
        )
        return
    resource_pools = persistent_resource.resource_pools

    head_resource_pool = resource_pools[0]
    head_id = head_resource_pool.id
    head_image_uri = (
        persistent_resource.resource_runtime_spec.ray_spec.resource_pool_images[head_id]
    )

    if not head_image_uri:
        head_image_uri = persistent_resource.resource_runtime_spec.ray_spec.image_uri

    try:
        python_version, ray_version = _validation_utils.get_versions_from_image_uri(
            head_image_uri
        )
    except IndexError:
        if _PRIVATE_PREVIEW_IMAGE in head_image_uri:
            # If using outdated images
            logging.info(
                "[Ray on Vertex AI]: The image of cluster %s is outdated."
                " It is recommended to delete and recreate the cluster to obtain"
                " the latest image." % persistent_resource.name
            )
            return None
        else:
            # Custom image might also cause IndexError
            python_version = None
            ray_version = None
    cluster.python_version = python_version
    cluster.ray_version = ray_version

    accelerator_type = head_resource_pool.machine_spec.accelerator_type
    if accelerator_type.value != 0:
        accelerator_type = accelerator_type.name
    else:
        accelerator_type = None
    if _OFFICIAL_IMAGE in head_image_uri:
        # Official training image is not custom
        head_image_uri = None
    head_node_type = Resources(
        machine_type=head_resource_pool.machine_spec.machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=head_resource_pool.machine_spec.accelerator_count,
        boot_disk_type=head_resource_pool.disk_spec.boot_disk_type,
        boot_disk_size_gb=head_resource_pool.disk_spec.boot_disk_size_gb,
        node_count=1,
        custom_image=head_image_uri,
    )
    worker_node_types = []
    if head_resource_pool.replica_count > 1:
        # head_node_type.node_count must be 1. If the head_resource_pool (the first
        # resource pool) has replica_count > 1, the rest replica are worker nodes.
        worker_node_count = head_resource_pool.replica_count - 1
        worker_node_types.append(
            Resources(
                machine_type=head_resource_pool.machine_spec.machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=head_resource_pool.machine_spec.accelerator_count,
                boot_disk_type=head_resource_pool.disk_spec.boot_disk_type,
                boot_disk_size_gb=head_resource_pool.disk_spec.boot_disk_size_gb,
                node_count=worker_node_count,
                custom_image=head_image_uri,
            )
        )
    for i in range(len(resource_pools) - 1):
        # Convert the second and more resource pools to vertex_ray.Resources,
        # and append then to worker_node_types.
        accelerator_type = resource_pools[i + 1].machine_spec.accelerator_type
        if accelerator_type.value != 0:
            accelerator_type = accelerator_type.name
        else:
            accelerator_type = None
        worker_image_uri = (
            persistent_resource.resource_runtime_spec.ray_spec.resource_pool_images[
                resource_pools[i + 1].id
            ]
        )
        if _OFFICIAL_IMAGE in worker_image_uri:
            # Official training image is not custom
            worker_image_uri = None
        worker_node_types.append(
            Resources(
                machine_type=resource_pools[i + 1].machine_spec.machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=resource_pools[i + 1].machine_spec.accelerator_count,
                boot_disk_type=resource_pools[i + 1].disk_spec.boot_disk_type,
                boot_disk_size_gb=resource_pools[i + 1].disk_spec.boot_disk_size_gb,
                node_count=resource_pools[i + 1].replica_count,
                custom_image=worker_image_uri,
            )
        )

    cluster.head_node_type = head_node_type
    cluster.worker_node_types = worker_node_types

    return cluster
