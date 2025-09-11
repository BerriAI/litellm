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

import copy
import logging
import time
from typing import Dict, List, Optional

from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.utils import resource_manager_utils
from google.cloud.aiplatform_v1beta1.types import persistent_resource_service

from google.cloud.aiplatform_v1beta1.types.persistent_resource import (
    PersistentResource,
    RaySpec,
    RayMetricSpec,
    ResourcePool,
    ResourceRuntimeSpec,
)

from google.cloud.aiplatform.preview.vertex_ray.util import (
    _gapic_utils,
    _validation_utils,
    resources,
)

from google.protobuf import field_mask_pb2  # type: ignore


def create_ray_cluster(
    head_node_type: Optional[resources.Resources] = resources.Resources(),
    python_version: Optional[str] = "3.10",
    ray_version: Optional[str] = "2.9",
    network: Optional[str] = None,
    cluster_name: Optional[str] = None,
    worker_node_types: Optional[List[resources.Resources]] = None,
    custom_images: Optional[resources.NodeImages] = None,
    enable_metrics_collection: Optional[bool] = True,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Create a ray cluster on the Vertex AI.

    Sample usage:

    from vertex_ray import Resources

    head_node_type = Resources(
        machine_type="n1-standard-8",
        node_count=1,
        accelerator_type="NVIDIA_TESLA_K80",
        accelerator_count=1,
        custom_image="us-docker.pkg.dev/my-project/ray-cpu-image.2.9:latest",  # Optional
    )

    worker_node_types = [Resources(
        machine_type="n1-standard-8",
        node_count=2,
        accelerator_type="NVIDIA_TESLA_K80",
        accelerator_count=1,
        custom_image="us-docker.pkg.dev/my-project/ray-gpu-image.2.9:latest",  # Optional
    )]

    cluster_resource_name = vertex_ray.create_ray_cluster(
        head_node_type=head_node_type,
        network="projects/my-project-number/global/networks/my-vpc-name",
        worker_node_types=worker_node_types,
        ray_version="2.9",
    )

    After a ray cluster is set up, you can call
    `ray.init(f"vertex_ray://{cluster_resource_name}", runtime_env=...)` without
    specifying ray cluster address to connect to the cluster. To shut down the
    cluster you can call `ray.delete_ray_cluster()`.
    Note: If the active ray cluster has not finished shutting down, you cannot
    create a new ray cluster with the same cluster_name.

    Args:
        head_node_type: The head node resource. Resources.node_count must be 1.
            If not set, default value of Resources() class will be used.
        python_version: Python version for the ray cluster.
        ray_version: Ray version for the ray cluster.
        network: Virtual private cloud (VPC) network. For Ray Client, VPC
            peering is required to connect to the Ray Cluster managed in the
            Vertex API service. For Ray Job API, VPC network is not required
            because Ray Cluster connection can be accessed through dashboard
            address.
        cluster_name: This value may be up to 63 characters, and valid
            characters are `[a-z0-9_-]`. The first character cannot be a number
            or hyphen.
        worker_node_types: The list of Resources of the worker nodes. The same
            Resources object should not appear multiple times in the list.
        custom_images: The NodeImages which specifies head node and worker nodes
            images. All the workers will share the same image. If each Resource
            has a specific custom image, use `Resources.custom_image` for
            head/worker_node_type(s). Note that configuring `Resources.custom_image`
            will override `custom_images` here. Allowlist only.
        enable_metrics_collection: Enable Ray metrics collection for visualization.
        labels:
            The labels with user-defined metadata to organize Ray cluster.

            Label keys and values can be no longer than 64 characters (Unicode
            codepoints), can only contain lowercase letters, numeric characters,
            underscores and dashes. International characters are allowed.

            See https://goo.gl/xmQnxf for more information and examples of labels.

    Returns:
        The cluster_resource_name of the initiated Ray cluster on Vertex.
    """

    if network is None:
        logging.info(
            "[Ray on Vertex]: No VPC network configured. It is required for client connection."
        )

    local_ray_verion = _validation_utils.get_local_ray_version()
    if ray_version != local_ray_verion:
        if custom_images is None and head_node_type.custom_image is None:
            install_ray_version = "2.9.3" if ray_version == "2.9" else "2.4.0"
            logging.info(
                "[Ray on Vertex]: Local runtime has Ray version %s"
                ", but the requested cluster runtime has %s. Please "
                "ensure that the Ray versions match for client connectivity. You may "
                '"pip install --user --force-reinstall ray[default]==%s"'
                " and restart runtime before cluster connection."
                % (local_ray_verion, ray_version, install_ray_version)
            )
        else:
            logging.info(
                "[Ray on Vertex]: Local runtime has Ray version %s."
                "Please ensure that the Ray versions match for client connectivity."
                % local_ray_verion
            )

    if cluster_name is None:
        cluster_name = "ray-cluster-" + utils.timestamped_unique_name()

    if head_node_type:
        if head_node_type.node_count != 1:
            raise ValueError(
                "[Ray on Vertex AI]: For head_node_type, "
                + "Resources.node_count must be 1."
            )
        if (
            head_node_type.accelerator_type is None
            and head_node_type.accelerator_count > 0
        ):
            raise ValueError(
                "[Ray on Vertex]: accelerator_type must be specified when"
                + " accelerator_count is set to a value other than 0."
            )

    resource_pool_images = {}

    # head node
    resource_pool_0 = ResourcePool()
    resource_pool_0.id = "head-node"
    resource_pool_0.replica_count = head_node_type.node_count
    resource_pool_0.machine_spec.machine_type = head_node_type.machine_type
    resource_pool_0.machine_spec.accelerator_count = head_node_type.accelerator_count
    resource_pool_0.machine_spec.accelerator_type = head_node_type.accelerator_type
    resource_pool_0.disk_spec.boot_disk_type = head_node_type.boot_disk_type
    resource_pool_0.disk_spec.boot_disk_size_gb = head_node_type.boot_disk_size_gb

    enable_cuda = True if head_node_type.accelerator_count > 0 else False
    if head_node_type.custom_image is not None:
        image_uri = head_node_type.custom_image
    elif custom_images is None:
        image_uri = _validation_utils.get_image_uri(
            ray_version, python_version, enable_cuda
        )
    elif custom_images.head is not None and custom_images.worker is not None:
        image_uri = custom_images.head
    else:
        raise ValueError(
            "[Ray on Vertex AI]: custom_images.head and custom_images.worker must be specified when custom_images is set."
        )

    resource_pool_images[resource_pool_0.id] = image_uri

    worker_pools = []
    i = 0
    if worker_node_types:
        for worker_node_type in worker_node_types:
            if (
                worker_node_type.accelerator_type is None
                and worker_node_type.accelerator_count > 0
            ):
                raise ValueError(
                    "[Ray on Vertex]: accelerator_type must be specified when"
                    + " accelerator_count is set to a value other than 0."
                )
            # Worker and head share the same MachineSpec, merge them into the
            # same ResourcePool
            additional_replica_count = resources._check_machine_spec_identical(
                head_node_type, worker_node_type
            )
            resource_pool_0.replica_count = (
                resource_pool_0.replica_count + additional_replica_count
            )
            if additional_replica_count == 0:
                resource_pool = ResourcePool()
                resource_pool.id = f"worker-pool{i+1}"
                resource_pool.replica_count = worker_node_type.node_count
                resource_pool.machine_spec.machine_type = worker_node_type.machine_type
                resource_pool.machine_spec.accelerator_count = (
                    worker_node_type.accelerator_count
                )
                resource_pool.machine_spec.accelerator_type = (
                    worker_node_type.accelerator_type
                )
                resource_pool.disk_spec.boot_disk_type = worker_node_type.boot_disk_type
                resource_pool.disk_spec.boot_disk_size_gb = (
                    worker_node_type.boot_disk_size_gb
                )
                worker_pools.append(resource_pool)
                enable_cuda = True if worker_node_type.accelerator_count > 0 else False

                if worker_node_type.custom_image is not None:
                    image_uri = worker_node_type.custom_image
                elif custom_images is None:
                    image_uri = _validation_utils.get_image_uri(
                        ray_version, python_version, enable_cuda
                    )
                else:
                    image_uri = custom_images.worker

                resource_pool_images[resource_pool.id] = image_uri

            i += 1

    resource_pools = [resource_pool_0] + worker_pools
    disabled = not enable_metrics_collection
    ray_metric_spec = RayMetricSpec(disabled=disabled)
    ray_spec = RaySpec(
        resource_pool_images=resource_pool_images, ray_metric_spec=ray_metric_spec
    )
    resource_runtime_spec = ResourceRuntimeSpec(ray_spec=ray_spec)
    persistent_resource = PersistentResource(
        resource_pools=resource_pools,
        network=network,
        labels=labels,
        resource_runtime_spec=resource_runtime_spec,
    )

    location = initializer.global_config.location
    project_id = initializer.global_config.project
    project_number = resource_manager_utils.get_project_number(project_id)

    parent = f"projects/{project_number}/locations/{location}"
    request = persistent_resource_service.CreatePersistentResourceRequest(
        parent=parent,
        persistent_resource=persistent_resource,
        persistent_resource_id=cluster_name,
    )

    client = _gapic_utils.create_persistent_resource_client()
    try:
        _ = client.create_persistent_resource(request)
    except Exception as e:
        raise ValueError("Failed in cluster creation due to: ", e) from e

    # Get persisent resource
    cluster_resource_name = f"{parent}/persistentResources/{cluster_name}"
    response = _gapic_utils.get_persistent_resource(
        persistent_resource_name=cluster_resource_name,
        tolerance=1,  # allow 1 retry to avoid get request before creation
    )
    return response.name


def delete_ray_cluster(cluster_resource_name: str) -> None:
    """Delete Ray Cluster.

    Args:
        cluster_resource_name: Cluster resource name.
    Raises:
        FailedPrecondition: If the cluster is deleted already.
    """
    client = _gapic_utils.create_persistent_resource_client()
    request = persistent_resource_service.DeletePersistentResourceRequest(
        name=cluster_resource_name
    )

    try:
        client.delete_persistent_resource(request)
        print("[Ray on Vertex AI]: Successfully deleted the cluster.")
    except Exception as e:
        raise ValueError(
            "[Ray on Vertex AI]: Failed in cluster deletion due to: ", e
        ) from e


def get_ray_cluster(cluster_resource_name: str) -> resources.Cluster:
    """Get Ray Cluster.

    Args:
        cluster_resource_name: Cluster resource name.
    Returns:
        A Cluster object.
    """
    client = _gapic_utils.create_persistent_resource_client()
    request = persistent_resource_service.GetPersistentResourceRequest(
        name=cluster_resource_name
    )
    try:
        response = client.get_persistent_resource(request)
    except Exception as e:
        raise ValueError(
            "[Ray on Vertex AI]: Failed in getting the cluster due to: ", e
        ) from e

    cluster = _gapic_utils.persistent_resource_to_cluster(persistent_resource=response)
    if cluster:
        return cluster
    raise ValueError(
        "[Ray on Vertex AI]: Please delete and recreate the cluster (The cluster is not a Ray cluster or the cluster image is outdated)."
    )


def list_ray_clusters() -> List[resources.Cluster]:
    """List Ray Clusters under the currently authenticated project.

    Returns:
        List of Cluster objects that exists in the current authorized project.
    """
    location = initializer.global_config.location
    project_id = initializer.global_config.project
    project_number = resource_manager_utils.get_project_number(project_id)
    parent = f"projects/{project_number}/locations/{location}"
    request = persistent_resource_service.ListPersistentResourcesRequest(
        parent=parent,
    )
    client = _gapic_utils.create_persistent_resource_client()
    try:
        response = client.list_persistent_resources(request)
    except Exception as e:
        raise ValueError(
            "[Ray on Vertex AI]: Failed in listing the clusters due to: ", e
        ) from e

    ray_clusters = []
    for persistent_resource in response:
        ray_cluster = _gapic_utils.persistent_resource_to_cluster(
            persistent_resource=persistent_resource
        )
        if ray_cluster:
            ray_clusters.append(ray_cluster)

    return ray_clusters


def update_ray_cluster(
    cluster_resource_name: str, worker_node_types: List[resources.Resources]
) -> str:
    """Update Ray Cluster (currently support resizing node counts for worker nodes).

    Sample usage:

    my_cluster = vertex_ray.get_ray_cluster(
            cluster_resource_name=my_existing_cluster_resource_name,
    )

    # Declaration to resize all the worker_node_type to node_count=1
    new_worker_node_types = []
    for worker_node_type in my_cluster.worker_node_types:
        worker_node_type.node_count = 1
        new_worker_node_types.append(worker_node_type)

    # Execution to update new node_count (block until complete)
    vertex_ray.update_ray_cluster(
            cluster_resource_name=my_cluster.cluster_resource_name,
            worker_node_types=new_worker_node_types,
    )

    Args:
        cluster_resource_name:
        worker_node_types: The list of Resources of the resized worker nodes.
            The same Resources object should not appear multiple times in the list.
    Returns:
        The cluster_resource_name of the Ray cluster on Vertex.
    """
    # worker_node_types should not be duplicated.
    for i in range(len(worker_node_types)):
        for j in range(len(worker_node_types)):
            additional_replica_count = resources._check_machine_spec_identical(
                worker_node_types[i], worker_node_types[j]
            )
            if additional_replica_count > 0 and i != j:
                raise ValueError(
                    "[Ray on Vertex AI]: Worker_node_types have duplicate "
                    + f"machine specs: {worker_node_types[i]} "
                    + f"and {worker_node_types[j]}"
                )

    persistent_resource = _gapic_utils.get_persistent_resource(
        persistent_resource_name=cluster_resource_name
    )

    current_persistent_resource = copy.deepcopy(persistent_resource)
    current_persistent_resource.resource_pools[0].replica_count = 1

    previous_ray_cluster = get_ray_cluster(cluster_resource_name)
    head_node_type = previous_ray_cluster.head_node_type
    previous_worker_node_types = previous_ray_cluster.worker_node_types

    # new worker_node_types and previous_worker_node_types should be the same length.
    if len(worker_node_types) != len(previous_worker_node_types):
        raise ValueError(
            "[Ray on Vertex AI]: Desired number of worker_node_types "
            + "(%i) does not match the number of the "
            + "existing worker_node_type(%i).",
            len(worker_node_types),
            len(previous_worker_node_types),
        )

    # Merge worker_node_type and head_node_type if they share
    # the same machine spec.
    not_merged = 1
    for i in range(len(worker_node_types)):
        additional_replica_count = resources._check_machine_spec_identical(
            head_node_type, worker_node_types[i]
        )
        if additional_replica_count != 0 or (
            additional_replica_count == 0 and worker_node_types[i].node_count == 0
        ):
            # Merge the 1st duplicated worker with head, allow scale down to 0 worker
            current_persistent_resource.resource_pools[0].replica_count = (
                1 + additional_replica_count
            )
            # Reset not_merged
            not_merged = 0
        else:
            # No duplication w/ head node, write the 2nd worker node to the 2nd resource pool.
            current_persistent_resource.resource_pools[
                i + not_merged
            ].replica_count = worker_node_types[i].node_count
            # New worker_node_type.node_count should be >=1 unless the worker_node_type
            # and head_node_type are merged due to the same machine specs.
            if worker_node_types[i].node_count == 0:
                raise ValueError(
                    "[Ray on Vertex AI]: Worker_node_type "
                    + f"({worker_node_types[i]}) must update to >= 1 nodes",
                )

    request = persistent_resource_service.UpdatePersistentResourceRequest(
        persistent_resource=current_persistent_resource,
        update_mask=field_mask_pb2.FieldMask(paths=["resource_pools.replica_count"]),
    )
    client = _gapic_utils.create_persistent_resource_client()
    try:
        operation_future = client.update_persistent_resource(request)
    except Exception as e:
        raise ValueError(
            "[Ray on Vertex AI]: Failed in updating the cluster due to: ", e
        ) from e

    # block before returning
    start_time = time.time()
    response = operation_future.result()
    duration = (time.time() - start_time) // 60
    print(
        "[Ray on Vertex AI]: Successfully updated the cluster ({} mininutes elapsed).".format(
            duration
        )
    )
    return response.name
