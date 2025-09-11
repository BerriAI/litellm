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

from typing import Optional

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from vertexai.preview._workflow.executor import (
    persistent_resource_util,
)
from vertexai.preview._workflow.shared import configs


_LOGGER = base.Logger(__name__)


class _Config:
    """Store common configurations and current workflow for remote execution."""

    def __init__(self):
        self._remote = False
        self._cluster = None

    def init(
        self,
        *,
        remote: Optional[bool] = None,
        autolog: Optional[bool] = None,
        cluster: Optional[configs.PersistentResourceConfig] = None,
    ):
        """Updates preview global parameters for Vertex remote execution.

        Args:
            remote (bool):
                Optional. A global flag to indicate whether or not a method will
                be executed remotely. Default is Flase. The method level remote
                flag has higher priority than this global flag.
            autolog (bool):
                Optional. Whether or not to turn on autologging feature for remote
                execution. To learn more about the autologging feature, see
                https://cloud.google.com/vertex-ai/docs/experiments/autolog-data.
            cluster (PersistentResourceConfig):
                Optional. If passed, check if the cluster exists. If not, create
                a default one (single node, "n1-standard-4", no GPU) with the
                given name. Then use the cluster to run CustomJobs. Default is
                None. Example usage:
                from vertexai.preview.shared.configs import PersistentResourceConfig
                cluster = PersistentResourceConfig(
                        name="my-cluster-1",
                        resource_pools=[
                                ResourcePool(replica_count=1,),
                                ResourcePool(
                                        machine_type="n1-standard-8",
                                        replica_count=2,
                                        accelerator_type="NVIDIA_TESLA_P100",
                                        accelerator_count=1,
                                        ),
                        ]
                )
        """
        if remote is not None:
            self._remote = remote

        if autolog is True:
            aiplatform.autolog()
        elif autolog is False:
            aiplatform.autolog(disable=True)

        if cluster is not None:
            if cluster.disable:
                self._cluster = None
            else:
                self._cluster = cluster
                cluster_resource_name = persistent_resource_util.cluster_resource_name(
                    project=self.project,
                    location=self.location,
                    name=self._cluster.name,
                )
                cluster_exists = persistent_resource_util.check_persistent_resource(
                    cluster_resource_name=cluster_resource_name,
                    service_account=cluster.service_account,
                )
                if cluster_exists:
                    _LOGGER.info(f"Using existing cluster: {cluster_resource_name}")
                    return
                # create a cluster
                persistent_resource_util.create_persistent_resource(
                    cluster_resource_name=cluster_resource_name,
                    resource_pools=cluster.resource_pools,
                    service_account=cluster.service_account,
                )

    @property
    def remote(self):
        return self._remote

    @property
    def autolog(self):
        return aiplatform.utils.autologging_utils._is_autologging_enabled()

    @property
    def cluster(self):
        return self._cluster

    def __getattr__(self, name):
        return getattr(aiplatform.initializer.global_config, name)


global_config = _Config()
