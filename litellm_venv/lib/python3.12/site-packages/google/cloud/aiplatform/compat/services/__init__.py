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

from google.cloud.aiplatform_v1beta1.services.dataset_service import (
    client as dataset_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.deployment_resource_pool_service import (
    client as deployment_resource_pool_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.endpoint_service import (
    client as endpoint_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.extension_execution_service import (
    client as extension_execution_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.extension_registry_service import (
    client as extension_registry_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.feature_online_store_service import (
    client as feature_online_store_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.feature_online_store_admin_service import (
    client as feature_online_store_admin_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.featurestore_online_serving_service import (
    client as featurestore_online_serving_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.featurestore_service import (
    client as featurestore_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.index_service import (
    client as index_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.index_endpoint_service import (
    client as index_endpoint_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.job_service import (
    client as job_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.match_service import (
    client as match_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.metadata_service import (
    client as metadata_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.model_garden_service import (
    client as model_garden_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.model_service import (
    client as model_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.pipeline_service import (
    client as pipeline_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.persistent_resource_service import (
    client as persistent_resource_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    client as prediction_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    async_client as prediction_service_async_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.reasoning_engine_service import (
    client as reasoning_engine_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.reasoning_engine_execution_service import (
    client as reasoning_engine_execution_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.schedule_service import (
    client as schedule_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.specialist_pool_service import (
    client as specialist_pool_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.tensorboard_service import (
    client as tensorboard_service_client_v1beta1,
)
from google.cloud.aiplatform_v1beta1.services.vizier_service import (
    client as vizier_service_client_v1beta1,
)

from google.cloud.aiplatform_v1.services.dataset_service import (
    client as dataset_service_client_v1,
)
from google.cloud.aiplatform_v1.services.endpoint_service import (
    client as endpoint_service_client_v1,
)
from google.cloud.aiplatform_v1.services.feature_online_store_service import (
    client as feature_online_store_service_client_v1,
)
from google.cloud.aiplatform_v1.services.feature_online_store_admin_service import (
    client as feature_online_store_admin_service_client_v1,
)
from google.cloud.aiplatform_v1.services.featurestore_online_serving_service import (
    client as featurestore_online_serving_service_client_v1,
)
from google.cloud.aiplatform_v1.services.featurestore_service import (
    client as featurestore_service_client_v1,
)
from google.cloud.aiplatform_v1.services.index_service import (
    client as index_service_client_v1,
)
from google.cloud.aiplatform_v1.services.index_endpoint_service import (
    client as index_endpoint_service_client_v1,
)
from google.cloud.aiplatform_v1.services.job_service import (
    client as job_service_client_v1,
)
from google.cloud.aiplatform_v1.services.metadata_service import (
    client as metadata_service_client_v1,
)
from google.cloud.aiplatform_v1.services.model_garden_service import (
    client as model_garden_service_client_v1,
)
from google.cloud.aiplatform_v1.services.model_service import (
    client as model_service_client_v1,
)
from google.cloud.aiplatform_v1.services.persistent_resource_service import (
    client as persistent_resource_service_client_v1,
)
from google.cloud.aiplatform_v1.services.pipeline_service import (
    client as pipeline_service_client_v1,
)
from google.cloud.aiplatform_v1.services.prediction_service import (
    client as prediction_service_client_v1,
)
from google.cloud.aiplatform_v1.services.prediction_service import (
    async_client as prediction_service_async_client_v1,
)
from google.cloud.aiplatform_v1.services.schedule_service import (
    client as schedule_service_client_v1,
)
from google.cloud.aiplatform_v1.services.specialist_pool_service import (
    client as specialist_pool_service_client_v1,
)
from google.cloud.aiplatform_v1.services.tensorboard_service import (
    client as tensorboard_service_client_v1,
)
from google.cloud.aiplatform_v1.services.vizier_service import (
    client as vizier_service_client_v1,
)

__all__ = (
    # v1
    dataset_service_client_v1,
    endpoint_service_client_v1,
    feature_online_store_service_client_v1,
    feature_online_store_admin_service_client_v1,
    featurestore_online_serving_service_client_v1,
    featurestore_service_client_v1,
    index_service_client_v1,
    index_endpoint_service_client_v1,
    job_service_client_v1,
    metadata_service_client_v1,
    model_garden_service_client_v1,
    model_service_client_v1,
    persistent_resource_service_client_v1,
    pipeline_service_client_v1,
    prediction_service_client_v1,
    prediction_service_async_client_v1,
    schedule_service_client_v1,
    specialist_pool_service_client_v1,
    tensorboard_service_client_v1,
    vizier_service_client_v1,
    # v1beta1
    dataset_service_client_v1beta1,
    deployment_resource_pool_service_client_v1beta1,
    endpoint_service_client_v1beta1,
    feature_online_store_service_client_v1beta1,
    feature_online_store_admin_service_client_v1beta1,
    featurestore_online_serving_service_client_v1beta1,
    featurestore_service_client_v1beta1,
    index_service_client_v1beta1,
    index_endpoint_service_client_v1beta1,
    job_service_client_v1beta1,
    match_service_client_v1beta1,
    model_garden_service_client_v1beta1,
    model_service_client_v1beta1,
    persistent_resource_service_client_v1beta1,
    pipeline_service_client_v1beta1,
    prediction_service_client_v1beta1,
    prediction_service_async_client_v1beta1,
    schedule_service_client_v1beta1,
    specialist_pool_service_client_v1beta1,
    metadata_service_client_v1beta1,
    tensorboard_service_client_v1beta1,
    vizier_service_client_v1beta1,
)
