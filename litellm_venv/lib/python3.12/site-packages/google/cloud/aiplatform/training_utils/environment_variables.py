# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Environment variables used in Vertex AI Training.

import json
import os

from typing import Dict, Optional


def _json_helper(env_var: str) -> Optional[Dict]:
    """Helper to convert a dictionary represented as a string to a dictionary.

    Args:
        env_var (str):
            Required. The name of the environment variable.

    Returns:
        A dictionary if the variable was found, None otherwise.
    """
    env = os.environ.get(env_var)
    if env is not None:
        return json.loads(env)
    else:
        return None


# Cloud Storage URI of a directory intended for training data.
training_data_uri = os.environ.get("AIP_TRAINING_DATA_URI")

# Cloud Storage URI of a directory intended for validation data.
validation_data_uri = os.environ.get("AIP_VALIDATION_DATA_URI")

# Cloud Storage URI of a directory intended for test data.
test_data_uri = os.environ.get("AIP_TEST_DATA_URI")

# Cloud Storage URI of a directory intended for saving model artefacts.
model_dir = os.environ.get("AIP_MODEL_DIR")

# Cloud Storage URI of a directory intended for saving checkpoints.
checkpoint_dir = os.environ.get("AIP_CHECKPOINT_DIR")

# Cloud Storage URI of a directory intended for saving TensorBoard logs.
tensorboard_log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")

# json string as described in https://cloud.google.com/ai-platform-unified/docs/training/distributed-training#cluster-variables
cluster_spec = _json_helper("CLUSTER_SPEC")

# json string as described in https://cloud.google.com/ai-platform-unified/docs/training/distributed-training#tf-config
tf_config = _json_helper("TF_CONFIG")

# Profiler port used for capturing profiling samples.
tf_profiler_port = os.environ.get("AIP_TF_PROFILER_PORT")

# API URI used for the tensorboard uploader.
tensorboard_api_uri = os.environ.get("AIP_TENSORBOARD_API_URI")

# The name of the tensorboard resource, in the form:
# `projects/{project_id}/locations/{location}/tensorboards/{tensorboard_name}`
tensorboard_resource_name = os.environ.get("AIP_TENSORBOARD_RESOURCE_NAME")

# The name given to the training job.
cloud_ml_job_id = os.environ.get("CLOUD_ML_JOB_ID")

# The HTTP Handler port to use to host the profiling webserver.
http_handler_port = os.environ.get("AIP_HTTP_HANDLER_PORT")
