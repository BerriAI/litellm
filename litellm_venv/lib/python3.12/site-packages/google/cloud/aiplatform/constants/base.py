# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from google.cloud.aiplatform import version as aiplatform_version


DEFAULT_REGION = "us-central1"
SUPPORTED_REGIONS = frozenset(
    {
        "africa-south1",
        "asia-east1",
        "asia-east2",
        "asia-northeast1",
        "asia-northeast2",
        "asia-northeast3",
        "asia-south1",
        "asia-southeast1",
        "asia-southeast2",
        "australia-southeast1",
        "australia-southeast2",
        "europe-central2",
        "europe-north1",
        "europe-southwest1",
        "europe-west1",
        "europe-west2",
        "europe-west3",
        "europe-west4",
        "europe-west6",
        "europe-west8",
        "europe-west9",
        "europe-west12",
        "me-central1",
        "me-central2",
        "me-west1",
        "northamerica-northeast1",
        "northamerica-northeast2",
        "southamerica-east1",
        "southamerica-west1",
        "us-central1",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-south1",
        "us-west1",
        "us-west2",
        "us-west3",
        "us-west4",
    }
)

API_BASE_PATH = "aiplatform.googleapis.com"
PREDICTION_API_BASE_PATH = API_BASE_PATH

# Batch Prediction
BATCH_PREDICTION_INPUT_STORAGE_FORMATS = (
    "jsonl",
    "csv",
    "tf-record",
    "tf-record-gzip",
    "bigquery",
    "file-list",
)
BATCH_PREDICTION_OUTPUT_STORAGE_FORMATS = ("jsonl", "csv", "bigquery")

MOBILE_TF_MODEL_TYPES = {
    "MOBILE_TF_LOW_LATENCY_1",
    "MOBILE_TF_VERSATILE_1",
    "MOBILE_TF_HIGH_ACCURACY_1",
}

MODEL_GARDEN_ICN_MODEL_TYPES = {
    "EFFICIENTNET",
    "MAXVIT",
    "VIT",
    "COCA",
}

MODEL_GARDEN_IOD_MODEL_TYPES = {
    "SPINENET",
    "YOLO",
}

# TODO(b/177079208): Use EPCL Enums for validating Model Types
# Defined by gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_image_*
# Format: "prediction_type": set() of model_type's
#
# NOTE: When adding a new prediction_type's, ensure it fits the pattern
#       "automl_image_{prediction_type}_*" used by the YAML schemas on GCS
AUTOML_IMAGE_PREDICTION_MODEL_TYPES = {
    "classification": {"CLOUD", "CLOUD_1"}
    | MOBILE_TF_MODEL_TYPES
    | MODEL_GARDEN_ICN_MODEL_TYPES,
    "object_detection": {"CLOUD_1", "CLOUD_HIGH_ACCURACY_1", "CLOUD_LOW_LATENCY_1"}
    | MOBILE_TF_MODEL_TYPES
    | MODEL_GARDEN_IOD_MODEL_TYPES,
}

AUTOML_VIDEO_PREDICTION_MODEL_TYPES = {
    "classification": {"CLOUD"} | {"MOBILE_VERSATILE_1"},
    "action_recognition": {"CLOUD"} | {"MOBILE_VERSATILE_1"},
    "object_tracking": {"CLOUD"}
    | {
        "MOBILE_VERSATILE_1",
        "MOBILE_CORAL_VERSATILE_1",
        "MOBILE_CORAL_LOW_LATENCY_1",
        "MOBILE_JETSON_VERSATILE_1",
        "MOBILE_JETSON_LOW_LATENCY_1",
    },
}

# Used in constructing the requests user_agent header for metrics reporting.
USER_AGENT_PRODUCT = "model-builder"
# This field is used to pass the name of the specific SDK method
# that is being used for usage metrics tracking purposes.
# For more details on go/oneplatform-api-analytics
USER_AGENT_SDK_COMMAND = ""

# Needed for Endpoint.raw_predict
DEFAULT_AUTHED_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Used in CustomJob.from_local_script for experiments integration in training
AIPLATFORM_DEPENDENCY_PATH = (
    f"google-cloud-aiplatform=={aiplatform_version.__version__}"
)

AIPLATFORM_AUTOLOG_DEPENDENCY_PATH = (
    f"google-cloud-aiplatform[autologging]=={aiplatform_version.__version__}"
)
