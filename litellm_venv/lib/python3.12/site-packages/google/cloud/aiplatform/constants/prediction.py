# Copyright 2022 Google LLC
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

import re

from collections import defaultdict

# [region]-docker.pkg.dev/vertex-ai/prediction/[framework]-[accelerator].[version]:latest
CONTAINER_URI_PATTERN = re.compile(
    r"(?P<region>[\w]+)\-docker\.pkg\.dev\/vertex\-ai\/prediction\/"
    r"(?P<framework>[\w]+)\-(?P<accelerator>[\w]+)\.(?P<version>[\d-]+):latest"
)

CONTAINER_URI_REGEX = (
    r"^(us|europe|asia)-docker.pkg.dev/"
    r"vertex-ai/prediction/"
    r"(tf|sklearn|xgboost|pytorch).+$"
)

SKLEARN = "sklearn"
TF = "tf"
TF2 = "tf2"
XGBOOST = "xgboost"

XGBOOST_CONTAINER_URIS = [
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-3:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-3:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-3:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-2:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-2:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-2:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.0-90:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.0-90:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.0-90:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.0-82:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.0-82:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.0-82:latest",
]

SKLEARN_CONTAINER_URIS = [
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-22:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-22:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-22:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-20:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-20:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-20:latest",
]

TF_CONTAINER_URIS = [
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-13:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-13:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-13:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-12:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-12:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-12:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-11:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-10:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-10:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-10:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-10:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-10:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-10:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-9:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-9:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-9:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-8:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-8:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-8:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-7:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-7:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-7:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-5:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-5:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-5:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-5:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-5:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-5:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-4:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-4:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-4:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-4:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-4:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-4:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-3:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-3:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-3:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-2:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-2:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-2:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-2:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-2:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-2:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-1:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-1:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-1:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-1:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-1:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-1:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf-cpu.1-15:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf-cpu.1-15:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf-cpu.1-15:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/tf-gpu.1-15:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/tf-gpu.1-15:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/tf-gpu.1-15:latest",
]

PYTORCH_CONTAINER_URIS = [
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-0:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-11:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-11:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-11:latest",
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-11:latest",
    "europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-11:latest",
    "asia-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-11:latest",
]

SERVING_CONTAINER_URIS = (
    SKLEARN_CONTAINER_URIS
    + TF_CONTAINER_URIS
    + XGBOOST_CONTAINER_URIS
    + PYTORCH_CONTAINER_URIS
)

# Map of all first-party prediction containers
d = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))

for container_uri in SERVING_CONTAINER_URIS:
    m = CONTAINER_URI_PATTERN.match(container_uri)
    region, framework, accelerator, version = m[1], m[2], m[3], m[4]
    version = version.replace("-", ".")

    if framework in (TF2, TF):  # Store both `tf`, `tf2` as `tensorflow`
        framework = "tensorflow"

    d[region][framework][accelerator][version] = container_uri

_SERVING_CONTAINER_URI_MAP = d

_SERVING_CONTAINER_DOCUMENTATION_URL = (
    "https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers"
)

# Variables set by Vertex AI. For more details, please refer to
# https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables
DEFAULT_AIP_HTTP_PORT = 8080
AIP_HTTP_PORT = "AIP_HTTP_PORT"
AIP_HEALTH_ROUTE = "AIP_HEALTH_ROUTE"
AIP_PREDICT_ROUTE = "AIP_PREDICT_ROUTE"
AIP_STORAGE_URI = "AIP_STORAGE_URI"

# Default values for Prediction local experience.
DEFAULT_LOCAL_PREDICT_ROUTE = "/predict"
DEFAULT_LOCAL_HEALTH_ROUTE = "/health"
DEFAULT_LOCAL_RUN_GPU_CAPABILITIES = [["utility", "compute"]]
DEFAULT_LOCAL_RUN_GPU_COUNT = -1

CUSTOM_PREDICTION_ROUTINES = "custom-prediction-routines"
CUSTOM_PREDICTION_ROUTINES_SERVER_ERROR_HEADER_KEY = "X-AIP-CPR-SYSTEM-ERROR"

# Headers' related constants for the handler usage.
CONTENT_TYPE_HEADER_REGEX = re.compile("^[Cc]ontent-?[Tt]ype$")
ACCEPT_HEADER_REGEX = re.compile("^[Aa]ccept$")
ANY_ACCEPT_TYPE = "*/*"
DEFAULT_ACCEPT_VALUE = "application/json"

# Model filenames.
MODEL_FILENAME_BST = "model.bst"
MODEL_FILENAME_JOBLIB = "model.joblib"
MODEL_FILENAME_PKL = "model.pkl"
