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

from google.cloud.aiplatform.utils.enhanced_library import _decorators
from google.cloud.aiplatform.v1.schema import predict
from google.cloud.aiplatform.v1.schema import trainingjob
from google.cloud.aiplatform.v1beta1.schema import predict as predict_v1beta1
from google.cloud.aiplatform.v1beta1.schema import predict as trainingjob_v1beta1

# import the v1 submodules for enhancement
from google.cloud.aiplatform.v1.schema.predict.instance_v1 import types as instance
from google.cloud.aiplatform.v1.schema.predict.params_v1 import types as params
from google.cloud.aiplatform.v1.schema.predict.prediction_v1 import types as prediction
from google.cloud.aiplatform.v1.schema.trainingjob.definition_v1 import (
    types as definition,
)

# import the v1beta1 submodules for enhancement
from google.cloud.aiplatform.v1beta1.schema.predict.instance_v1beta1 import (
    types as instance_v1beta1,
)
from google.cloud.aiplatform.v1beta1.schema.predict.params_v1beta1 import (
    types as params_v1beta1,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1 import (
    types as prediction_v1beta1,
)
from google.cloud.aiplatform.v1beta1.schema.trainingjob.definition_v1beta1 import (
    types as definition_v1beta1,
)

__all__ = (
    "predict",
    "trainingjob",
    "predict_v1beta1",
    "trainingjob_v1beta1",
)

enhanced_types_packages = [
    instance,
    params,
    prediction,
    definition,
    instance_v1beta1,
    params_v1beta1,
    prediction_v1beta1,
    definition_v1beta1,
]

for pkg in enhanced_types_packages:
    _decorators._add_methods_to_classes_in_package(pkg)
