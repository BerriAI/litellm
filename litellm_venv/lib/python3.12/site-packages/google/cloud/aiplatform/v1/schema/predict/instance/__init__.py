# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
from google.cloud.aiplatform.v1.schema.predict.instance import (
    gapic_version as package_version,
)

__version__ = package_version.__version__


from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.image_classification import (
    ImageClassificationPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.image_object_detection import (
    ImageObjectDetectionPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.image_segmentation import (
    ImageSegmentationPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.text_classification import (
    TextClassificationPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.text_extraction import (
    TextExtractionPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.text_sentiment import (
    TextSentimentPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.video_action_recognition import (
    VideoActionRecognitionPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.video_classification import (
    VideoClassificationPredictionInstance,
)
from google.cloud.aiplatform.v1.schema.predict.instance_v1.types.video_object_tracking import (
    VideoObjectTrackingPredictionInstance,
)

__all__ = (
    "ImageClassificationPredictionInstance",
    "ImageObjectDetectionPredictionInstance",
    "ImageSegmentationPredictionInstance",
    "TextClassificationPredictionInstance",
    "TextExtractionPredictionInstance",
    "TextSentimentPredictionInstance",
    "VideoActionRecognitionPredictionInstance",
    "VideoClassificationPredictionInstance",
    "VideoObjectTrackingPredictionInstance",
)
