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
from google.cloud.aiplatform.v1.schema.predict.prediction_v1 import (
    gapic_version as package_version,
)

__version__ = package_version.__version__


from .types.classification import ClassificationPredictionResult
from .types.image_object_detection import ImageObjectDetectionPredictionResult
from .types.image_segmentation import ImageSegmentationPredictionResult
from .types.tabular_classification import TabularClassificationPredictionResult
from .types.tabular_regression import TabularRegressionPredictionResult
from .types.text_extraction import TextExtractionPredictionResult
from .types.text_sentiment import TextSentimentPredictionResult
from .types.video_action_recognition import VideoActionRecognitionPredictionResult
from .types.video_classification import VideoClassificationPredictionResult
from .types.video_object_tracking import VideoObjectTrackingPredictionResult

__all__ = (
    "ClassificationPredictionResult",
    "ImageObjectDetectionPredictionResult",
    "ImageSegmentationPredictionResult",
    "TabularClassificationPredictionResult",
    "TabularRegressionPredictionResult",
    "TextExtractionPredictionResult",
    "TextSentimentPredictionResult",
    "VideoActionRecognitionPredictionResult",
    "VideoClassificationPredictionResult",
    "VideoObjectTrackingPredictionResult",
)
