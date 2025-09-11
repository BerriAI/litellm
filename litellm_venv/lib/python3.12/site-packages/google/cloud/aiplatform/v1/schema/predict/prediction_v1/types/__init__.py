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
from .classification import (
    ClassificationPredictionResult,
)
from .image_object_detection import (
    ImageObjectDetectionPredictionResult,
)
from .image_segmentation import (
    ImageSegmentationPredictionResult,
)
from .tabular_classification import (
    TabularClassificationPredictionResult,
)
from .tabular_regression import (
    TabularRegressionPredictionResult,
)
from .text_extraction import (
    TextExtractionPredictionResult,
)
from .text_sentiment import (
    TextSentimentPredictionResult,
)
from .video_action_recognition import (
    VideoActionRecognitionPredictionResult,
)
from .video_classification import (
    VideoClassificationPredictionResult,
)
from .video_object_tracking import (
    VideoObjectTrackingPredictionResult,
)

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
