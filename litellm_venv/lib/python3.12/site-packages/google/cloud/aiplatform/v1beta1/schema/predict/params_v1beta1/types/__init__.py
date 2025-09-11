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
from .image_classification import (
    ImageClassificationPredictionParams,
)
from .image_object_detection import (
    ImageObjectDetectionPredictionParams,
)
from .image_segmentation import (
    ImageSegmentationPredictionParams,
)
from .video_action_recognition import (
    VideoActionRecognitionPredictionParams,
)
from .video_classification import (
    VideoClassificationPredictionParams,
)
from .video_object_tracking import (
    VideoObjectTrackingPredictionParams,
)

__all__ = (
    "ImageClassificationPredictionParams",
    "ImageObjectDetectionPredictionParams",
    "ImageSegmentationPredictionParams",
    "VideoActionRecognitionPredictionParams",
    "VideoClassificationPredictionParams",
    "VideoObjectTrackingPredictionParams",
)
