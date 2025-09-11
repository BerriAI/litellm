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
from google.cloud.aiplatform.v1beta1.schema.predict.prediction import (
    gapic_version as package_version,
)

__version__ = package_version.__version__


from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.classification import (
    ClassificationPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.image_object_detection import (
    ImageObjectDetectionPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.image_segmentation import (
    ImageSegmentationPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.tabular_classification import (
    TabularClassificationPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.tabular_regression import (
    TabularRegressionPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.text_extraction import (
    TextExtractionPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.text_sentiment import (
    TextSentimentPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.time_series_forecasting import (
    TimeSeriesForecastingPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.video_action_recognition import (
    VideoActionRecognitionPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.video_classification import (
    VideoClassificationPredictionResult,
)
from google.cloud.aiplatform.v1beta1.schema.predict.prediction_v1beta1.types.video_object_tracking import (
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
    "TimeSeriesForecastingPredictionResult",
    "VideoActionRecognitionPredictionResult",
    "VideoClassificationPredictionResult",
    "VideoObjectTrackingPredictionResult",
)
