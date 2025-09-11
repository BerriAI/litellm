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
"""Classes for working with vision models."""

from vertexai.vision_models._vision_models import (
    GeneratedImage,
    Image,
    ImageCaptioningModel,
    ImageGenerationModel,
    ImageGenerationResponse,
    ImageQnAModel,
    ImageTextModel,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
    Video,
    VideoEmbedding,
    VideoSegmentConfig,
    WatermarkVerificationModel,
    WatermarkVerificationResponse,
)

__all__ = [
    "Image",
    "ImageGenerationModel",
    "ImageGenerationResponse",
    "ImageCaptioningModel",
    "ImageQnAModel",
    "ImageTextModel",
    "WatermarkVerificationModel",
    "GeneratedImage",
    "MultiModalEmbeddingModel",
    "MultiModalEmbeddingResponse",
    "Video",
    "VideoEmbedding",
    "VideoSegmentConfig",
    "WatermarkVerificationResponse",
]
