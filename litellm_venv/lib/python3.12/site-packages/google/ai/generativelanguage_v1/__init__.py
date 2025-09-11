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
from google.ai.generativelanguage_v1 import gapic_version as package_version

__version__ = package_version.__version__


from .services.generative_service import (
    GenerativeServiceAsyncClient,
    GenerativeServiceClient,
)
from .services.model_service import ModelServiceAsyncClient, ModelServiceClient
from .types.citation import CitationMetadata, CitationSource
from .types.content import Blob, Content, Part
from .types.generative_service import (
    BatchEmbedContentsRequest,
    BatchEmbedContentsResponse,
    Candidate,
    ContentEmbedding,
    CountTokensRequest,
    CountTokensResponse,
    EmbedContentRequest,
    EmbedContentResponse,
    GenerateContentRequest,
    GenerateContentResponse,
    GenerationConfig,
    TaskType,
)
from .types.model import Model
from .types.model_service import GetModelRequest, ListModelsRequest, ListModelsResponse
from .types.safety import HarmCategory, SafetyRating, SafetySetting

__all__ = (
    "GenerativeServiceAsyncClient",
    "ModelServiceAsyncClient",
    "BatchEmbedContentsRequest",
    "BatchEmbedContentsResponse",
    "Blob",
    "Candidate",
    "CitationMetadata",
    "CitationSource",
    "Content",
    "ContentEmbedding",
    "CountTokensRequest",
    "CountTokensResponse",
    "EmbedContentRequest",
    "EmbedContentResponse",
    "GenerateContentRequest",
    "GenerateContentResponse",
    "GenerationConfig",
    "GenerativeServiceClient",
    "GetModelRequest",
    "HarmCategory",
    "ListModelsRequest",
    "ListModelsResponse",
    "Model",
    "ModelServiceClient",
    "Part",
    "SafetyRating",
    "SafetySetting",
    "TaskType",
)
