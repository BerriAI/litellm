# -*- coding: utf-8 -*-
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
from google.ai.generativelanguage_v1beta2 import gapic_version as package_version

__version__ = package_version.__version__


from .services.discuss_service import DiscussServiceAsyncClient, DiscussServiceClient
from .services.model_service import ModelServiceAsyncClient, ModelServiceClient
from .services.text_service import TextServiceAsyncClient, TextServiceClient
from .types.citation import CitationMetadata, CitationSource
from .types.discuss_service import (
    CountMessageTokensRequest,
    CountMessageTokensResponse,
    Example,
    GenerateMessageRequest,
    GenerateMessageResponse,
    Message,
    MessagePrompt,
)
from .types.model import Model
from .types.model_service import GetModelRequest, ListModelsRequest, ListModelsResponse
from .types.safety import (
    ContentFilter,
    HarmCategory,
    SafetyFeedback,
    SafetyRating,
    SafetySetting,
)
from .types.text_service import (
    Embedding,
    EmbedTextRequest,
    EmbedTextResponse,
    GenerateTextRequest,
    GenerateTextResponse,
    TextCompletion,
    TextPrompt,
)

__all__ = (
    "DiscussServiceAsyncClient",
    "ModelServiceAsyncClient",
    "TextServiceAsyncClient",
    "CitationMetadata",
    "CitationSource",
    "ContentFilter",
    "CountMessageTokensRequest",
    "CountMessageTokensResponse",
    "DiscussServiceClient",
    "EmbedTextRequest",
    "EmbedTextResponse",
    "Embedding",
    "Example",
    "GenerateMessageRequest",
    "GenerateMessageResponse",
    "GenerateTextRequest",
    "GenerateTextResponse",
    "GetModelRequest",
    "HarmCategory",
    "ListModelsRequest",
    "ListModelsResponse",
    "Message",
    "MessagePrompt",
    "Model",
    "ModelServiceClient",
    "SafetyFeedback",
    "SafetyRating",
    "SafetySetting",
    "TextCompletion",
    "TextPrompt",
    "TextServiceClient",
)
