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
from google.ai.generativelanguage import gapic_version as package_version

__version__ = package_version.__version__


from google.ai.generativelanguage_v1beta3.services.discuss_service.async_client import (
    DiscussServiceAsyncClient,
)
from google.ai.generativelanguage_v1beta3.services.discuss_service.client import (
    DiscussServiceClient,
)
from google.ai.generativelanguage_v1beta3.services.model_service.async_client import (
    ModelServiceAsyncClient,
)
from google.ai.generativelanguage_v1beta3.services.model_service.client import (
    ModelServiceClient,
)
from google.ai.generativelanguage_v1beta3.services.permission_service.async_client import (
    PermissionServiceAsyncClient,
)
from google.ai.generativelanguage_v1beta3.services.permission_service.client import (
    PermissionServiceClient,
)
from google.ai.generativelanguage_v1beta3.services.text_service.async_client import (
    TextServiceAsyncClient,
)
from google.ai.generativelanguage_v1beta3.services.text_service.client import (
    TextServiceClient,
)
from google.ai.generativelanguage_v1beta3.types.citation import (
    CitationMetadata,
    CitationSource,
)
from google.ai.generativelanguage_v1beta3.types.discuss_service import (
    CountMessageTokensRequest,
    CountMessageTokensResponse,
    Example,
    GenerateMessageRequest,
    GenerateMessageResponse,
    Message,
    MessagePrompt,
)
from google.ai.generativelanguage_v1beta3.types.model import Model
from google.ai.generativelanguage_v1beta3.types.model_service import (
    CreateTunedModelMetadata,
    CreateTunedModelRequest,
    DeleteTunedModelRequest,
    GetModelRequest,
    GetTunedModelRequest,
    ListModelsRequest,
    ListModelsResponse,
    ListTunedModelsRequest,
    ListTunedModelsResponse,
    UpdateTunedModelRequest,
)
from google.ai.generativelanguage_v1beta3.types.permission import Permission
from google.ai.generativelanguage_v1beta3.types.permission_service import (
    CreatePermissionRequest,
    DeletePermissionRequest,
    GetPermissionRequest,
    ListPermissionsRequest,
    ListPermissionsResponse,
    TransferOwnershipRequest,
    TransferOwnershipResponse,
    UpdatePermissionRequest,
)
from google.ai.generativelanguage_v1beta3.types.safety import (
    ContentFilter,
    HarmCategory,
    SafetyFeedback,
    SafetyRating,
    SafetySetting,
)
from google.ai.generativelanguage_v1beta3.types.text_service import (
    BatchEmbedTextRequest,
    BatchEmbedTextResponse,
    CountTextTokensRequest,
    CountTextTokensResponse,
    Embedding,
    EmbedTextRequest,
    EmbedTextResponse,
    GenerateTextRequest,
    GenerateTextResponse,
    TextCompletion,
    TextPrompt,
)
from google.ai.generativelanguage_v1beta3.types.tuned_model import (
    Dataset,
    Hyperparameters,
    TunedModel,
    TunedModelSource,
    TuningExample,
    TuningExamples,
    TuningSnapshot,
    TuningTask,
)

__all__ = (
    "DiscussServiceClient",
    "DiscussServiceAsyncClient",
    "ModelServiceClient",
    "ModelServiceAsyncClient",
    "PermissionServiceClient",
    "PermissionServiceAsyncClient",
    "TextServiceClient",
    "TextServiceAsyncClient",
    "CitationMetadata",
    "CitationSource",
    "CountMessageTokensRequest",
    "CountMessageTokensResponse",
    "Example",
    "GenerateMessageRequest",
    "GenerateMessageResponse",
    "Message",
    "MessagePrompt",
    "Model",
    "CreateTunedModelMetadata",
    "CreateTunedModelRequest",
    "DeleteTunedModelRequest",
    "GetModelRequest",
    "GetTunedModelRequest",
    "ListModelsRequest",
    "ListModelsResponse",
    "ListTunedModelsRequest",
    "ListTunedModelsResponse",
    "UpdateTunedModelRequest",
    "Permission",
    "CreatePermissionRequest",
    "DeletePermissionRequest",
    "GetPermissionRequest",
    "ListPermissionsRequest",
    "ListPermissionsResponse",
    "TransferOwnershipRequest",
    "TransferOwnershipResponse",
    "UpdatePermissionRequest",
    "ContentFilter",
    "SafetyFeedback",
    "SafetyRating",
    "SafetySetting",
    "HarmCategory",
    "BatchEmbedTextRequest",
    "BatchEmbedTextResponse",
    "CountTextTokensRequest",
    "CountTextTokensResponse",
    "Embedding",
    "EmbedTextRequest",
    "EmbedTextResponse",
    "GenerateTextRequest",
    "GenerateTextResponse",
    "TextCompletion",
    "TextPrompt",
    "Dataset",
    "Hyperparameters",
    "TunedModel",
    "TunedModelSource",
    "TuningExample",
    "TuningExamples",
    "TuningSnapshot",
    "TuningTask",
)
