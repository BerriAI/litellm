# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# ***********
# `huggingface_hub` init has 2 modes:
# - Normal usage:
#       If imported to use it, all modules and functions are lazy-loaded. This means
#       they exist at top level in module but are imported only the first time they are
#       used. This way, `from huggingface_hub import something` will import `something`
#       quickly without the hassle of importing all the features from `huggingface_hub`.
# - Static check:
#       If statically analyzed, all modules and functions are loaded normally. This way
#       static typing check works properly as well as autocomplete in text editors and
#       IDEs.
#
# The static model imports are done inside the `if TYPE_CHECKING:` statement at
# the bottom of this file. Since module/functions imports are duplicated, it is
# mandatory to make sure to add them twice when adding one. This is checked in the
# `make quality` command.
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_static_imports.py --update-file
#
# # Or run style on codebase
# make style
# ```
#
# ***********
# Lazy loader vendored from https://github.com/scientific-python/lazy_loader
import importlib
import os
import sys
from typing import TYPE_CHECKING


__version__ = "0.34.4"

# Alphabetical order of definitions is ensured in tests
# WARNING: any comment added in this dictionary definition will be lost when
# re-generating the file !
_SUBMOD_ATTRS = {
    "_commit_scheduler": [
        "CommitScheduler",
    ],
    "_inference_endpoints": [
        "InferenceEndpoint",
        "InferenceEndpointError",
        "InferenceEndpointStatus",
        "InferenceEndpointTimeoutError",
        "InferenceEndpointType",
    ],
    "_jobs_api": [
        "JobInfo",
        "JobOwner",
        "JobStage",
        "JobStatus",
    ],
    "_login": [
        "auth_list",
        "auth_switch",
        "interpreter_login",
        "login",
        "logout",
        "notebook_login",
    ],
    "_oauth": [
        "OAuthInfo",
        "OAuthOrgInfo",
        "OAuthUserInfo",
        "attach_huggingface_oauth",
        "parse_huggingface_oauth",
    ],
    "_snapshot_download": [
        "snapshot_download",
    ],
    "_space_api": [
        "SpaceHardware",
        "SpaceRuntime",
        "SpaceStage",
        "SpaceStorage",
        "SpaceVariable",
    ],
    "_tensorboard_logger": [
        "HFSummaryWriter",
    ],
    "_webhooks_payload": [
        "WebhookPayload",
        "WebhookPayloadComment",
        "WebhookPayloadDiscussion",
        "WebhookPayloadDiscussionChanges",
        "WebhookPayloadEvent",
        "WebhookPayloadMovedTo",
        "WebhookPayloadRepo",
        "WebhookPayloadUrl",
        "WebhookPayloadWebhook",
    ],
    "_webhooks_server": [
        "WebhooksServer",
        "webhook_endpoint",
    ],
    "community": [
        "Discussion",
        "DiscussionComment",
        "DiscussionCommit",
        "DiscussionEvent",
        "DiscussionStatusChange",
        "DiscussionTitleChange",
        "DiscussionWithDetails",
    ],
    "constants": [
        "CONFIG_NAME",
        "FLAX_WEIGHTS_NAME",
        "HUGGINGFACE_CO_URL_HOME",
        "HUGGINGFACE_CO_URL_TEMPLATE",
        "PYTORCH_WEIGHTS_NAME",
        "REPO_TYPE_DATASET",
        "REPO_TYPE_MODEL",
        "REPO_TYPE_SPACE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
    ],
    "fastai_utils": [
        "_save_pretrained_fastai",
        "from_pretrained_fastai",
        "push_to_hub_fastai",
    ],
    "file_download": [
        "HfFileMetadata",
        "_CACHED_NO_EXIST",
        "get_hf_file_metadata",
        "hf_hub_download",
        "hf_hub_url",
        "try_to_load_from_cache",
    ],
    "hf_api": [
        "Collection",
        "CollectionItem",
        "CommitInfo",
        "CommitOperation",
        "CommitOperationAdd",
        "CommitOperationCopy",
        "CommitOperationDelete",
        "DatasetInfo",
        "GitCommitInfo",
        "GitRefInfo",
        "GitRefs",
        "HfApi",
        "ModelInfo",
        "RepoUrl",
        "SpaceInfo",
        "User",
        "UserLikes",
        "WebhookInfo",
        "WebhookWatchedItem",
        "accept_access_request",
        "add_collection_item",
        "add_space_secret",
        "add_space_variable",
        "auth_check",
        "cancel_access_request",
        "cancel_job",
        "change_discussion_status",
        "comment_discussion",
        "create_branch",
        "create_collection",
        "create_commit",
        "create_discussion",
        "create_inference_endpoint",
        "create_inference_endpoint_from_catalog",
        "create_pull_request",
        "create_repo",
        "create_tag",
        "create_webhook",
        "dataset_info",
        "delete_branch",
        "delete_collection",
        "delete_collection_item",
        "delete_file",
        "delete_folder",
        "delete_inference_endpoint",
        "delete_repo",
        "delete_space_secret",
        "delete_space_storage",
        "delete_space_variable",
        "delete_tag",
        "delete_webhook",
        "disable_webhook",
        "duplicate_space",
        "edit_discussion_comment",
        "enable_webhook",
        "fetch_job_logs",
        "file_exists",
        "get_collection",
        "get_dataset_tags",
        "get_discussion_details",
        "get_full_repo_name",
        "get_inference_endpoint",
        "get_model_tags",
        "get_paths_info",
        "get_repo_discussions",
        "get_safetensors_metadata",
        "get_space_runtime",
        "get_space_variables",
        "get_token_permission",
        "get_user_overview",
        "get_webhook",
        "grant_access",
        "inspect_job",
        "list_accepted_access_requests",
        "list_collections",
        "list_datasets",
        "list_inference_catalog",
        "list_inference_endpoints",
        "list_jobs",
        "list_lfs_files",
        "list_liked_repos",
        "list_models",
        "list_organization_members",
        "list_papers",
        "list_pending_access_requests",
        "list_rejected_access_requests",
        "list_repo_commits",
        "list_repo_files",
        "list_repo_likers",
        "list_repo_refs",
        "list_repo_tree",
        "list_spaces",
        "list_user_followers",
        "list_user_following",
        "list_webhooks",
        "merge_pull_request",
        "model_info",
        "move_repo",
        "paper_info",
        "parse_safetensors_file_metadata",
        "pause_inference_endpoint",
        "pause_space",
        "permanently_delete_lfs_files",
        "preupload_lfs_files",
        "reject_access_request",
        "rename_discussion",
        "repo_exists",
        "repo_info",
        "repo_type_and_id_from_hf_id",
        "request_space_hardware",
        "request_space_storage",
        "restart_space",
        "resume_inference_endpoint",
        "revision_exists",
        "run_as_future",
        "run_job",
        "run_uv_job",
        "scale_to_zero_inference_endpoint",
        "set_space_sleep_time",
        "space_info",
        "super_squash_history",
        "unlike",
        "update_collection_item",
        "update_collection_metadata",
        "update_inference_endpoint",
        "update_repo_settings",
        "update_repo_visibility",
        "update_webhook",
        "upload_file",
        "upload_folder",
        "upload_large_folder",
        "whoami",
    ],
    "hf_file_system": [
        "HfFileSystem",
        "HfFileSystemFile",
        "HfFileSystemResolvedPath",
        "HfFileSystemStreamFile",
    ],
    "hub_mixin": [
        "ModelHubMixin",
        "PyTorchModelHubMixin",
    ],
    "inference._client": [
        "InferenceClient",
        "InferenceTimeoutError",
    ],
    "inference._generated._async_client": [
        "AsyncInferenceClient",
    ],
    "inference._generated.types": [
        "AudioClassificationInput",
        "AudioClassificationOutputElement",
        "AudioClassificationOutputTransform",
        "AudioClassificationParameters",
        "AudioToAudioInput",
        "AudioToAudioOutputElement",
        "AutomaticSpeechRecognitionEarlyStoppingEnum",
        "AutomaticSpeechRecognitionGenerationParameters",
        "AutomaticSpeechRecognitionInput",
        "AutomaticSpeechRecognitionOutput",
        "AutomaticSpeechRecognitionOutputChunk",
        "AutomaticSpeechRecognitionParameters",
        "ChatCompletionInput",
        "ChatCompletionInputFunctionDefinition",
        "ChatCompletionInputFunctionName",
        "ChatCompletionInputGrammarType",
        "ChatCompletionInputJSONSchema",
        "ChatCompletionInputMessage",
        "ChatCompletionInputMessageChunk",
        "ChatCompletionInputMessageChunkType",
        "ChatCompletionInputResponseFormatJSONObject",
        "ChatCompletionInputResponseFormatJSONSchema",
        "ChatCompletionInputResponseFormatText",
        "ChatCompletionInputStreamOptions",
        "ChatCompletionInputTool",
        "ChatCompletionInputToolCall",
        "ChatCompletionInputToolChoiceClass",
        "ChatCompletionInputToolChoiceEnum",
        "ChatCompletionInputURL",
        "ChatCompletionOutput",
        "ChatCompletionOutputComplete",
        "ChatCompletionOutputFunctionDefinition",
        "ChatCompletionOutputLogprob",
        "ChatCompletionOutputLogprobs",
        "ChatCompletionOutputMessage",
        "ChatCompletionOutputToolCall",
        "ChatCompletionOutputTopLogprob",
        "ChatCompletionOutputUsage",
        "ChatCompletionStreamOutput",
        "ChatCompletionStreamOutputChoice",
        "ChatCompletionStreamOutputDelta",
        "ChatCompletionStreamOutputDeltaToolCall",
        "ChatCompletionStreamOutputFunction",
        "ChatCompletionStreamOutputLogprob",
        "ChatCompletionStreamOutputLogprobs",
        "ChatCompletionStreamOutputTopLogprob",
        "ChatCompletionStreamOutputUsage",
        "DepthEstimationInput",
        "DepthEstimationOutput",
        "DocumentQuestionAnsweringInput",
        "DocumentQuestionAnsweringInputData",
        "DocumentQuestionAnsweringOutputElement",
        "DocumentQuestionAnsweringParameters",
        "FeatureExtractionInput",
        "FeatureExtractionInputTruncationDirection",
        "FillMaskInput",
        "FillMaskOutputElement",
        "FillMaskParameters",
        "ImageClassificationInput",
        "ImageClassificationOutputElement",
        "ImageClassificationOutputTransform",
        "ImageClassificationParameters",
        "ImageSegmentationInput",
        "ImageSegmentationOutputElement",
        "ImageSegmentationParameters",
        "ImageSegmentationSubtask",
        "ImageToImageInput",
        "ImageToImageOutput",
        "ImageToImageParameters",
        "ImageToImageTargetSize",
        "ImageToTextEarlyStoppingEnum",
        "ImageToTextGenerationParameters",
        "ImageToTextInput",
        "ImageToTextOutput",
        "ImageToTextParameters",
        "ImageToVideoInput",
        "ImageToVideoOutput",
        "ImageToVideoParameters",
        "ImageToVideoTargetSize",
        "ObjectDetectionBoundingBox",
        "ObjectDetectionInput",
        "ObjectDetectionOutputElement",
        "ObjectDetectionParameters",
        "Padding",
        "QuestionAnsweringInput",
        "QuestionAnsweringInputData",
        "QuestionAnsweringOutputElement",
        "QuestionAnsweringParameters",
        "SentenceSimilarityInput",
        "SentenceSimilarityInputData",
        "SummarizationInput",
        "SummarizationOutput",
        "SummarizationParameters",
        "SummarizationTruncationStrategy",
        "TableQuestionAnsweringInput",
        "TableQuestionAnsweringInputData",
        "TableQuestionAnsweringOutputElement",
        "TableQuestionAnsweringParameters",
        "Text2TextGenerationInput",
        "Text2TextGenerationOutput",
        "Text2TextGenerationParameters",
        "Text2TextGenerationTruncationStrategy",
        "TextClassificationInput",
        "TextClassificationOutputElement",
        "TextClassificationOutputTransform",
        "TextClassificationParameters",
        "TextGenerationInput",
        "TextGenerationInputGenerateParameters",
        "TextGenerationInputGrammarType",
        "TextGenerationOutput",
        "TextGenerationOutputBestOfSequence",
        "TextGenerationOutputDetails",
        "TextGenerationOutputFinishReason",
        "TextGenerationOutputPrefillToken",
        "TextGenerationOutputToken",
        "TextGenerationStreamOutput",
        "TextGenerationStreamOutputStreamDetails",
        "TextGenerationStreamOutputToken",
        "TextToAudioEarlyStoppingEnum",
        "TextToAudioGenerationParameters",
        "TextToAudioInput",
        "TextToAudioOutput",
        "TextToAudioParameters",
        "TextToImageInput",
        "TextToImageOutput",
        "TextToImageParameters",
        "TextToSpeechEarlyStoppingEnum",
        "TextToSpeechGenerationParameters",
        "TextToSpeechInput",
        "TextToSpeechOutput",
        "TextToSpeechParameters",
        "TextToVideoInput",
        "TextToVideoOutput",
        "TextToVideoParameters",
        "TokenClassificationAggregationStrategy",
        "TokenClassificationInput",
        "TokenClassificationOutputElement",
        "TokenClassificationParameters",
        "TranslationInput",
        "TranslationOutput",
        "TranslationParameters",
        "TranslationTruncationStrategy",
        "TypeEnum",
        "VideoClassificationInput",
        "VideoClassificationOutputElement",
        "VideoClassificationOutputTransform",
        "VideoClassificationParameters",
        "VisualQuestionAnsweringInput",
        "VisualQuestionAnsweringInputData",
        "VisualQuestionAnsweringOutputElement",
        "VisualQuestionAnsweringParameters",
        "ZeroShotClassificationInput",
        "ZeroShotClassificationOutputElement",
        "ZeroShotClassificationParameters",
        "ZeroShotImageClassificationInput",
        "ZeroShotImageClassificationOutputElement",
        "ZeroShotImageClassificationParameters",
        "ZeroShotObjectDetectionBoundingBox",
        "ZeroShotObjectDetectionInput",
        "ZeroShotObjectDetectionOutputElement",
        "ZeroShotObjectDetectionParameters",
    ],
    "inference._mcp.agent": [
        "Agent",
    ],
    "inference._mcp.mcp_client": [
        "MCPClient",
    ],
    "inference_api": [
        "InferenceApi",
    ],
    "keras_mixin": [
        "KerasModelHubMixin",
        "from_pretrained_keras",
        "push_to_hub_keras",
        "save_pretrained_keras",
    ],
    "repocard": [
        "DatasetCard",
        "ModelCard",
        "RepoCard",
        "SpaceCard",
        "metadata_eval_result",
        "metadata_load",
        "metadata_save",
        "metadata_update",
    ],
    "repocard_data": [
        "CardData",
        "DatasetCardData",
        "EvalResult",
        "ModelCardData",
        "SpaceCardData",
    ],
    "repository": [
        "Repository",
    ],
    "serialization": [
        "StateDictSplit",
        "get_tf_storage_size",
        "get_torch_storage_id",
        "get_torch_storage_size",
        "load_state_dict_from_file",
        "load_torch_model",
        "save_torch_model",
        "save_torch_state_dict",
        "split_state_dict_into_shards_factory",
        "split_tf_state_dict_into_shards",
        "split_torch_state_dict_into_shards",
    ],
    "serialization._dduf": [
        "DDUFEntry",
        "export_entries_as_dduf",
        "export_folder_as_dduf",
        "read_dduf_file",
    ],
    "utils": [
        "CacheNotFound",
        "CachedFileInfo",
        "CachedRepoInfo",
        "CachedRevisionInfo",
        "CorruptedCacheException",
        "DeleteCacheStrategy",
        "HFCacheInfo",
        "HfFolder",
        "cached_assets_path",
        "configure_http_backend",
        "dump_environment_info",
        "get_session",
        "get_token",
        "logging",
        "scan_cache_dir",
    ],
}

# WARNING: __all__ is generated automatically, Any manual edit will be lost when re-generating this file !
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_all_variable.py --update
#
# # Or run style on codebase
# make style
# ```

__all__ = [
    "Agent",
    "AsyncInferenceClient",
    "AudioClassificationInput",
    "AudioClassificationOutputElement",
    "AudioClassificationOutputTransform",
    "AudioClassificationParameters",
    "AudioToAudioInput",
    "AudioToAudioOutputElement",
    "AutomaticSpeechRecognitionEarlyStoppingEnum",
    "AutomaticSpeechRecognitionGenerationParameters",
    "AutomaticSpeechRecognitionInput",
    "AutomaticSpeechRecognitionOutput",
    "AutomaticSpeechRecognitionOutputChunk",
    "AutomaticSpeechRecognitionParameters",
    "CONFIG_NAME",
    "CacheNotFound",
    "CachedFileInfo",
    "CachedRepoInfo",
    "CachedRevisionInfo",
    "CardData",
    "ChatCompletionInput",
    "ChatCompletionInputFunctionDefinition",
    "ChatCompletionInputFunctionName",
    "ChatCompletionInputGrammarType",
    "ChatCompletionInputJSONSchema",
    "ChatCompletionInputMessage",
    "ChatCompletionInputMessageChunk",
    "ChatCompletionInputMessageChunkType",
    "ChatCompletionInputResponseFormatJSONObject",
    "ChatCompletionInputResponseFormatJSONSchema",
    "ChatCompletionInputResponseFormatText",
    "ChatCompletionInputStreamOptions",
    "ChatCompletionInputTool",
    "ChatCompletionInputToolCall",
    "ChatCompletionInputToolChoiceClass",
    "ChatCompletionInputToolChoiceEnum",
    "ChatCompletionInputURL",
    "ChatCompletionOutput",
    "ChatCompletionOutputComplete",
    "ChatCompletionOutputFunctionDefinition",
    "ChatCompletionOutputLogprob",
    "ChatCompletionOutputLogprobs",
    "ChatCompletionOutputMessage",
    "ChatCompletionOutputToolCall",
    "ChatCompletionOutputTopLogprob",
    "ChatCompletionOutputUsage",
    "ChatCompletionStreamOutput",
    "ChatCompletionStreamOutputChoice",
    "ChatCompletionStreamOutputDelta",
    "ChatCompletionStreamOutputDeltaToolCall",
    "ChatCompletionStreamOutputFunction",
    "ChatCompletionStreamOutputLogprob",
    "ChatCompletionStreamOutputLogprobs",
    "ChatCompletionStreamOutputTopLogprob",
    "ChatCompletionStreamOutputUsage",
    "Collection",
    "CollectionItem",
    "CommitInfo",
    "CommitOperation",
    "CommitOperationAdd",
    "CommitOperationCopy",
    "CommitOperationDelete",
    "CommitScheduler",
    "CorruptedCacheException",
    "DDUFEntry",
    "DatasetCard",
    "DatasetCardData",
    "DatasetInfo",
    "DeleteCacheStrategy",
    "DepthEstimationInput",
    "DepthEstimationOutput",
    "Discussion",
    "DiscussionComment",
    "DiscussionCommit",
    "DiscussionEvent",
    "DiscussionStatusChange",
    "DiscussionTitleChange",
    "DiscussionWithDetails",
    "DocumentQuestionAnsweringInput",
    "DocumentQuestionAnsweringInputData",
    "DocumentQuestionAnsweringOutputElement",
    "DocumentQuestionAnsweringParameters",
    "EvalResult",
    "FLAX_WEIGHTS_NAME",
    "FeatureExtractionInput",
    "FeatureExtractionInputTruncationDirection",
    "FillMaskInput",
    "FillMaskOutputElement",
    "FillMaskParameters",
    "GitCommitInfo",
    "GitRefInfo",
    "GitRefs",
    "HFCacheInfo",
    "HFSummaryWriter",
    "HUGGINGFACE_CO_URL_HOME",
    "HUGGINGFACE_CO_URL_TEMPLATE",
    "HfApi",
    "HfFileMetadata",
    "HfFileSystem",
    "HfFileSystemFile",
    "HfFileSystemResolvedPath",
    "HfFileSystemStreamFile",
    "HfFolder",
    "ImageClassificationInput",
    "ImageClassificationOutputElement",
    "ImageClassificationOutputTransform",
    "ImageClassificationParameters",
    "ImageSegmentationInput",
    "ImageSegmentationOutputElement",
    "ImageSegmentationParameters",
    "ImageSegmentationSubtask",
    "ImageToImageInput",
    "ImageToImageOutput",
    "ImageToImageParameters",
    "ImageToImageTargetSize",
    "ImageToTextEarlyStoppingEnum",
    "ImageToTextGenerationParameters",
    "ImageToTextInput",
    "ImageToTextOutput",
    "ImageToTextParameters",
    "ImageToVideoInput",
    "ImageToVideoOutput",
    "ImageToVideoParameters",
    "ImageToVideoTargetSize",
    "InferenceApi",
    "InferenceClient",
    "InferenceEndpoint",
    "InferenceEndpointError",
    "InferenceEndpointStatus",
    "InferenceEndpointTimeoutError",
    "InferenceEndpointType",
    "InferenceTimeoutError",
    "JobInfo",
    "JobOwner",
    "JobStage",
    "JobStatus",
    "KerasModelHubMixin",
    "MCPClient",
    "ModelCard",
    "ModelCardData",
    "ModelHubMixin",
    "ModelInfo",
    "OAuthInfo",
    "OAuthOrgInfo",
    "OAuthUserInfo",
    "ObjectDetectionBoundingBox",
    "ObjectDetectionInput",
    "ObjectDetectionOutputElement",
    "ObjectDetectionParameters",
    "PYTORCH_WEIGHTS_NAME",
    "Padding",
    "PyTorchModelHubMixin",
    "QuestionAnsweringInput",
    "QuestionAnsweringInputData",
    "QuestionAnsweringOutputElement",
    "QuestionAnsweringParameters",
    "REPO_TYPE_DATASET",
    "REPO_TYPE_MODEL",
    "REPO_TYPE_SPACE",
    "RepoCard",
    "RepoUrl",
    "Repository",
    "SentenceSimilarityInput",
    "SentenceSimilarityInputData",
    "SpaceCard",
    "SpaceCardData",
    "SpaceHardware",
    "SpaceInfo",
    "SpaceRuntime",
    "SpaceStage",
    "SpaceStorage",
    "SpaceVariable",
    "StateDictSplit",
    "SummarizationInput",
    "SummarizationOutput",
    "SummarizationParameters",
    "SummarizationTruncationStrategy",
    "TF2_WEIGHTS_NAME",
    "TF_WEIGHTS_NAME",
    "TableQuestionAnsweringInput",
    "TableQuestionAnsweringInputData",
    "TableQuestionAnsweringOutputElement",
    "TableQuestionAnsweringParameters",
    "Text2TextGenerationInput",
    "Text2TextGenerationOutput",
    "Text2TextGenerationParameters",
    "Text2TextGenerationTruncationStrategy",
    "TextClassificationInput",
    "TextClassificationOutputElement",
    "TextClassificationOutputTransform",
    "TextClassificationParameters",
    "TextGenerationInput",
    "TextGenerationInputGenerateParameters",
    "TextGenerationInputGrammarType",
    "TextGenerationOutput",
    "TextGenerationOutputBestOfSequence",
    "TextGenerationOutputDetails",
    "TextGenerationOutputFinishReason",
    "TextGenerationOutputPrefillToken",
    "TextGenerationOutputToken",
    "TextGenerationStreamOutput",
    "TextGenerationStreamOutputStreamDetails",
    "TextGenerationStreamOutputToken",
    "TextToAudioEarlyStoppingEnum",
    "TextToAudioGenerationParameters",
    "TextToAudioInput",
    "TextToAudioOutput",
    "TextToAudioParameters",
    "TextToImageInput",
    "TextToImageOutput",
    "TextToImageParameters",
    "TextToSpeechEarlyStoppingEnum",
    "TextToSpeechGenerationParameters",
    "TextToSpeechInput",
    "TextToSpeechOutput",
    "TextToSpeechParameters",
    "TextToVideoInput",
    "TextToVideoOutput",
    "TextToVideoParameters",
    "TokenClassificationAggregationStrategy",
    "TokenClassificationInput",
    "TokenClassificationOutputElement",
    "TokenClassificationParameters",
    "TranslationInput",
    "TranslationOutput",
    "TranslationParameters",
    "TranslationTruncationStrategy",
    "TypeEnum",
    "User",
    "UserLikes",
    "VideoClassificationInput",
    "VideoClassificationOutputElement",
    "VideoClassificationOutputTransform",
    "VideoClassificationParameters",
    "VisualQuestionAnsweringInput",
    "VisualQuestionAnsweringInputData",
    "VisualQuestionAnsweringOutputElement",
    "VisualQuestionAnsweringParameters",
    "WebhookInfo",
    "WebhookPayload",
    "WebhookPayloadComment",
    "WebhookPayloadDiscussion",
    "WebhookPayloadDiscussionChanges",
    "WebhookPayloadEvent",
    "WebhookPayloadMovedTo",
    "WebhookPayloadRepo",
    "WebhookPayloadUrl",
    "WebhookPayloadWebhook",
    "WebhookWatchedItem",
    "WebhooksServer",
    "ZeroShotClassificationInput",
    "ZeroShotClassificationOutputElement",
    "ZeroShotClassificationParameters",
    "ZeroShotImageClassificationInput",
    "ZeroShotImageClassificationOutputElement",
    "ZeroShotImageClassificationParameters",
    "ZeroShotObjectDetectionBoundingBox",
    "ZeroShotObjectDetectionInput",
    "ZeroShotObjectDetectionOutputElement",
    "ZeroShotObjectDetectionParameters",
    "_CACHED_NO_EXIST",
    "_save_pretrained_fastai",
    "accept_access_request",
    "add_collection_item",
    "add_space_secret",
    "add_space_variable",
    "attach_huggingface_oauth",
    "auth_check",
    "auth_list",
    "auth_switch",
    "cached_assets_path",
    "cancel_access_request",
    "cancel_job",
    "change_discussion_status",
    "comment_discussion",
    "configure_http_backend",
    "create_branch",
    "create_collection",
    "create_commit",
    "create_discussion",
    "create_inference_endpoint",
    "create_inference_endpoint_from_catalog",
    "create_pull_request",
    "create_repo",
    "create_tag",
    "create_webhook",
    "dataset_info",
    "delete_branch",
    "delete_collection",
    "delete_collection_item",
    "delete_file",
    "delete_folder",
    "delete_inference_endpoint",
    "delete_repo",
    "delete_space_secret",
    "delete_space_storage",
    "delete_space_variable",
    "delete_tag",
    "delete_webhook",
    "disable_webhook",
    "dump_environment_info",
    "duplicate_space",
    "edit_discussion_comment",
    "enable_webhook",
    "export_entries_as_dduf",
    "export_folder_as_dduf",
    "fetch_job_logs",
    "file_exists",
    "from_pretrained_fastai",
    "from_pretrained_keras",
    "get_collection",
    "get_dataset_tags",
    "get_discussion_details",
    "get_full_repo_name",
    "get_hf_file_metadata",
    "get_inference_endpoint",
    "get_model_tags",
    "get_paths_info",
    "get_repo_discussions",
    "get_safetensors_metadata",
    "get_session",
    "get_space_runtime",
    "get_space_variables",
    "get_tf_storage_size",
    "get_token",
    "get_token_permission",
    "get_torch_storage_id",
    "get_torch_storage_size",
    "get_user_overview",
    "get_webhook",
    "grant_access",
    "hf_hub_download",
    "hf_hub_url",
    "inspect_job",
    "interpreter_login",
    "list_accepted_access_requests",
    "list_collections",
    "list_datasets",
    "list_inference_catalog",
    "list_inference_endpoints",
    "list_jobs",
    "list_lfs_files",
    "list_liked_repos",
    "list_models",
    "list_organization_members",
    "list_papers",
    "list_pending_access_requests",
    "list_rejected_access_requests",
    "list_repo_commits",
    "list_repo_files",
    "list_repo_likers",
    "list_repo_refs",
    "list_repo_tree",
    "list_spaces",
    "list_user_followers",
    "list_user_following",
    "list_webhooks",
    "load_state_dict_from_file",
    "load_torch_model",
    "logging",
    "login",
    "logout",
    "merge_pull_request",
    "metadata_eval_result",
    "metadata_load",
    "metadata_save",
    "metadata_update",
    "model_info",
    "move_repo",
    "notebook_login",
    "paper_info",
    "parse_huggingface_oauth",
    "parse_safetensors_file_metadata",
    "pause_inference_endpoint",
    "pause_space",
    "permanently_delete_lfs_files",
    "preupload_lfs_files",
    "push_to_hub_fastai",
    "push_to_hub_keras",
    "read_dduf_file",
    "reject_access_request",
    "rename_discussion",
    "repo_exists",
    "repo_info",
    "repo_type_and_id_from_hf_id",
    "request_space_hardware",
    "request_space_storage",
    "restart_space",
    "resume_inference_endpoint",
    "revision_exists",
    "run_as_future",
    "run_job",
    "run_uv_job",
    "save_pretrained_keras",
    "save_torch_model",
    "save_torch_state_dict",
    "scale_to_zero_inference_endpoint",
    "scan_cache_dir",
    "set_space_sleep_time",
    "snapshot_download",
    "space_info",
    "split_state_dict_into_shards_factory",
    "split_tf_state_dict_into_shards",
    "split_torch_state_dict_into_shards",
    "super_squash_history",
    "try_to_load_from_cache",
    "unlike",
    "update_collection_item",
    "update_collection_metadata",
    "update_inference_endpoint",
    "update_repo_settings",
    "update_repo_visibility",
    "update_webhook",
    "upload_file",
    "upload_folder",
    "upload_large_folder",
    "webhook_endpoint",
    "whoami",
]


def _attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows:

    ```py
    import mysubmodule
    import anothersubmodule

    from .foo import someattr
    ```

    The idea is to replace a package's `__getattr__`, `__dir__`, such that all imports
    work exactly the way they would with normal imports, except that the import occurs
    upon first use.

    The typical way to call this function, replacing the above imports, is:

    ```python
    __getattr__, __dir__ = lazy.attach(
        __name__,
        ['mysubmodule', 'anothersubmodule'],
        {'foo': ['someattr']}
    )
    ```
    This functionality requires Python 3.7 or higher.

    Args:
        package_name (`str`):
            Typically use `__name__`.
        submodules (`set`):
            List of submodules to attach.
        submod_attrs (`dict`):
            Dictionary of submodule -> list of attributes / functions.
            These attributes are imported as they are used.

    Returns:
        __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    def __getattr__(name):
        if name in submodules:
            try:
                return importlib.import_module(f"{package_name}.{name}")
            except Exception as e:
                print(f"Error importing {package_name}.{name}: {e}")
                raise
        elif name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            try:
                submod = importlib.import_module(submod_path)
            except Exception as e:
                print(f"Error importing {submod_path}: {e}")
                raise
            attr = getattr(submod, name)

            # If the attribute lives in a file (module) with the same
            # name as the attribute, ensure that the attribute and *not*
            # the module is accessible on the package.
            if name == attr_to_modules[name]:
                pkg = sys.modules[package_name]
                pkg.__dict__[name] = attr

            return attr
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__

    return __getattr__, __dir__


__getattr__, __dir__ = _attach(__name__, submodules=[], submod_attrs=_SUBMOD_ATTRS)

if os.environ.get("EAGER_IMPORT", ""):
    for attr in __all__:
        __getattr__(attr)

# WARNING: any content below this statement is generated automatically. Any manual edit
# will be lost when re-generating this file !
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_static_imports.py --update
#
# # Or run style on codebase
# make style
# ```
if TYPE_CHECKING:  # pragma: no cover
    from ._commit_scheduler import CommitScheduler  # noqa: F401
    from ._inference_endpoints import (
        InferenceEndpoint,  # noqa: F401
        InferenceEndpointError,  # noqa: F401
        InferenceEndpointStatus,  # noqa: F401
        InferenceEndpointTimeoutError,  # noqa: F401
        InferenceEndpointType,  # noqa: F401
    )
    from ._jobs_api import (
        JobInfo,  # noqa: F401
        JobOwner,  # noqa: F401
        JobStage,  # noqa: F401
        JobStatus,  # noqa: F401
    )
    from ._login import (
        auth_list,  # noqa: F401
        auth_switch,  # noqa: F401
        interpreter_login,  # noqa: F401
        login,  # noqa: F401
        logout,  # noqa: F401
        notebook_login,  # noqa: F401
    )
    from ._oauth import (
        OAuthInfo,  # noqa: F401
        OAuthOrgInfo,  # noqa: F401
        OAuthUserInfo,  # noqa: F401
        attach_huggingface_oauth,  # noqa: F401
        parse_huggingface_oauth,  # noqa: F401
    )
    from ._snapshot_download import snapshot_download  # noqa: F401
    from ._space_api import (
        SpaceHardware,  # noqa: F401
        SpaceRuntime,  # noqa: F401
        SpaceStage,  # noqa: F401
        SpaceStorage,  # noqa: F401
        SpaceVariable,  # noqa: F401
    )
    from ._tensorboard_logger import HFSummaryWriter  # noqa: F401
    from ._webhooks_payload import (
        WebhookPayload,  # noqa: F401
        WebhookPayloadComment,  # noqa: F401
        WebhookPayloadDiscussion,  # noqa: F401
        WebhookPayloadDiscussionChanges,  # noqa: F401
        WebhookPayloadEvent,  # noqa: F401
        WebhookPayloadMovedTo,  # noqa: F401
        WebhookPayloadRepo,  # noqa: F401
        WebhookPayloadUrl,  # noqa: F401
        WebhookPayloadWebhook,  # noqa: F401
    )
    from ._webhooks_server import (
        WebhooksServer,  # noqa: F401
        webhook_endpoint,  # noqa: F401
    )
    from .community import (
        Discussion,  # noqa: F401
        DiscussionComment,  # noqa: F401
        DiscussionCommit,  # noqa: F401
        DiscussionEvent,  # noqa: F401
        DiscussionStatusChange,  # noqa: F401
        DiscussionTitleChange,  # noqa: F401
        DiscussionWithDetails,  # noqa: F401
    )
    from .constants import (
        CONFIG_NAME,  # noqa: F401
        FLAX_WEIGHTS_NAME,  # noqa: F401
        HUGGINGFACE_CO_URL_HOME,  # noqa: F401
        HUGGINGFACE_CO_URL_TEMPLATE,  # noqa: F401
        PYTORCH_WEIGHTS_NAME,  # noqa: F401
        REPO_TYPE_DATASET,  # noqa: F401
        REPO_TYPE_MODEL,  # noqa: F401
        REPO_TYPE_SPACE,  # noqa: F401
        TF2_WEIGHTS_NAME,  # noqa: F401
        TF_WEIGHTS_NAME,  # noqa: F401
    )
    from .fastai_utils import (
        _save_pretrained_fastai,  # noqa: F401
        from_pretrained_fastai,  # noqa: F401
        push_to_hub_fastai,  # noqa: F401
    )
    from .file_download import (
        _CACHED_NO_EXIST,  # noqa: F401
        HfFileMetadata,  # noqa: F401
        get_hf_file_metadata,  # noqa: F401
        hf_hub_download,  # noqa: F401
        hf_hub_url,  # noqa: F401
        try_to_load_from_cache,  # noqa: F401
    )
    from .hf_api import (
        Collection,  # noqa: F401
        CollectionItem,  # noqa: F401
        CommitInfo,  # noqa: F401
        CommitOperation,  # noqa: F401
        CommitOperationAdd,  # noqa: F401
        CommitOperationCopy,  # noqa: F401
        CommitOperationDelete,  # noqa: F401
        DatasetInfo,  # noqa: F401
        GitCommitInfo,  # noqa: F401
        GitRefInfo,  # noqa: F401
        GitRefs,  # noqa: F401
        HfApi,  # noqa: F401
        ModelInfo,  # noqa: F401
        RepoUrl,  # noqa: F401
        SpaceInfo,  # noqa: F401
        User,  # noqa: F401
        UserLikes,  # noqa: F401
        WebhookInfo,  # noqa: F401
        WebhookWatchedItem,  # noqa: F401
        accept_access_request,  # noqa: F401
        add_collection_item,  # noqa: F401
        add_space_secret,  # noqa: F401
        add_space_variable,  # noqa: F401
        auth_check,  # noqa: F401
        cancel_access_request,  # noqa: F401
        cancel_job,  # noqa: F401
        change_discussion_status,  # noqa: F401
        comment_discussion,  # noqa: F401
        create_branch,  # noqa: F401
        create_collection,  # noqa: F401
        create_commit,  # noqa: F401
        create_discussion,  # noqa: F401
        create_inference_endpoint,  # noqa: F401
        create_inference_endpoint_from_catalog,  # noqa: F401
        create_pull_request,  # noqa: F401
        create_repo,  # noqa: F401
        create_tag,  # noqa: F401
        create_webhook,  # noqa: F401
        dataset_info,  # noqa: F401
        delete_branch,  # noqa: F401
        delete_collection,  # noqa: F401
        delete_collection_item,  # noqa: F401
        delete_file,  # noqa: F401
        delete_folder,  # noqa: F401
        delete_inference_endpoint,  # noqa: F401
        delete_repo,  # noqa: F401
        delete_space_secret,  # noqa: F401
        delete_space_storage,  # noqa: F401
        delete_space_variable,  # noqa: F401
        delete_tag,  # noqa: F401
        delete_webhook,  # noqa: F401
        disable_webhook,  # noqa: F401
        duplicate_space,  # noqa: F401
        edit_discussion_comment,  # noqa: F401
        enable_webhook,  # noqa: F401
        fetch_job_logs,  # noqa: F401
        file_exists,  # noqa: F401
        get_collection,  # noqa: F401
        get_dataset_tags,  # noqa: F401
        get_discussion_details,  # noqa: F401
        get_full_repo_name,  # noqa: F401
        get_inference_endpoint,  # noqa: F401
        get_model_tags,  # noqa: F401
        get_paths_info,  # noqa: F401
        get_repo_discussions,  # noqa: F401
        get_safetensors_metadata,  # noqa: F401
        get_space_runtime,  # noqa: F401
        get_space_variables,  # noqa: F401
        get_token_permission,  # noqa: F401
        get_user_overview,  # noqa: F401
        get_webhook,  # noqa: F401
        grant_access,  # noqa: F401
        inspect_job,  # noqa: F401
        list_accepted_access_requests,  # noqa: F401
        list_collections,  # noqa: F401
        list_datasets,  # noqa: F401
        list_inference_catalog,  # noqa: F401
        list_inference_endpoints,  # noqa: F401
        list_jobs,  # noqa: F401
        list_lfs_files,  # noqa: F401
        list_liked_repos,  # noqa: F401
        list_models,  # noqa: F401
        list_organization_members,  # noqa: F401
        list_papers,  # noqa: F401
        list_pending_access_requests,  # noqa: F401
        list_rejected_access_requests,  # noqa: F401
        list_repo_commits,  # noqa: F401
        list_repo_files,  # noqa: F401
        list_repo_likers,  # noqa: F401
        list_repo_refs,  # noqa: F401
        list_repo_tree,  # noqa: F401
        list_spaces,  # noqa: F401
        list_user_followers,  # noqa: F401
        list_user_following,  # noqa: F401
        list_webhooks,  # noqa: F401
        merge_pull_request,  # noqa: F401
        model_info,  # noqa: F401
        move_repo,  # noqa: F401
        paper_info,  # noqa: F401
        parse_safetensors_file_metadata,  # noqa: F401
        pause_inference_endpoint,  # noqa: F401
        pause_space,  # noqa: F401
        permanently_delete_lfs_files,  # noqa: F401
        preupload_lfs_files,  # noqa: F401
        reject_access_request,  # noqa: F401
        rename_discussion,  # noqa: F401
        repo_exists,  # noqa: F401
        repo_info,  # noqa: F401
        repo_type_and_id_from_hf_id,  # noqa: F401
        request_space_hardware,  # noqa: F401
        request_space_storage,  # noqa: F401
        restart_space,  # noqa: F401
        resume_inference_endpoint,  # noqa: F401
        revision_exists,  # noqa: F401
        run_as_future,  # noqa: F401
        run_job,  # noqa: F401
        run_uv_job,  # noqa: F401
        scale_to_zero_inference_endpoint,  # noqa: F401
        set_space_sleep_time,  # noqa: F401
        space_info,  # noqa: F401
        super_squash_history,  # noqa: F401
        unlike,  # noqa: F401
        update_collection_item,  # noqa: F401
        update_collection_metadata,  # noqa: F401
        update_inference_endpoint,  # noqa: F401
        update_repo_settings,  # noqa: F401
        update_repo_visibility,  # noqa: F401
        update_webhook,  # noqa: F401
        upload_file,  # noqa: F401
        upload_folder,  # noqa: F401
        upload_large_folder,  # noqa: F401
        whoami,  # noqa: F401
    )
    from .hf_file_system import (
        HfFileSystem,  # noqa: F401
        HfFileSystemFile,  # noqa: F401
        HfFileSystemResolvedPath,  # noqa: F401
        HfFileSystemStreamFile,  # noqa: F401
    )
    from .hub_mixin import (
        ModelHubMixin,  # noqa: F401
        PyTorchModelHubMixin,  # noqa: F401
    )
    from .inference._client import (
        InferenceClient,  # noqa: F401
        InferenceTimeoutError,  # noqa: F401
    )
    from .inference._generated._async_client import AsyncInferenceClient  # noqa: F401
    from .inference._generated.types import (
        AudioClassificationInput,  # noqa: F401
        AudioClassificationOutputElement,  # noqa: F401
        AudioClassificationOutputTransform,  # noqa: F401
        AudioClassificationParameters,  # noqa: F401
        AudioToAudioInput,  # noqa: F401
        AudioToAudioOutputElement,  # noqa: F401
        AutomaticSpeechRecognitionEarlyStoppingEnum,  # noqa: F401
        AutomaticSpeechRecognitionGenerationParameters,  # noqa: F401
        AutomaticSpeechRecognitionInput,  # noqa: F401
        AutomaticSpeechRecognitionOutput,  # noqa: F401
        AutomaticSpeechRecognitionOutputChunk,  # noqa: F401
        AutomaticSpeechRecognitionParameters,  # noqa: F401
        ChatCompletionInput,  # noqa: F401
        ChatCompletionInputFunctionDefinition,  # noqa: F401
        ChatCompletionInputFunctionName,  # noqa: F401
        ChatCompletionInputGrammarType,  # noqa: F401
        ChatCompletionInputJSONSchema,  # noqa: F401
        ChatCompletionInputMessage,  # noqa: F401
        ChatCompletionInputMessageChunk,  # noqa: F401
        ChatCompletionInputMessageChunkType,  # noqa: F401
        ChatCompletionInputResponseFormatJSONObject,  # noqa: F401
        ChatCompletionInputResponseFormatJSONSchema,  # noqa: F401
        ChatCompletionInputResponseFormatText,  # noqa: F401
        ChatCompletionInputStreamOptions,  # noqa: F401
        ChatCompletionInputTool,  # noqa: F401
        ChatCompletionInputToolCall,  # noqa: F401
        ChatCompletionInputToolChoiceClass,  # noqa: F401
        ChatCompletionInputToolChoiceEnum,  # noqa: F401
        ChatCompletionInputURL,  # noqa: F401
        ChatCompletionOutput,  # noqa: F401
        ChatCompletionOutputComplete,  # noqa: F401
        ChatCompletionOutputFunctionDefinition,  # noqa: F401
        ChatCompletionOutputLogprob,  # noqa: F401
        ChatCompletionOutputLogprobs,  # noqa: F401
        ChatCompletionOutputMessage,  # noqa: F401
        ChatCompletionOutputToolCall,  # noqa: F401
        ChatCompletionOutputTopLogprob,  # noqa: F401
        ChatCompletionOutputUsage,  # noqa: F401
        ChatCompletionStreamOutput,  # noqa: F401
        ChatCompletionStreamOutputChoice,  # noqa: F401
        ChatCompletionStreamOutputDelta,  # noqa: F401
        ChatCompletionStreamOutputDeltaToolCall,  # noqa: F401
        ChatCompletionStreamOutputFunction,  # noqa: F401
        ChatCompletionStreamOutputLogprob,  # noqa: F401
        ChatCompletionStreamOutputLogprobs,  # noqa: F401
        ChatCompletionStreamOutputTopLogprob,  # noqa: F401
        ChatCompletionStreamOutputUsage,  # noqa: F401
        DepthEstimationInput,  # noqa: F401
        DepthEstimationOutput,  # noqa: F401
        DocumentQuestionAnsweringInput,  # noqa: F401
        DocumentQuestionAnsweringInputData,  # noqa: F401
        DocumentQuestionAnsweringOutputElement,  # noqa: F401
        DocumentQuestionAnsweringParameters,  # noqa: F401
        FeatureExtractionInput,  # noqa: F401
        FeatureExtractionInputTruncationDirection,  # noqa: F401
        FillMaskInput,  # noqa: F401
        FillMaskOutputElement,  # noqa: F401
        FillMaskParameters,  # noqa: F401
        ImageClassificationInput,  # noqa: F401
        ImageClassificationOutputElement,  # noqa: F401
        ImageClassificationOutputTransform,  # noqa: F401
        ImageClassificationParameters,  # noqa: F401
        ImageSegmentationInput,  # noqa: F401
        ImageSegmentationOutputElement,  # noqa: F401
        ImageSegmentationParameters,  # noqa: F401
        ImageSegmentationSubtask,  # noqa: F401
        ImageToImageInput,  # noqa: F401
        ImageToImageOutput,  # noqa: F401
        ImageToImageParameters,  # noqa: F401
        ImageToImageTargetSize,  # noqa: F401
        ImageToTextEarlyStoppingEnum,  # noqa: F401
        ImageToTextGenerationParameters,  # noqa: F401
        ImageToTextInput,  # noqa: F401
        ImageToTextOutput,  # noqa: F401
        ImageToTextParameters,  # noqa: F401
        ImageToVideoInput,  # noqa: F401
        ImageToVideoOutput,  # noqa: F401
        ImageToVideoParameters,  # noqa: F401
        ImageToVideoTargetSize,  # noqa: F401
        ObjectDetectionBoundingBox,  # noqa: F401
        ObjectDetectionInput,  # noqa: F401
        ObjectDetectionOutputElement,  # noqa: F401
        ObjectDetectionParameters,  # noqa: F401
        Padding,  # noqa: F401
        QuestionAnsweringInput,  # noqa: F401
        QuestionAnsweringInputData,  # noqa: F401
        QuestionAnsweringOutputElement,  # noqa: F401
        QuestionAnsweringParameters,  # noqa: F401
        SentenceSimilarityInput,  # noqa: F401
        SentenceSimilarityInputData,  # noqa: F401
        SummarizationInput,  # noqa: F401
        SummarizationOutput,  # noqa: F401
        SummarizationParameters,  # noqa: F401
        SummarizationTruncationStrategy,  # noqa: F401
        TableQuestionAnsweringInput,  # noqa: F401
        TableQuestionAnsweringInputData,  # noqa: F401
        TableQuestionAnsweringOutputElement,  # noqa: F401
        TableQuestionAnsweringParameters,  # noqa: F401
        Text2TextGenerationInput,  # noqa: F401
        Text2TextGenerationOutput,  # noqa: F401
        Text2TextGenerationParameters,  # noqa: F401
        Text2TextGenerationTruncationStrategy,  # noqa: F401
        TextClassificationInput,  # noqa: F401
        TextClassificationOutputElement,  # noqa: F401
        TextClassificationOutputTransform,  # noqa: F401
        TextClassificationParameters,  # noqa: F401
        TextGenerationInput,  # noqa: F401
        TextGenerationInputGenerateParameters,  # noqa: F401
        TextGenerationInputGrammarType,  # noqa: F401
        TextGenerationOutput,  # noqa: F401
        TextGenerationOutputBestOfSequence,  # noqa: F401
        TextGenerationOutputDetails,  # noqa: F401
        TextGenerationOutputFinishReason,  # noqa: F401
        TextGenerationOutputPrefillToken,  # noqa: F401
        TextGenerationOutputToken,  # noqa: F401
        TextGenerationStreamOutput,  # noqa: F401
        TextGenerationStreamOutputStreamDetails,  # noqa: F401
        TextGenerationStreamOutputToken,  # noqa: F401
        TextToAudioEarlyStoppingEnum,  # noqa: F401
        TextToAudioGenerationParameters,  # noqa: F401
        TextToAudioInput,  # noqa: F401
        TextToAudioOutput,  # noqa: F401
        TextToAudioParameters,  # noqa: F401
        TextToImageInput,  # noqa: F401
        TextToImageOutput,  # noqa: F401
        TextToImageParameters,  # noqa: F401
        TextToSpeechEarlyStoppingEnum,  # noqa: F401
        TextToSpeechGenerationParameters,  # noqa: F401
        TextToSpeechInput,  # noqa: F401
        TextToSpeechOutput,  # noqa: F401
        TextToSpeechParameters,  # noqa: F401
        TextToVideoInput,  # noqa: F401
        TextToVideoOutput,  # noqa: F401
        TextToVideoParameters,  # noqa: F401
        TokenClassificationAggregationStrategy,  # noqa: F401
        TokenClassificationInput,  # noqa: F401
        TokenClassificationOutputElement,  # noqa: F401
        TokenClassificationParameters,  # noqa: F401
        TranslationInput,  # noqa: F401
        TranslationOutput,  # noqa: F401
        TranslationParameters,  # noqa: F401
        TranslationTruncationStrategy,  # noqa: F401
        TypeEnum,  # noqa: F401
        VideoClassificationInput,  # noqa: F401
        VideoClassificationOutputElement,  # noqa: F401
        VideoClassificationOutputTransform,  # noqa: F401
        VideoClassificationParameters,  # noqa: F401
        VisualQuestionAnsweringInput,  # noqa: F401
        VisualQuestionAnsweringInputData,  # noqa: F401
        VisualQuestionAnsweringOutputElement,  # noqa: F401
        VisualQuestionAnsweringParameters,  # noqa: F401
        ZeroShotClassificationInput,  # noqa: F401
        ZeroShotClassificationOutputElement,  # noqa: F401
        ZeroShotClassificationParameters,  # noqa: F401
        ZeroShotImageClassificationInput,  # noqa: F401
        ZeroShotImageClassificationOutputElement,  # noqa: F401
        ZeroShotImageClassificationParameters,  # noqa: F401
        ZeroShotObjectDetectionBoundingBox,  # noqa: F401
        ZeroShotObjectDetectionInput,  # noqa: F401
        ZeroShotObjectDetectionOutputElement,  # noqa: F401
        ZeroShotObjectDetectionParameters,  # noqa: F401
    )
    from .inference._mcp.agent import Agent  # noqa: F401
    from .inference._mcp.mcp_client import MCPClient  # noqa: F401
    from .inference_api import InferenceApi  # noqa: F401
    from .keras_mixin import (
        KerasModelHubMixin,  # noqa: F401
        from_pretrained_keras,  # noqa: F401
        push_to_hub_keras,  # noqa: F401
        save_pretrained_keras,  # noqa: F401
    )
    from .repocard import (
        DatasetCard,  # noqa: F401
        ModelCard,  # noqa: F401
        RepoCard,  # noqa: F401
        SpaceCard,  # noqa: F401
        metadata_eval_result,  # noqa: F401
        metadata_load,  # noqa: F401
        metadata_save,  # noqa: F401
        metadata_update,  # noqa: F401
    )
    from .repocard_data import (
        CardData,  # noqa: F401
        DatasetCardData,  # noqa: F401
        EvalResult,  # noqa: F401
        ModelCardData,  # noqa: F401
        SpaceCardData,  # noqa: F401
    )
    from .repository import Repository  # noqa: F401
    from .serialization import (
        StateDictSplit,  # noqa: F401
        get_tf_storage_size,  # noqa: F401
        get_torch_storage_id,  # noqa: F401
        get_torch_storage_size,  # noqa: F401
        load_state_dict_from_file,  # noqa: F401
        load_torch_model,  # noqa: F401
        save_torch_model,  # noqa: F401
        save_torch_state_dict,  # noqa: F401
        split_state_dict_into_shards_factory,  # noqa: F401
        split_tf_state_dict_into_shards,  # noqa: F401
        split_torch_state_dict_into_shards,  # noqa: F401
    )
    from .serialization._dduf import (
        DDUFEntry,  # noqa: F401
        export_entries_as_dduf,  # noqa: F401
        export_folder_as_dduf,  # noqa: F401
        read_dduf_file,  # noqa: F401
    )
    from .utils import (
        CachedFileInfo,  # noqa: F401
        CachedRepoInfo,  # noqa: F401
        CachedRevisionInfo,  # noqa: F401
        CacheNotFound,  # noqa: F401
        CorruptedCacheException,  # noqa: F401
        DeleteCacheStrategy,  # noqa: F401
        HFCacheInfo,  # noqa: F401
        HfFolder,  # noqa: F401
        cached_assets_path,  # noqa: F401
        configure_http_backend,  # noqa: F401
        dump_environment_info,  # noqa: F401
        get_session,  # noqa: F401
        get_token,  # noqa: F401
        logging,  # noqa: F401
        scan_cache_dir,  # noqa: F401
    )
