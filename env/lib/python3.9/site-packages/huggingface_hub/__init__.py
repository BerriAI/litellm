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


__version__ = "0.16.4"

# Alphabetical order of definitions is ensured in tests
# WARNING: any comment added in this dictionary definition will be lost when
# re-generating the file !
_SUBMOD_ATTRS = {
    "_commit_scheduler": [
        "CommitScheduler",
    ],
    "_login": [
        "interpreter_login",
        "login",
        "logout",
        "notebook_login",
    ],
    "_multi_commits": [
        "MultiCommitException",
        "plan_multi_commits",
    ],
    "_snapshot_download": [
        "snapshot_download",
    ],
    "_space_api": [
        "SpaceHardware",
        "SpaceRuntime",
        "SpaceStage",
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
        "cached_download",
        "get_hf_file_metadata",
        "hf_hub_download",
        "hf_hub_url",
        "try_to_load_from_cache",
    ],
    "hf_api": [
        "CommitInfo",
        "CommitOperation",
        "CommitOperationAdd",
        "CommitOperationCopy",
        "CommitOperationDelete",
        "DatasetSearchArguments",
        "GitCommitInfo",
        "GitRefInfo",
        "GitRefs",
        "HfApi",
        "ModelSearchArguments",
        "RepoUrl",
        "UserLikes",
        "add_space_secret",
        "change_discussion_status",
        "comment_discussion",
        "create_branch",
        "create_commit",
        "create_commits_on_pr",
        "create_discussion",
        "create_pull_request",
        "create_repo",
        "create_tag",
        "dataset_info",
        "delete_branch",
        "delete_file",
        "delete_folder",
        "delete_repo",
        "delete_space_secret",
        "delete_tag",
        "duplicate_space",
        "edit_discussion_comment",
        "get_dataset_tags",
        "get_discussion_details",
        "get_full_repo_name",
        "get_model_tags",
        "get_repo_discussions",
        "get_space_runtime",
        "get_token_permission",
        "like",
        "list_datasets",
        "list_files_info",
        "list_liked_repos",
        "list_metrics",
        "list_models",
        "list_repo_commits",
        "list_repo_files",
        "list_repo_refs",
        "list_spaces",
        "merge_pull_request",
        "model_info",
        "move_repo",
        "pause_space",
        "rename_discussion",
        "repo_info",
        "repo_type_and_id_from_hf_id",
        "request_space_hardware",
        "restart_space",
        "run_as_future",
        "set_space_sleep_time",
        "space_info",
        "unlike",
        "update_repo_visibility",
        "upload_file",
        "upload_folder",
        "whoami",
    ],
    "hf_file_system": [
        "HfFileSystem",
        "HfFileSystemFile",
        "HfFileSystemResolvedPath",
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
        "logging",
        "scan_cache_dir",
    ],
    "utils.endpoint_helpers": [
        "DatasetFilter",
        "ModelFilter",
    ],
}


def _attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows:

    ```py
    import mysubmodule
    import anothersubmodule

    from .foo import someattr
    ```

    The idea is to replace a package's `__getattr__`, `__dir__`, and
    `__all__`, such that all imports work exactly the way they would
    with normal imports, except that the import occurs upon first use.

    The typical way to call this function, replacing the above imports, is:

    ```python
    __getattr__, __dir__, __all__ = lazy.attach(
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

    attr_to_modules = {attr: mod for mod, attrs in submod_attrs.items() for attr in attrs}

    __all__ = list(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        elif name in attr_to_modules:
            submod_path = f"{package_name}.{attr_to_modules[name]}"
            submod = importlib.import_module(submod_path)
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

    if os.environ.get("EAGER_IMPORT", ""):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


__getattr__, __dir__, __all__ = _attach(__name__, submodules=[], submod_attrs=_SUBMOD_ATTRS)

# WARNING: any content below this statement is generated automatically. Any manual edit
# will be lost when re-generating this file !
#
# To update the static imports, please run the following command and commit the changes.
# ```
# # Use script
# python utils/check_static_imports.py --update-file
#
# # Or run style on codebase
# make style
# ```
if TYPE_CHECKING:  # pragma: no cover
    from ._commit_scheduler import CommitScheduler  # noqa: F401
    from ._login import (
        interpreter_login,  # noqa: F401
        login,  # noqa: F401
        logout,  # noqa: F401
        notebook_login,  # noqa: F401
    )
    from ._multi_commits import (
        MultiCommitException,  # noqa: F401
        plan_multi_commits,  # noqa: F401
    )
    from ._snapshot_download import snapshot_download  # noqa: F401
    from ._space_api import (
        SpaceHardware,  # noqa: F401
        SpaceRuntime,  # noqa: F401
        SpaceStage,  # noqa: F401
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
        cached_download,  # noqa: F401
        get_hf_file_metadata,  # noqa: F401
        hf_hub_download,  # noqa: F401
        hf_hub_url,  # noqa: F401
        try_to_load_from_cache,  # noqa: F401
    )
    from .hf_api import (
        CommitInfo,  # noqa: F401
        CommitOperation,  # noqa: F401
        CommitOperationAdd,  # noqa: F401
        CommitOperationCopy,  # noqa: F401
        CommitOperationDelete,  # noqa: F401
        DatasetSearchArguments,  # noqa: F401
        GitCommitInfo,  # noqa: F401
        GitRefInfo,  # noqa: F401
        GitRefs,  # noqa: F401
        HfApi,  # noqa: F401
        ModelSearchArguments,  # noqa: F401
        RepoUrl,  # noqa: F401
        UserLikes,  # noqa: F401
        add_space_secret,  # noqa: F401
        change_discussion_status,  # noqa: F401
        comment_discussion,  # noqa: F401
        create_branch,  # noqa: F401
        create_commit,  # noqa: F401
        create_commits_on_pr,  # noqa: F401
        create_discussion,  # noqa: F401
        create_pull_request,  # noqa: F401
        create_repo,  # noqa: F401
        create_tag,  # noqa: F401
        dataset_info,  # noqa: F401
        delete_branch,  # noqa: F401
        delete_file,  # noqa: F401
        delete_folder,  # noqa: F401
        delete_repo,  # noqa: F401
        delete_space_secret,  # noqa: F401
        delete_tag,  # noqa: F401
        duplicate_space,  # noqa: F401
        edit_discussion_comment,  # noqa: F401
        get_dataset_tags,  # noqa: F401
        get_discussion_details,  # noqa: F401
        get_full_repo_name,  # noqa: F401
        get_model_tags,  # noqa: F401
        get_repo_discussions,  # noqa: F401
        get_space_runtime,  # noqa: F401
        get_token_permission,  # noqa: F401
        like,  # noqa: F401
        list_datasets,  # noqa: F401
        list_files_info,  # noqa: F401
        list_liked_repos,  # noqa: F401
        list_metrics,  # noqa: F401
        list_models,  # noqa: F401
        list_repo_commits,  # noqa: F401
        list_repo_files,  # noqa: F401
        list_repo_refs,  # noqa: F401
        list_spaces,  # noqa: F401
        merge_pull_request,  # noqa: F401
        model_info,  # noqa: F401
        move_repo,  # noqa: F401
        pause_space,  # noqa: F401
        rename_discussion,  # noqa: F401
        repo_info,  # noqa: F401
        repo_type_and_id_from_hf_id,  # noqa: F401
        request_space_hardware,  # noqa: F401
        restart_space,  # noqa: F401
        run_as_future,  # noqa: F401
        set_space_sleep_time,  # noqa: F401
        space_info,  # noqa: F401
        unlike,  # noqa: F401
        update_repo_visibility,  # noqa: F401
        upload_file,  # noqa: F401
        upload_folder,  # noqa: F401
        whoami,  # noqa: F401
    )
    from .hf_file_system import (
        HfFileSystem,  # noqa: F401
        HfFileSystemFile,  # noqa: F401
        HfFileSystemResolvedPath,  # noqa: F401
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
        logging,  # noqa: F401
        scan_cache_dir,  # noqa: F401
    )
    from .utils.endpoint_helpers import (
        DatasetFilter,  # noqa: F401
        ModelFilter,  # noqa: F401
    )
