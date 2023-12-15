# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
from __future__ import annotations

import inspect
import json
import pprint
import re
import textwrap
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union, overload
from urllib.parse import quote

import requests
from requests.exceptions import HTTPError

from huggingface_hub.utils import (
    IGNORE_GIT_FOLDER_PATTERNS,
    EntryNotFoundError,
    LocalTokenNotFoundError,
    RepositoryNotFoundError,
    experimental,
    get_session,
)

from ._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    fetch_lfs_files_to_copy,
    fetch_upload_modes,
    prepare_commit_payload,
    upload_lfs_files,
    warn_on_overwriting_operations,
)
from ._multi_commits import (
    MULTI_COMMIT_PR_CLOSE_COMMENT_FAILURE_BAD_REQUEST_TEMPLATE,
    MULTI_COMMIT_PR_CLOSE_COMMENT_FAILURE_NO_CHANGES_TEMPLATE,
    MULTI_COMMIT_PR_CLOSING_COMMENT_TEMPLATE,
    MULTI_COMMIT_PR_COMPLETION_COMMENT_TEMPLATE,
    MultiCommitException,
    MultiCommitStep,
    MultiCommitStrategy,
    multi_commit_create_pull_request,
    multi_commit_generate_comment,
    multi_commit_parse_pr_description,
    plan_multi_commits,
)
from ._space_api import SpaceHardware, SpaceRuntime
from .community import (
    Discussion,
    DiscussionComment,
    DiscussionStatusChange,
    DiscussionTitleChange,
    DiscussionWithDetails,
    deserialize_event,
)
from .constants import (
    DEFAULT_REVISION,
    ENDPOINT,
    REGEX_COMMIT_OID,
    REPO_TYPE_MODEL,
    REPO_TYPES,
    REPO_TYPES_MAPPING,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
)
from .utils import (  # noqa: F401 # imported for backward compatibility
    BadRequestError,
    HfFolder,
    HfHubHTTPError,
    build_hf_headers,
    filter_repo_objects,
    hf_raise_for_status,
    logging,
    paginate,
    parse_datetime,
    validate_hf_hub_args,
)
from .utils._deprecation import (
    _deprecate_arguments,
)
from .utils._typing import CallableT, Literal, TypedDict
from .utils.endpoint_helpers import (
    AttributeDictionary,
    DatasetFilter,
    DatasetTags,
    ModelFilter,
    ModelTags,
    _filter_emissions,
)


R = TypeVar("R")  # Return type

USERNAME_PLACEHOLDER = "hf_user"
_REGEX_DISCUSSION_URL = re.compile(r".*/discussions/(\d+)$")


logger = logging.get_logger(__name__)


class ReprMixin:
    """Mixin to create the __repr__ for a class"""

    def __repr__(self):
        formatted_value = pprint.pformat(self.__dict__, width=119, compact=True)
        if "\n" in formatted_value:
            return f"{self.__class__.__name__}: {{ \n{textwrap.indent(formatted_value, '  ')}\n}}"
        else:
            return f"{self.__class__.__name__}: {formatted_value}"


def repo_type_and_id_from_hf_id(hf_id: str, hub_url: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns the repo type and ID from a huggingface.co URL linking to a
    repository

    Args:
        hf_id (`str`):
            An URL or ID of a repository on the HF hub. Accepted values are:

            - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
            - https://huggingface.co/<namespace>/<repo_id>
            - hf://<repo_type>/<namespace>/<repo_id>
            - hf://<namespace>/<repo_id>
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
        hub_url (`str`, *optional*):
            The URL of the HuggingFace Hub, defaults to https://huggingface.co

    Returns:
        A tuple with three items: repo_type (`str` or `None`), namespace (`str` or
        `None`) and repo_id (`str`).

    Raises:
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If URL cannot be parsed.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `repo_type` is unknown.
    """
    input_hf_id = hf_id
    hub_url = re.sub(r"https?://", "", hub_url if hub_url is not None else ENDPOINT)
    is_hf_url = hub_url in hf_id and "@" not in hf_id

    HFFS_PREFIX = "hf://"
    if hf_id.startswith(HFFS_PREFIX):  # Remove "hf://" prefix if exists
        hf_id = hf_id[len(HFFS_PREFIX) :]

    url_segments = hf_id.split("/")
    is_hf_id = len(url_segments) <= 3

    namespace: Optional[str]
    if is_hf_url:
        namespace, repo_id = url_segments[-2:]
        if namespace == hub_url:
            namespace = None
        if len(url_segments) > 2 and hub_url not in url_segments[-3]:
            repo_type = url_segments[-3]
        elif namespace in REPO_TYPES_MAPPING:
            # Mean canonical dataset or model
            repo_type = REPO_TYPES_MAPPING[namespace]
            namespace = None
        else:
            repo_type = None
    elif is_hf_id:
        if len(url_segments) == 3:
            # Passed <repo_type>/<user>/<model_id> or <repo_type>/<org>/<model_id>
            repo_type, namespace, repo_id = url_segments[-3:]
        elif len(url_segments) == 2:
            if url_segments[0] in REPO_TYPES_MAPPING:
                # Passed '<model_id>' or 'datasets/<dataset_id>' for a canonical model or dataset
                repo_type = REPO_TYPES_MAPPING[url_segments[0]]
                namespace = None
                repo_id = hf_id.split("/")[-1]
            else:
                # Passed <user>/<model_id> or <org>/<model_id>
                namespace, repo_id = hf_id.split("/")[-2:]
                repo_type = None
        else:
            # Passed <model_id>
            repo_id = url_segments[0]
            namespace, repo_type = None, None
    else:
        raise ValueError(f"Unable to retrieve user and repo ID from the passed HF ID: {hf_id}")

    # Check if repo type is known (mapping "spaces" => "space" + empty value => `None`)
    if repo_type in REPO_TYPES_MAPPING:
        repo_type = REPO_TYPES_MAPPING[repo_type]
    if repo_type == "":
        repo_type = None
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Unknown `repo_type`: '{repo_type}' ('{input_hf_id}')")

    return repo_type, namespace, repo_id


class BlobLfsInfo(TypedDict, total=False):
    size: int
    sha256: str
    pointer_size: int


@dataclass
class CommitInfo:
    """Data structure containing information about a newly created commit.

    Returned by [`create_commit`].

    Args:
        commit_url (`str`):
            Url where to find the commit.

        commit_message (`str`):
            The summary (first line) of the commit that has been created.

        commit_description (`str`):
            Description of the commit that has been created. Can be empty.

        oid (`str`):
            Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.

        pr_url (`str`, *optional*):
            Url to the PR that has been created, if any. Populated when `create_pr=True`
            is passed.

        pr_revision (`str`, *optional*):
            Revision of the PR that has been created, if any. Populated when
            `create_pr=True` is passed. Example: `"refs/pr/1"`.

        pr_num (`int`, *optional*):
            Number of the PR discussion that has been created, if any. Populated when
            `create_pr=True` is passed. Can be passed as `discussion_num` in
            [`get_discussion_details`]. Example: `1`.
    """

    commit_url: str
    commit_message: str
    commit_description: str
    oid: str
    pr_url: Optional[str] = None

    # Computed from `pr_url` in `__post_init__`
    pr_revision: Optional[str] = field(init=False)
    pr_num: Optional[str] = field(init=False)

    def __post_init__(self):
        """Populate pr-related fields after initialization.

        See https://docs.python.org/3.10/library/dataclasses.html#post-init-processing.
        """
        if self.pr_url is not None:
            self.pr_revision = _parse_revision_from_pr_url(self.pr_url)
            self.pr_num = int(self.pr_revision.split("/")[-1])
        else:
            self.pr_revision = None
            self.pr_num = None


class RepoUrl(str):
    """Subclass of `str` describing a repo URL on the Hub.

    `RepoUrl` is returned by `HfApi.create_repo`. It inherits from `str` for backward
    compatibility. At initialization, the URL is parsed to populate properties:
    - endpoint (`str`)
    - namespace (`Optional[str]`)
    - repo_name (`str`)
    - repo_id (`str`)
    - repo_type (`Literal["model", "dataset", "space"]`)
    - url (`str`)

    Args:
        url (`Any`):
            String value of the repo url.
        endpoint (`str`, *optional*):
            Endpoint of the Hub. Defaults to <https://huggingface.co>.

    Example:
    ```py
    >>> RepoUrl('https://huggingface.co/gpt2')
    RepoUrl('https://huggingface.co/gpt2', endpoint='https://huggingface.co', repo_type='model', repo_id='gpt2')

    >>> RepoUrl('https://hub-ci.huggingface.co/datasets/dummy_user/dummy_dataset', endpoint='https://hub-ci.huggingface.co')
    RepoUrl('https://hub-ci.huggingface.co/datasets/dummy_user/dummy_dataset', endpoint='https://hub-ci.huggingface.co', repo_type='dataset', repo_id='dummy_user/dummy_dataset')

    >>> RepoUrl('hf://datasets/my-user/my-dataset')
    RepoUrl('hf://datasets/my-user/my-dataset', endpoint='https://huggingface.co', repo_type='dataset', repo_id='user/dataset')

    >>> HfApi.create_repo("dummy_model")
    RepoUrl('https://huggingface.co/Wauplin/dummy_model', endpoint='https://huggingface.co', repo_type='model', repo_id='Wauplin/dummy_model')
    ```

    Raises:
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If URL cannot be parsed.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `repo_type` is unknown.
    """

    def __new__(cls, url: Any, endpoint: Optional[str] = None):
        return super(RepoUrl, cls).__new__(cls, url)

    def __init__(self, url: Any, endpoint: Optional[str] = None) -> None:
        super().__init__()
        # Parse URL
        self.endpoint = endpoint or ENDPOINT
        repo_type, namespace, repo_name = repo_type_and_id_from_hf_id(self, hub_url=self.endpoint)

        # Populate fields
        self.namespace = namespace
        self.repo_name = repo_name
        self.repo_id = repo_name if namespace is None else f"{namespace}/{repo_name}"
        self.repo_type = repo_type or REPO_TYPE_MODEL
        self.url = str(self)  # just in case it's needed

    def __repr__(self) -> str:
        return f"RepoUrl('{self}', endpoint='{self.endpoint}', repo_type='{self.repo_type}', repo_id='{self.repo_id}')"


class RepoFile(ReprMixin):
    """
    Data structure that represents a public file inside a repo, accessible from huggingface.co

    Args:
        rfilename (str):
            file name, relative to the repo root. This is the only attribute that's guaranteed to be here, but under
            certain conditions there can certain other stuff.
        size (`int`, *optional*):
            The file's size, in bytes. This attribute is present when `files_metadata` argument of [`repo_info`] is set
            to `True`. It's `None` otherwise.
        blob_id (`str`, *optional*):
            The file's git OID. This attribute is present when `files_metadata` argument of [`repo_info`] is set to
            `True`. It's `None` otherwise.
        lfs (`BlobLfsInfo`, *optional*):
            The file's LFS metadata. This attribute is present when`files_metadata` argument of [`repo_info`] is set to
            `True` and the file is stored with Git LFS. It's `None` otherwise.
    """

    def __init__(
        self,
        rfilename: str,
        size: Optional[int] = None,
        blobId: Optional[str] = None,
        lfs: Optional[BlobLfsInfo] = None,
        **kwargs,
    ):
        self.rfilename = rfilename  # filename relative to the repo root

        # Optional file metadata
        self.size = size
        self.blob_id = blobId
        self.lfs = lfs

        # Hack to ensure backward compatibility with future versions of the API.
        # See discussion in https://github.com/huggingface/huggingface_hub/pull/951#discussion_r926460408
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelInfo(ReprMixin):
    """
    Info about a model accessible from huggingface.co

    Attributes:
        modelId (`str`, *optional*):
            ID of model repository.
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        tags (`List[str]`, *optional*):
            List of tags.
        pipeline_tag (`str`, *optional*):
            Pipeline tag to identify the correct widget.
        siblings (`List[RepoFile]`, *optional*):
            list of ([`huggingface_hub.hf_api.RepoFile`]) objects that constitute the model.
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        config (`Dict`, *optional*):
            Model configuration information
        securityStatus (`Dict`, *optional*):
            Security status of the model.
            Example: `{"containsInfected": False}`
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        modelId: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pipeline_tag: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        config: Optional[Dict] = None,
        securityStatus: Optional[Dict] = None,
        **kwargs,
    ):
        self.modelId = modelId
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = [RepoFile(**x) for x in siblings] if siblings is not None else []
        self.private = private
        self.author = author
        self.config = config
        self.securityStatus = securityStatus
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        r = f"Model Name: {self.modelId}, Tags: {self.tags}"
        if self.pipeline_tag:
            r += f", Task: {self.pipeline_tag}"
        return r


class DatasetInfo(ReprMixin):
    """
    Info about a dataset accessible from huggingface.co

    Attributes:
        id (`str`, *optional*):
            ID of dataset repository.
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        tags (`List[str]`, *optional*):
            List of tags.
        siblings (`List[RepoFile]`, *optional*):
            list of [`huggingface_hub.hf_api.RepoFile`] objects that constitute the dataset.
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        description (`str`, *optional*):
            Description of the dataset
        citation (`str`, *optional*):
            Dataset citation
        cardData (`Dict`, *optional*):
            Metadata of the model card as a dictionary.
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        description: Optional[str] = None,
        citation: Optional[str] = None,
        cardData: Optional[dict] = None,
        **kwargs,
    ):
        self.id = id
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.private = private
        self.author = author
        self.description = description
        self.citation = citation
        self.cardData = cardData
        self.siblings = [RepoFile(**x) for x in siblings] if siblings is not None else []
        # Legacy stuff, "key" is always returned with an empty string
        # because of old versions of the datasets lib that need this field
        kwargs.pop("key", None)
        # Store all the other fields returned by the API
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        r = f"Dataset Name: {self.id}, Tags: {self.tags}"
        return r


class SpaceInfo(ReprMixin):
    """
    Info about a Space accessible from huggingface.co

    This is a "dataclass" like container that just sets on itself any attribute
    passed by the server.

    Attributes:
        id (`str`, *optional*):
            id of space
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        siblings (`List[RepoFile]`, *optional*):
            list of [`huggingface_hub.hf_api.RepoFIle`] objects that constitute the Space
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        **kwargs,
    ):
        self.id = id
        self.sha = sha
        self.lastModified = lastModified
        self.siblings = [RepoFile(**x) for x in siblings] if siblings is not None else []
        self.private = private
        self.author = author
        for k, v in kwargs.items():
            setattr(self, k, v)


class MetricInfo(ReprMixin):
    """
    Info about a public metric accessible from huggingface.co
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,  # id of metric
        description: Optional[str] = None,
        citation: Optional[str] = None,
        **kwargs,
    ):
        self.id = id
        self.description = description
        self.citation = citation
        # Legacy stuff, "key" is always returned with an empty string
        # because of old versions of the datasets lib that need this field
        kwargs.pop("key", None)
        # Store all the other fields returned by the API
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        r = f"Metric Name: {self.id}"
        return r


class ModelSearchArguments(AttributeDictionary):
    """
    A nested namespace object holding all possible values for properties of
    models currently hosted in the Hub with tab-completion. If a value starts
    with a number, it will only exist in the dictionary

    Example:

    ```python
    >>> args = ModelSearchArguments()

    >>> args.author.huggingface
    'huggingface'

    >>> args.language.en
    'en'
    ```

    <Tip warning={true}>

    `ModelSearchArguments` is a legacy class meant for exploratory purposes only. Its
    initialization requires listing all models on the Hub which makes it increasingly
    slower as the number of repos on the Hub increases.

    </Tip>
    """

    def __init__(self, api: Optional["HfApi"] = None):
        self._api = api if api is not None else HfApi()
        tags = self._api.get_model_tags()
        super().__init__(tags)
        self._process_models()

    def _process_models(self):
        def clean(s: str) -> str:
            return s.replace(" ", "").replace("-", "_").replace(".", "_")

        models = self._api.list_models()
        author_dict, model_name_dict = AttributeDictionary(), AttributeDictionary()
        for model in models:
            if "/" in model.modelId:
                author, name = model.modelId.split("/")
                author_dict[author] = clean(author)
            else:
                name = model.modelId
            model_name_dict[name] = clean(name)
        self["model_name"] = model_name_dict
        self["author"] = author_dict


class DatasetSearchArguments(AttributeDictionary):
    """
    A nested namespace object holding all possible values for properties of
    datasets currently hosted in the Hub with tab-completion. If a value starts
    with a number, it will only exist in the dictionary

    Example:

    ```python
    >>> args = DatasetSearchArguments()

    >>> args.author.huggingface
    'huggingface'

    >>> args.language.en
    'language:en'
    ```

    <Tip warning={true}>

    `DatasetSearchArguments` is a legacy class meant for exploratory purposes only. Its
    initialization requires listing all datasets on the Hub which makes it increasingly
    slower as the number of repos on the Hub increases.

    </Tip>
    """

    def __init__(self, api: Optional["HfApi"] = None):
        self._api = api if api is not None else HfApi()
        tags = self._api.get_dataset_tags()
        super().__init__(tags)
        self._process_models()

    def _process_models(self):
        def clean(s: str):
            return s.replace(" ", "").replace("-", "_").replace(".", "_")

        datasets = self._api.list_datasets()
        author_dict, dataset_name_dict = AttributeDictionary(), AttributeDictionary()
        for dataset in datasets:
            if "/" in dataset.id:
                author, name = dataset.id.split("/")
                author_dict[author] = clean(author)
            else:
                name = dataset.id
            dataset_name_dict[name] = clean(name)
        self["dataset_name"] = dataset_name_dict
        self["author"] = author_dict


@dataclass
class GitRefInfo:
    """
    Contains information about a git reference for a repo on the Hub.

    Args:
        name (`str`):
            Name of the reference (e.g. tag name or branch name).
        ref (`str`):
            Full git ref on the Hub (e.g. `"refs/heads/main"` or `"refs/tags/v1.0"`).
        target_commit (`str`):
            OID of the target commit for the ref (e.g. `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`)
    """

    name: str
    ref: str
    target_commit: str

    def __init__(self, data: Dict) -> None:
        self.name = data["name"]
        self.ref = data["ref"]
        self.target_commit = data["targetCommit"]


@dataclass
class GitRefs:
    """
    Contains information about all git references for a repo on the Hub.

    Object is returned by [`list_repo_refs`].

    Args:
        branches (`List[GitRefInfo]`):
            A list of [`GitRefInfo`] containing information about branches on the repo.
        converts (`List[GitRefInfo]`):
            A list of [`GitRefInfo`] containing information about "convert" refs on the repo.
            Converts are refs used (internally) to push preprocessed data in Dataset repos.
        tags (`List[GitRefInfo]`):
            A list of [`GitRefInfo`] containing information about tags on the repo.
    """

    branches: List[GitRefInfo]
    converts: List[GitRefInfo]
    tags: List[GitRefInfo]


@dataclass
class GitCommitInfo:
    """
    Contains information about a git commit for a repo on the Hub. Check out [`list_repo_commits`] for more details.

    Args:
        commit_id (`str`):
            OID of the commit (e.g. `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`)
        authors (`List[str]`):
            List of authors of the commit.
        created_at (`datetime`):
            Datetime when the commit was created.
        title (`str`):
            Title of the commit. This is a free-text value entered by the authors.
        message (`str`):
            Description of the commit. This is a free-text value entered by the authors.
        formatted_title (`str`):
            Title of the commit formatted as HTML. Only returned if `formatted=True` is set.
        formatted_message (`str`):
            Description of the commit formatted as HTML. Only returned if `formatted=True` is set.
    """

    commit_id: str

    authors: List[str]
    created_at: datetime
    title: str
    message: str

    formatted_title: Optional[str]
    formatted_message: Optional[str]

    def __init__(self, data: Dict) -> None:
        self.commit_id = data["id"]
        self.authors = [author["user"] for author in data["authors"]]
        self.created_at = parse_datetime(data["date"])
        self.title = data["title"]
        self.message = data["message"]

        self.formatted_title = data.get("formatted", {}).get("title")
        self.formatted_message = data.get("formatted", {}).get("message")


@dataclass
class UserLikes:
    """
    Contains information about a user likes on the Hub.

    Args:
        user (`str`):
            Name of the user for which we fetched the likes.
        total (`int`):
            Total number of likes.
        datasets (`List[str]`):
            List of datasets liked by the user (as repo_ids).
        models (`List[str]`):
            List of models liked by the user (as repo_ids).
        spaces (`List[str]`):
            List of spaces liked by the user (as repo_ids).
    """

    # Metadata
    user: str
    total: int

    # User likes
    datasets: List[str]
    models: List[str]
    spaces: List[str]


def future_compatible(fn: CallableT) -> CallableT:
    """Wrap a method of `HfApi` to handle `run_as_future=True`.

    A method flagged as "future_compatible" will be called in a thread if `run_as_future=True` and return a
    `concurrent.futures.Future` instance. Otherwise, it will be called normally and return the result.
    """
    sig = inspect.signature(fn)
    args_params = list(sig.parameters)[1:]  # remove "self" from list

    @wraps(fn)
    def _inner(self, *args, **kwargs):
        # Get `run_as_future` value if provided (default to False)
        if "run_as_future" in kwargs:
            run_as_future = kwargs["run_as_future"]
            kwargs["run_as_future"] = False  # avoid recursion error
        else:
            run_as_future = False
            for param, value in zip(args_params, args):
                if param == "run_as_future":
                    run_as_future = value
                    break

        # Call the function in a thread if `run_as_future=True`
        if run_as_future:
            return self.run_as_future(fn, self, *args, **kwargs)

        # Otherwise, call the function normally
        return fn(self, *args, **kwargs)

    _inner.is_future_compatible = True  # type: ignore
    return _inner  # type: ignore


class HfApi:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
    ) -> None:
        """Create a HF client to interact with the Hub via HTTP.

        The client is initialized with some high-level settings used in all requests
        made to the Hub (HF endpoint, authentication, user agents...). Using the `HfApi`
        client is preferred but not mandatory as all of its public methods are exposed
        directly at the root of `huggingface_hub`.

        Args:
            endpoint (`str`, *optional*):
                Hugging Face Hub base url. Will default to https://huggingface.co/. To
                be set if you are using a private hub. Otherwise, one can set the
                `HF_ENDPOINT` environment variable.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if
                not provided.
            library_name (`str`, *optional*):
                The name of the library that is making the HTTP request. Will be added to
                the user-agent header. Example: `"transformers"`.
            library_version (`str`, *optional*):
                The version of the library that is making the HTTP request. Will be added
                to the user-agent header. Example: `"4.24.0"`.
            user_agent (`str`, `dict`, *optional*):
                The user agent info in the form of a dictionary or a single string. It will
                be completed with information about the installed packages.
        """
        self.endpoint = endpoint if endpoint is not None else ENDPOINT
        self.token = token
        self.library_name = library_name
        self.library_version = library_version
        self.user_agent = user_agent
        self._thread_pool: Optional[ThreadPoolExecutor] = None

    def run_as_future(self, fn: Callable[..., R], *args, **kwargs) -> Future[R]:
        """
        Run a method in the background and return a Future instance.

        The main goal is to run methods without blocking the main thread (e.g. to push data during a training).
        Background jobs are queued to preserve order but are not ran in parallel. If you need to speed-up your scripts
        by parallelizing lots of call to the API, you must setup and use your own [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor).

        Note: Most-used methods like [`upload_file`], [`upload_folder`] and [`create_commit`] have a `run_as_future: bool`
        argument to directly call them in the background. This is equivalent to calling `api.run_as_future(...)` on them
        but less verbose.

        Args:
            fn (`Callable`):
                The method to run in the background.
            *args, **kwargs:
                Arguments with which the method will be called.

        Return:
            `Future`: a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) instance to
            get the result of the task.

        Example:
            ```py
            >>> from huggingface_hub import HfApi
            >>> api = HfApi()
            >>> future = api.run_as_future(api.whoami) # instant
            >>> future.done()
            False
            >>> future.result() # wait until complete and return result
            (...)
            >>> future.done()
            True
            ```
        """
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._thread_pool
        return self._thread_pool.submit(fn, *args, **kwargs)

    @validate_hf_hub_args
    def whoami(self, token: Optional[str] = None) -> Dict:
        """
        Call HF API to know "whoami".

        Args:
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if
                not provided.
        """
        r = get_session().get(
            f"{self.endpoint}/api/whoami-v2",
            headers=self._build_hf_headers(
                # If `token` is provided and not `None`, it will be used by default.
                # Otherwise, the token must be retrieved from cache or env variable.
                token=(token or self.token or True),
            ),
        )
        try:
            hf_raise_for_status(r)
        except HTTPError as e:
            raise HTTPError(
                "Invalid user token. If you didn't pass a user token, make sure you "
                "are properly logged in by executing `huggingface-cli login`, and "
                "if you did pass a user token, double-check it's correct."
            ) from e
        return r.json()

    def get_token_permission(self, token: Optional[str] = None) -> Literal["read", "write", None]:
        """
        Check if a given `token` is valid and return its permissions.

        For more details about tokens, please refer to https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens.

        Args:
            token (`str`, *optional*):
                The token to check for validity. Defaults to the one saved locally.

        Returns:
            `Literal["read", "write", None]`: Permission granted by the token ("read" or "write"). Returns `None` if no
            token passed or token is invalid.
        """
        try:
            return self.whoami(token=token)["auth"]["accessToken"]["role"]
        except (LocalTokenNotFoundError, HTTPError):
            return None

    def get_model_tags(self) -> ModelTags:
        """
        List all valid model tags as a nested namespace object
        """
        path = f"{self.endpoint}/api/models-tags-by-type"
        r = get_session().get(path)
        hf_raise_for_status(r)
        d = r.json()
        return ModelTags(d)

    def get_dataset_tags(self) -> DatasetTags:
        """
        List all valid dataset tags as a nested namespace object.
        """
        path = f"{self.endpoint}/api/datasets-tags-by-type"
        r = get_session().get(path)
        hf_raise_for_status(r)
        d = r.json()
        return DatasetTags(d)

    @validate_hf_hub_args
    def list_models(
        self,
        *,
        filter: Union[ModelFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        emissions_thresholds: Optional[Tuple[float, float]] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        full: Optional[bool] = None,
        cardData: bool = False,
        fetch_config: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> Iterable[ModelInfo]:
        """
        List models hosted on the Huggingface Hub, given some filters.

        Args:
            filter ([`ModelFilter`] or `str` or `Iterable`, *optional*):
                A string or [`ModelFilter`] which can be used to identify models
                on the Hub.
            author (`str`, *optional*):
                A string which identify the author (user or organization) of the
                returned models
            search (`str`, *optional*):
                A string that will be contained in the returned model ids.
            emissions_thresholds (`Tuple`, *optional*):
                A tuple of two ints or floats representing a minimum and maximum
                carbon footprint to filter the resulting models with in grams.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting models. Possible values
                are the properties of the [`huggingface_hub.hf_api.ModelInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of models fetched. Leaving this option
                to `None` fetches all models.
            full (`bool`, *optional*):
                Whether to fetch all model data, including the `lastModified`,
                the `sha`, the files and the `tags`. This is set to `True` by
                default when using a filter.
            cardData (`bool`, *optional*):
                Whether to grab the metadata for the model as well. Can contain
                useful information such as carbon emissions, metrics, and
                datasets trained on.
            fetch_config (`bool`, *optional*):
                Whether to fetch the model configs as well. This is not included
                in `full` due to its size.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `Iterable[ModelInfo]`: an iterable of [`huggingface_hub.hf_api.ModelInfo`] objects.

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models
        >>> api.list_models()

        >>> # Get all valid search arguments
        >>> args = ModelSearchArguments()

        >>> # List only the text classification models
        >>> api.list_models(filter="text-classification")
        >>> # Using the `ModelFilter`
        >>> filt = ModelFilter(task="text-classification")
        >>> # With `ModelSearchArguments`
        >>> filt = ModelFilter(task=args.pipeline_tags.TextClassification)
        >>> api.list_models(filter=filt)

        >>> # Using `ModelFilter` and `ModelSearchArguments` to find text classification in both PyTorch and TensorFlow
        >>> filt = ModelFilter(
        ...     task=args.pipeline_tags.TextClassification,
        ...     library=[args.library.PyTorch, args.library.TensorFlow],
        ... )
        >>> api.list_models(filter=filt)

        >>> # List only models from the AllenNLP library
        >>> api.list_models(filter="allennlp")
        >>> # Using `ModelFilter` and `ModelSearchArguments`
        >>> filt = ModelFilter(library=args.library.allennlp)
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models with "bert" in their name
        >>> api.list_models(search="bert")

        >>> # List all models with "bert" in their name made by google
        >>> api.list_models(search="bert", author="google")
        ```
        """
        if emissions_thresholds is not None and cardData is None:
            raise ValueError("`emissions_thresholds` were passed without setting `cardData=True`.")

        path = f"{self.endpoint}/api/models"
        headers = self._build_hf_headers(token=token)
        params = {}
        if filter is not None:
            if isinstance(filter, ModelFilter):
                params = self._unpack_model_filter(filter)
            else:
                params.update({"filter": filter})
            params.update({"full": True})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full is not None:
            if full:
                params.update({"full": True})
            elif "full" in params:
                del params["full"]
        if fetch_config:
            params.update({"config": True})
        if cardData:
            params.update({"cardData": True})

        # `items` is a generator
        items = paginate(path, params=params, headers=headers)
        if limit is not None:
            items = islice(items, limit)  # Do not iterate over all pages
        if emissions_thresholds is not None:
            items = _filter_emissions(items, *emissions_thresholds)
        for item in items:
            yield ModelInfo(**item)

    def _unpack_model_filter(self, model_filter: ModelFilter):
        """
        Unpacks a [`ModelFilter`] into something readable for `list_models`
        """
        model_str = ""
        tags = []

        # Handling author
        if model_filter.author is not None:
            model_str = f"{model_filter.author}/"

        # Handling model_name
        if model_filter.model_name is not None:
            model_str += model_filter.model_name

        filter_list: List[str] = []

        # Handling tasks
        if model_filter.task is not None:
            filter_list.extend([model_filter.task] if isinstance(model_filter.task, str) else model_filter.task)

        # Handling dataset
        if model_filter.trained_dataset is not None:
            if not isinstance(model_filter.trained_dataset, (list, tuple)):
                model_filter.trained_dataset = [model_filter.trained_dataset]
            for dataset in model_filter.trained_dataset:
                if "dataset:" not in dataset:
                    dataset = f"dataset:{dataset}"
                filter_list.append(dataset)

        # Handling library
        if model_filter.library:
            filter_list.extend(
                [model_filter.library] if isinstance(model_filter.library, str) else model_filter.library
            )

        # Handling tags
        if model_filter.tags:
            tags.extend([model_filter.tags] if isinstance(model_filter.tags, str) else model_filter.tags)

        query_dict: Dict[str, Any] = {}
        if model_str is not None:
            query_dict["search"] = model_str
        if len(tags) > 0:
            query_dict["tags"] = tags
        if isinstance(model_filter.language, list):
            filter_list.extend(model_filter.language)
        elif isinstance(model_filter.language, str):
            filter_list.append(model_filter.language)
        query_dict["filter"] = tuple(filter_list)
        return query_dict

    @validate_hf_hub_args
    def list_datasets(
        self,
        *,
        filter: Union[DatasetFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        full: Optional[bool] = None,
        token: Optional[str] = None,
    ) -> Iterable[DatasetInfo]:
        """
        List datasets hosted on the Huggingface Hub, given some filters.

        Args:
            filter ([`DatasetFilter`] or `str` or `Iterable`, *optional*):
                A string or [`DatasetFilter`] which can be used to identify
                datasets on the hub.
            author (`str`, *optional*):
                A string which identify the author of the returned datasets.
            search (`str`, *optional*):
                A string that will be contained in the returned datasets.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting datasets. Possible
                values are the properties of the [`huggingface_hub.hf_api.DatasetInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of datasets fetched. Leaving this option
                to `None` fetches all datasets.
            full (`bool`, *optional*):
                Whether to fetch all dataset data, including the `lastModified`
                and the `cardData`. Can contain useful information such as the
                PapersWithCode ID.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `Iterable[DatasetInfo]`: an iterable of [`huggingface_hub.hf_api.DatasetInfo`] objects.

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all datasets
        >>> api.list_datasets()

        >>> # Get all valid search arguments
        >>> args = DatasetSearchArguments()

        >>> # List only the text classification datasets
        >>> api.list_datasets(filter="task_categories:text-classification")
        >>> # Using the `DatasetFilter`
        >>> filt = DatasetFilter(task_categories="text-classification")
        >>> # With `DatasetSearchArguments`
        >>> filt = DatasetFilter(task=args.task_categories.text_classification)
        >>> api.list_models(filter=filt)

        >>> # List only the datasets in russian for language modeling
        >>> api.list_datasets(
        ...     filter=("language:ru", "task_ids:language-modeling")
        ... )
        >>> # Using the `DatasetFilter`
        >>> filt = DatasetFilter(language="ru", task_ids="language-modeling")
        >>> # With `DatasetSearchArguments`
        >>> filt = DatasetFilter(
        ...     language=args.language.ru,
        ...     task_ids=args.task_ids.language_modeling,
        ... )
        >>> api.list_datasets(filter=filt)
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all datasets with "text" in their name
        >>> api.list_datasets(search="text")

        >>> # List all datasets with "text" in their name made by google
        >>> api.list_datasets(search="text", author="google")
        ```
        """
        path = f"{self.endpoint}/api/datasets"
        headers = self._build_hf_headers(token=token)
        params = {}
        if filter is not None:
            if isinstance(filter, DatasetFilter):
                params = self._unpack_dataset_filter(filter)
            else:
                params.update({"filter": filter})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full:
            params.update({"full": True})

        items = paginate(path, params=params, headers=headers)
        if limit is not None:
            items = islice(items, limit)  # Do not iterate over all pages
        for item in items:
            yield DatasetInfo(**item)

    def _unpack_dataset_filter(self, dataset_filter: DatasetFilter):
        """
        Unpacks a [`DatasetFilter`] into something readable for `list_datasets`
        """
        dataset_str = ""

        # Handling author
        if dataset_filter.author is not None:
            dataset_str = f"{dataset_filter.author}/"

        # Handling dataset_name
        if dataset_filter.dataset_name is not None:
            dataset_str += dataset_filter.dataset_name

        filter_list = []
        data_attributes = [
            "benchmark",
            "language_creators",
            "language",
            "multilinguality",
            "size_categories",
            "task_categories",
            "task_ids",
        ]

        for attr in data_attributes:
            curr_attr = getattr(dataset_filter, attr)
            if curr_attr is not None:
                if not isinstance(curr_attr, (list, tuple)):
                    curr_attr = [curr_attr]
                for data in curr_attr:
                    if f"{attr}:" not in data:
                        data = f"{attr}:{data}"
                    filter_list.append(data)

        query_dict: Dict[str, Any] = {}
        if dataset_str is not None:
            query_dict["search"] = dataset_str
        query_dict["filter"] = tuple(filter_list)
        return query_dict

    def list_metrics(self) -> List[MetricInfo]:
        """
        Get the public list of all the metrics on huggingface.co

        Returns:
            `List[MetricInfo]`: a list of [`MetricInfo`] objects which.
        """
        path = f"{self.endpoint}/api/metrics"
        r = get_session().get(path)
        hf_raise_for_status(r)
        d = r.json()
        return [MetricInfo(**x) for x in d]

    @validate_hf_hub_args
    def list_spaces(
        self,
        *,
        filter: Union[str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        datasets: Union[str, Iterable[str], None] = None,
        models: Union[str, Iterable[str], None] = None,
        linked: bool = False,
        full: Optional[bool] = None,
        token: Optional[str] = None,
    ) -> Iterable[SpaceInfo]:
        """
        List spaces hosted on the Huggingface Hub, given some filters.

        Args:
            filter (`str` or `Iterable`, *optional*):
                A string tag or list of tags that can be used to identify Spaces on the Hub.
            author (`str`, *optional*):
                A string which identify the author of the returned Spaces.
            search (`str`, *optional*):
                A string that will be contained in the returned Spaces.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting Spaces. Possible
                values are the properties of the [`huggingface_hub.hf_api.SpaceInfo`]` class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of Spaces fetched. Leaving this option
                to `None` fetches all Spaces.
            datasets (`str` or `Iterable`, *optional*):
                Whether to return Spaces that make use of a dataset.
                The name of a specific dataset can be passed as a string.
            models (`str` or `Iterable`, *optional*):
                Whether to return Spaces that make use of a model.
                The name of a specific model can be passed as a string.
            linked (`bool`, *optional*):
                Whether to return Spaces that make use of either a model or a dataset.
            full (`bool`, *optional*):
                Whether to fetch all Spaces data, including the `lastModified`
                and the `cardData`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `Iterable[SpaceInfo]`: an iterable of [`huggingface_hub.hf_api.SpaceInfo`] objects.
        """
        path = f"{self.endpoint}/api/spaces"
        headers = self._build_hf_headers(token=token)
        params: Dict[str, Any] = {}
        if filter is not None:
            params.update({"filter": filter})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full:
            params.update({"full": True})
        if linked:
            params.update({"linked": True})
        if datasets is not None:
            params.update({"datasets": datasets})
        if models is not None:
            params.update({"models": models})

        items = paginate(path, params=params, headers=headers)
        if limit is not None:
            items = islice(items, limit)  # Do not iterate over all pages
        for item in items:
            yield SpaceInfo(**item)

    @validate_hf_hub_args
    def like(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Like a given repo on the Hub (e.g. set as favorite).

        See also [`unlike`] and [`list_liked_repos`].

        Args:
            repo_id (`str`):
                The repository to like. Example: `"user/my-cool-model"`.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if liking a dataset or space, `None` or
                `"model"` if liking a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.

        Example:
        ```python
        >>> from huggingface_hub import like, list_liked_repos, unlike
        >>> like("gpt2")
        >>> "gpt2" in list_liked_repos().models
        True
        >>> unlike("gpt2")
        >>> "gpt2" in list_liked_repos().models
        False
        ```
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        response = get_session().post(
            url=f"{self.endpoint}/api/{repo_type}s/{repo_id}/like",
            headers=self._build_hf_headers(token=token),
        )
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def unlike(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Unlike a given repo on the Hub (e.g. remove from favorite list).

        See also [`like`] and [`list_liked_repos`].

        Args:
            repo_id (`str`):
                The repository to unlike. Example: `"user/my-cool-model"`.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if unliking a dataset or space, `None` or
                `"model"` if unliking a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.

        Example:
        ```python
        >>> from huggingface_hub import like, list_liked_repos, unlike
        >>> like("gpt2")
        >>> "gpt2" in list_liked_repos().models
        True
        >>> unlike("gpt2")
        >>> "gpt2" in list_liked_repos().models
        False
        ```
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        response = get_session().delete(
            url=f"{self.endpoint}/api/{repo_type}s/{repo_id}/like", headers=self._build_hf_headers(token=token)
        )
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def list_liked_repos(
        self,
        user: Optional[str] = None,
        *,
        token: Optional[str] = None,
    ) -> UserLikes:
        """
        List all public repos liked by a user on huggingface.co.

        This list is public so token is optional. If `user` is not passed, it defaults to
        the logged in user.

        See also [`like`] and [`unlike`].

        Args:
            user (`str`, *optional*):
                Name of the user for which you want to fetch the likes.
            token (`str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                Used only if `user` is not passed to implicitly determine the current
                user name.

        Returns:
            [`UserLikes`]: object containing the user name and 3 lists of repo ids (1 for
            models, 1 for datasets and 1 for Spaces).

        Raises:
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If `user` is not passed and no token found (either from argument or from machine).

        Example:
        ```python
        >>> from huggingface_hub import list_liked_repos

        >>> likes = list_liked_repos("julien-c")

        >>> likes.user
        "julien-c"

        >>> likes.models
        ["osanseviero/streamlit_1.15", "Xhaheen/ChatGPT_HF", ...]
        ```
        """
        # User is either provided explicitly or retrieved from current token.
        if user is None:
            me = self.whoami(token=token)
            if me["type"] == "user":
                user = me["name"]
            else:
                raise ValueError(
                    "Cannot list liked repos. You must provide a 'user' as input or be logged in as a user."
                )

        path = f"{self.endpoint}/api/users/{user}/likes"
        headers = self._build_hf_headers(token=token)

        likes = list(paginate(path, params={}, headers=headers))
        # Looping over a list of items similar to:
        #   {
        #       'createdAt': '2021-09-09T21:53:27.000Z',
        #       'repo': {
        #           'name': 'PaddlePaddle/PaddleOCR',
        #           'type': 'space'
        #        }
        #   }
        # Let's loop 3 times over the received list. Less efficient but more straightforward to read.
        return UserLikes(
            user=user,
            total=len(likes),
            models=[like["repo"]["name"] for like in likes if like["repo"]["type"] == "model"],
            datasets=[like["repo"]["name"] for like in likes if like["repo"]["type"] == "dataset"],
            spaces=[like["repo"]["name"] for like in likes if like["repo"]["type"] == "space"],
        )

    @validate_hf_hub_args
    def model_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        securityStatus: Optional[bool] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> ModelInfo:
        """
        Get info on one specific model on huggingface.co

        Model can be private if you pass an acceptable token or are logged in.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the model repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            securityStatus (`bool`, *optional*):
                Whether to retrieve the security status from the model
                repository as well.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`huggingface_hub.hf_api.ModelInfo`]: The model repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        path = (
            f"{self.endpoint}/api/models/{repo_id}"
            if revision is None
            else (f"{self.endpoint}/api/models/{repo_id}/revision/{quote(revision, safe='')}")
        )
        params = {}
        if securityStatus:
            params["securityStatus"] = True
        if files_metadata:
            params["blobs"] = True
        r = get_session().get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return ModelInfo(**d)

    @validate_hf_hub_args
    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> DatasetInfo:
        """
        Get info on one specific dataset on huggingface.co.

        Dataset can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the dataset repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`hf_api.DatasetInfo`]: The dataset repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        path = (
            f"{self.endpoint}/api/datasets/{repo_id}"
            if revision is None
            else (f"{self.endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}")
        )
        params = {}
        if files_metadata:
            params["blobs"] = True

        r = get_session().get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return DatasetInfo(**d)

    @validate_hf_hub_args
    def space_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> SpaceInfo:
        """
        Get info on one specific Space on huggingface.co.

        Space can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the space repository from which to get the
                information.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            [`~hf_api.SpaceInfo`]: The space repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = self._build_hf_headers(token=token)
        path = (
            f"{self.endpoint}/api/spaces/{repo_id}"
            if revision is None
            else (f"{self.endpoint}/api/spaces/{repo_id}/revision/{quote(revision, safe='')}")
        )
        params = {}
        if files_metadata:
            params["blobs"] = True

        r = get_session().get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return SpaceInfo(**d)

    @validate_hf_hub_args
    def repo_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        token: Optional[Union[bool, str]] = None,
    ) -> Union[ModelInfo, DatasetInfo, SpaceInfo]:
        """
        Get the info object for a given repo of a given type.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the repository from which to get the
                information.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space,
                `None` or `"model"` if getting repository info from a model. Default is `None`.
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `Union[SpaceInfo, DatasetInfo, ModelInfo]`: The repository information, as a
            [`huggingface_hub.hf_api.DatasetInfo`], [`huggingface_hub.hf_api.ModelInfo`]
            or [`huggingface_hub.hf_api.SpaceInfo`] object.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        if repo_type is None or repo_type == "model":
            method = self.model_info
        elif repo_type == "dataset":
            method = self.dataset_info  # type: ignore
        elif repo_type == "space":
            method = self.space_info  # type: ignore
        else:
            raise ValueError("Unsupported repo type.")
        return method(
            repo_id,
            revision=revision,
            token=token,
            timeout=timeout,
            files_metadata=files_metadata,
        )

    @validate_hf_hub_args
    def list_files_info(
        self,
        repo_id: str,
        paths: Union[List[str], str, None] = None,
        *,
        expand: bool = False,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> Iterable[RepoFile]:
        """
        List files on a repo and get information about them.

        Takes as input a list of paths. Those paths can be either files or folders. Two server endpoints are called:
        1. POST "/paths-info" to get information about the provided paths. Called once.
        2. GET  "/tree?recursive=True" to paginate over the input folders. Called only if a folder path is provided as
           input. Will be called multiple times to follow pagination.
        If no path is provided as input, step 1. is ignored and all files from the repo are listed.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            paths (`Union[List[str], str, None]`, *optional*):
                The paths to get information about. Paths to files are directly resolved. Paths to folders are resolved
                recursively which means that information is returned about all files in the folder and its subfolders.
                If `None`, all files are returned (the default). If a path do not exist, it is ignored without raising
                an exception.
            expand (`bool`, *optional*, defaults to `False`):
                Whether to fetch more information about the files (e.g. last commit and security scan results). This
                operation is more expensive for the server so only 50 results are returned per page (instead of 1000).
                As pagination is implemented in `huggingface_hub`, this is transparent for you except for the time it
                takes to get the results.
            revision (`str`, *optional*):
                The revision of the repository from which to get the information. Defaults to `"main"` branch.
            repo_type (`str`, *optional*):
                The type of the repository from which to get the information (`"model"`, `"dataset"` or `"space"`.
                Defaults to `"model"`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token). If `None` or `True` and
                machine is logged in (through `huggingface-cli login` or [`~huggingface_hub.login`]), token will be
                retrieved from the cache. If `False`, token is not sent in the request header.

        Returns:
            `Iterable[RepoFile]`:
                The information about the files, as an iterable of [`RepoFile`] objects. The order of the files is
                not guaranteed.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo
                does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.

        Examples:

            Get information about files on a repo.
            ```py
            >>> from huggingface_hub import list_files_info
            >>> files_info = list_files_info("lysandre/arxiv-nlp", ["README.md", "config.json"])
            >>> files_info
            <generator object HfApi.list_files_info at 0x7f93b848e730>
            >>> list(files_info)
            [
                RepoFile: {"blob_id": "43bd404b159de6fba7c2f4d3264347668d43af25", "lfs": None, "rfilename": "README.md", "size": 391},
                RepoFile: {"blob_id": "2f9618c3a19b9a61add74f70bfb121335aeef666", "lfs": None, "rfilename": "config.json", "size": 554},
            ]
            ```

            Get even more information about files on a repo (last commit and security scan results)
            ```py
            >>> from huggingface_hub import list_files_info
            >>> files_info = list_files_info("prompthero/openjourney-v4", expand=True)
            >>> list(files_info)
            [
                RepoFile: {
                {'blob_id': '815004af1a321eaed1d93f850b2e94b0c0678e42',
                'lastCommit': {'date': '2023-03-21T09:05:27.000Z',
                                'id': '47b62b20b20e06b9de610e840282b7e6c3d51190',
                                'title': 'Upload diffusers weights (#48)'},
                'lfs': None,
                'rfilename': 'model_index.json',
                'security': {'avScan': {'virusFound': False, 'virusNames': None},
                                'blobId': '815004af1a321eaed1d93f850b2e94b0c0678e42',
                                'name': 'model_index.json',
                                'pickleImportScan': None,
                                'repositoryId': 'models/prompthero/openjourney-v4',
                                'safe': True},
                'size': 584}
                },
                RepoFile: {
                {'blob_id': 'd2343d78b33ac03dade1d525538b02b130d0a3a0',
                'lastCommit': {'date': '2023-03-21T09:05:27.000Z',
                                'id': '47b62b20b20e06b9de610e840282b7e6c3d51190',
                                'title': 'Upload diffusers weights (#48)'},
                'lfs': {'pointer_size': 134,
                        'sha256': 'dcf4507d99b88db73f3916e2a20169fe74ada6b5582e9af56cfa80f5f3141765',
                        'size': 334711857},
                'rfilename': 'vae/diffusion_pytorch_model.bin',
                'security': {'avScan': {'virusFound': False, 'virusNames': None},
                                'blobId': 'd2343d78b33ac03dade1d525538b02b130d0a3a0',
                                'name': 'vae/diffusion_pytorch_model.bin',
                                'pickleImportScan': {'highestSafetyLevel': 'innocuous',
                                                    'imports': [{'module': 'torch._utils',
                                                                'name': '_rebuild_tensor_v2',
                                                                'safety': 'innocuous'},
                                                                {'module': 'collections', 'name': 'OrderedDict', 'safety': 'innocuous'},
                                                                {'module': 'torch', 'name': 'FloatStorage', 'safety': 'innocuous'}]},
                                'repositoryId': 'models/prompthero/openjourney-v4',
                                'safe': True},
                'size': 334711857}
                },
                (...)
            ]
            ```

            List LFS files from the "vae/" folder in "stabilityai/stable-diffusion-2" repository.

            ```py
            >>> from huggingface_hub import list_files_info
            >>> [info.rfilename for info in list_files_info("stabilityai/stable-diffusion-2", "vae") if info.lfs is not None]
            ['vae/diffusion_pytorch_model.bin', 'vae/diffusion_pytorch_model.safetensors']
            ```

            List all files on a repo.
            ```py
            >>> from huggingface_hub import list_files_info
            >>> [info.rfilename for info in list_files_info("glue", repo_type="dataset")]
            ['.gitattributes', 'README.md', 'dataset_infos.json', 'glue.py']
            ```
        """
        repo_type = repo_type or REPO_TYPE_MODEL
        revision = quote(revision, safe="") if revision is not None else DEFAULT_REVISION
        headers = self._build_hf_headers(token=token)

        def _format_as_repo_file(info: Dict) -> RepoFile:
            # Quick alias very specific to the server return type of /paths-info and /tree endpoints. Let's keep this
            # logic here.
            rfilename = info.pop("path")
            size = info.pop("size")
            blobId = info.pop("oid")
            lfs = info.pop("lfs", None)
            info.pop("type", None)  # "file" or "folder" -> not needed in practice since we know it's a file
            if lfs is not None:
                lfs = BlobLfsInfo(size=lfs["size"], sha256=lfs["oid"], pointer_size=lfs["pointerSize"])
            return RepoFile(rfilename=rfilename, size=size, blobId=blobId, lfs=lfs, **info)

        folder_paths = []
        if paths is None:
            # `paths` is not provided => list all files from the repo
            folder_paths.append("")
        elif paths == []:
            # corner case: server would return a 400 error if `paths` is an empty list. Let's return early.
            return
        else:
            # `paths` is provided => get info about those
            response = get_session().post(
                f"{self.endpoint}/api/{repo_type}s/{repo_id}/paths-info/{revision}",
                data={
                    "paths": paths if isinstance(paths, list) else [paths],
                    "expand": True,
                },
                headers=headers,
            )
            hf_raise_for_status(response)
            paths_info = response.json()

            # List top-level files first
            for path_info in paths_info:
                if path_info["type"] == "file":
                    yield _format_as_repo_file(path_info)
                else:
                    folder_paths.append(path_info["path"])

        # List files in subdirectories
        for path in folder_paths:
            encoded_path = "/" + quote(path, safe="") if path else ""
            tree_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}{encoded_path}"
            for subpath_info in paginate(path=tree_url, headers=headers, params={"recursive": True, "expand": expand}):
                if subpath_info["type"] == "file":
                    yield _format_as_repo_file(subpath_info)

    @_deprecate_arguments(version="0.17", deprecated_args=["timeout"], custom_message="timeout is not used anymore.")
    @validate_hf_hub_args
    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        timeout: Optional[float] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> List[str]:
        """
        Get the list of files in a given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            revision (`str`, *optional*):
                The revision of the model repository from which to get the information.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to
                a model. Default is `None`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token). If `None` or `True` and
                machine is logged in (through `huggingface-cli login` or [`~huggingface_hub.login`]), token will be
                retrieved from the cache. If `False`, token is not sent in the request header.

        Returns:
            `List[str]`: the list of files in a given repository.
        """
        return [
            f.rfilename
            for f in self.list_files_info(
                repo_id=repo_id, paths=None, revision=revision, repo_type=repo_type, token=token
            )
        ]

    @validate_hf_hub_args
    def list_repo_refs(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> GitRefs:
        """
        Get the list of refs of a given repo (both tags and branches).

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if listing refs from a dataset or a Space,
                `None` or `"model"` if listing from a model. Default is `None`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Example:
        ```py
        >>> from huggingface_hub import HfApi
        >>> api = HfApi()
        >>> api.list_repo_refs("gpt2")
        GitRefs(branches=[GitRefInfo(name='main', ref='refs/heads/main', target_commit='e7da7f221d5bf496a48136c0cd264e630fe9fcc8')], converts=[], tags=[])

        >>> api.list_repo_refs("bigcode/the-stack", repo_type='dataset')
        GitRefs(
            branches=[
                GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
                GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
            ],
            converts=[],
            tags=[
                GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
            ]
        )
        ```

        Returns:
            [`GitRefs`]: object containing all information about branches and tags for a
            repo on the Hub.
        """
        repo_type = repo_type or REPO_TYPE_MODEL
        response = get_session().get(
            f"{self.endpoint}/api/{repo_type}s/{repo_id}/refs", headers=self._build_hf_headers(token=token)
        )
        hf_raise_for_status(response)
        data = response.json()
        return GitRefs(
            branches=[GitRefInfo(item) for item in data["branches"]],
            converts=[GitRefInfo(item) for item in data["converts"]],
            tags=[GitRefInfo(item) for item in data["tags"]],
        )

    @validate_hf_hub_args
    def list_repo_commits(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        formatted: bool = False,
    ) -> List[GitCommitInfo]:
        """
        Get the list of commits of a given revision for a repo on the Hub.

        Commits are sorted by date (last commit first).

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if listing commits from a dataset or a Space, `None` or `"model"` if
                listing from a model. Default is `None`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            formatted (`bool`):
                Whether to return the HTML-formatted title and description of the commits. Defaults to False.

        Example:
        ```py
        >>> from huggingface_hub import HfApi
        >>> api = HfApi()

        # Commits are sorted by date (last commit first)
        >>> initial_commit = api.list_repo_commits("gpt2")[-1]

        # Initial commit is always a system commit containing the `.gitattributes` file.
        >>> initial_commit
        GitCommitInfo(
            commit_id='9b865efde13a30c13e0a33e536cf3e4a5a9d71d8',
            authors=['system'],
            created_at=datetime.datetime(2019, 2, 18, 10, 36, 15, tzinfo=datetime.timezone.utc),
            title='initial commit',
            message='',
            formatted_title=None,
            formatted_message=None
        )

        # Create an empty branch by deriving from initial commit
        >>> api.create_branch("gpt2", "new_empty_branch", revision=initial_commit.commit_id)
        ```

        Returns:
            List[[`GitCommitInfo`]]: list of objects containing information about the commits for a repo on the Hub.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo
                does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.
        """
        repo_type = repo_type or REPO_TYPE_MODEL
        revision = quote(revision, safe="") if revision is not None else DEFAULT_REVISION

        # Paginate over results and return the list of commits.
        return [
            GitCommitInfo(item)
            for item in paginate(
                f"{self.endpoint}/api/{repo_type}s/{repo_id}/commits/{revision}",
                headers=self._build_hf_headers(token=token),
                params={"expand[]": "formatted"} if formatted else {},
            )
        ]

    @validate_hf_hub_args
    def create_repo(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        private: bool = False,
        repo_type: Optional[str] = None,
        exist_ok: bool = False,
        space_sdk: Optional[str] = None,
        space_hardware: Optional[str] = None,
    ) -> RepoUrl:
        """Create an empty repo on the HuggingFace Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if repo already exists.
            space_sdk (`str`, *optional*):
                Choice of SDK to use if repo_type is "space". Can be "streamlit", "gradio", "docker", or "static".
            space_hardware (`SpaceHardware` or `str`, *optional*):
                Choice of Hardware if repo_type is "space". See [`SpaceHardware`] for a complete list.

        Returns:
            [`RepoUrl`]: URL to the newly created repo. Value is a subclass of `str` containing
            attributes like `endpoint`, `repo_type` and `repo_id`.
        """
        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        path = f"{self.endpoint}/api/repos/create"

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization, "private": private}
        if repo_type is not None:
            json["type"] = repo_type
        if repo_type == "space":
            if space_sdk is None:
                raise ValueError(
                    "No space_sdk provided. `create_repo` expects space_sdk to be one"
                    f" of {SPACES_SDK_TYPES} when repo_type is 'space'`"
                )
            if space_sdk not in SPACES_SDK_TYPES:
                raise ValueError(f"Invalid space_sdk. Please choose one of {SPACES_SDK_TYPES}.")
            json["sdk"] = space_sdk

        if space_sdk is not None and repo_type != "space":
            warnings.warn("Ignoring provided space_sdk because repo_type is not 'space'.")

        if space_hardware is not None:
            if repo_type == "space":
                json["hardware"] = space_hardware
            else:
                warnings.warn("Ignoring provided space_hardware because repo_type is not 'space'.")

        if getattr(self, "_lfsmultipartthresh", None):
            # Testing purposes only.
            # See https://github.com/huggingface/huggingface_hub/pull/733/files#r820604472
            json["lfsmultipartthresh"] = self._lfsmultipartthresh  # type: ignore
        headers = self._build_hf_headers(token=token, is_write_action=True)
        r = get_session().post(path, headers=headers, json=json)

        try:
            hf_raise_for_status(r)
        except HTTPError as err:
            if exist_ok and err.response.status_code == 409:
                # Repo already exists and `exist_ok=True`
                pass
            elif exist_ok and err.response.status_code == 403:
                # No write permission on the namespace but repo might already exist
                try:
                    self.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
                    if repo_type is None or repo_type == REPO_TYPE_MODEL:
                        return RepoUrl(f"{self.endpoint}/{repo_id}")
                    return RepoUrl(f"{self.endpoint}/{repo_type}/{repo_id}")
                except HfHubHTTPError:
                    raise
            else:
                raise

        d = r.json()
        return RepoUrl(d["url"], endpoint=self.endpoint)

    @validate_hf_hub_args
    def delete_repo(
        self,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """
        Delete a repo from the HuggingFace Hub. CAUTION: this is irreversible.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        path = f"{self.endpoint}/api/repos/delete"

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization}
        if repo_type is not None:
            json["type"] = repo_type

        headers = self._build_hf_headers(token=token, is_write_action=True)
        r = get_session().delete(path, headers=headers, json=json)
        hf_raise_for_status(r)

    @validate_hf_hub_args
    def update_repo_visibility(
        self,
        repo_id: str,
        private: bool = False,
        *,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Update the visibility setting of a repository.

        Args:
            repo_id (`str`, *optional*):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns:
            The HTTP response in json.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        if organization is None:
            namespace = self.whoami(token)["name"]
        else:
            namespace = organization

        if repo_type is None:
            repo_type = REPO_TYPE_MODEL  # default repo type

        r = get_session().put(
            url=f"{self.endpoint}/api/{repo_type}s/{namespace}/{name}/settings",
            headers=self._build_hf_headers(token=token, is_write_action=True),
            json={"private": private},
        )
        hf_raise_for_status(r)
        return r.json()

    def move_repo(
        self,
        from_id: str,
        to_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Moving a repository from namespace1/repo_name1 to namespace2/repo_name2

        Note there are certain limitations. For more information about moving
        repositories, please see
        https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo.

        Args:
            from_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Original repository identifier.
            to_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Final repository identifier.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if len(from_id.split("/")) != 2:
            raise ValueError(f"Invalid repo_id: {from_id}. It should have a namespace (:namespace:/:repo_name:)")

        if len(to_id.split("/")) != 2:
            raise ValueError(f"Invalid repo_id: {to_id}. It should have a namespace (:namespace:/:repo_name:)")

        if repo_type is None:
            repo_type = REPO_TYPE_MODEL  # Hub won't accept `None`.

        json = {"fromRepo": from_id, "toRepo": to_id, "type": repo_type}

        path = f"{self.endpoint}/api/repos/move"
        headers = self._build_hf_headers(token=token, is_write_action=True)
        r = get_session().post(path, headers=headers, json=json)
        try:
            hf_raise_for_status(r)
        except HfHubHTTPError as e:
            e.append_to_message(
                "\nFor additional documentation please see"
                " https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo."
            )
            raise

    @overload
    def create_commit(  # type: ignore
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        num_threads: int = 5,
        parent_commit: Optional[str] = None,
        run_as_future: Literal[False] = ...,
    ) -> CommitInfo:
        ...

    @overload
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        num_threads: int = 5,
        parent_commit: Optional[str] = None,
        run_as_future: Literal[True] = ...,
    ) -> Future[CommitInfo]:
        ...

    @validate_hf_hub_args
    @future_compatible
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        num_threads: int = 5,
        parent_commit: Optional[str] = None,
        run_as_future: bool = False,
    ) -> Union[CommitInfo, Future[CommitInfo]]:
        """
        Creates a commit in the given repo, deleting & uploading files as needed.

        Args:
            repo_id (`str`):
                The repository in which the commit will be created, for example:
                `"username/custom_transformers"`

            operations (`Iterable` of [`~hf_api.CommitOperation`]):
                An iterable of operations to include in the commit, either:

                    - [`~hf_api.CommitOperationAdd`] to upload a file
                    - [`~hf_api.CommitOperationDelete`] to delete a file
                    - [`~hf_api.CommitOperationCopy`] to copy a file

            commit_message (`str`):
                The summary (first line) of the commit that will be created.

            commit_description (`str`, *optional*):
                The description of the commit that will be created

            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.

            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.

            num_threads (`int`, *optional*):
                Number of concurrent threads for uploading files. Defaults to 5.
                Setting it to 2 means at most 2 files will be uploaded concurrently.

            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string.
                Shorthands (7 first characters) are also supported. If specified and `create_pr` is `False`,
                the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr`
                is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit`
                ensures the repo has not changed before committing the changes, and can be especially useful
                if the repo is updated / committed to concurrently.
            run_as_future (`bool`, *optional*):
                Whether or not to run this method in the background. Background jobs are run sequentially without
                blocking the main thread. Passing `run_as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
                object. Defaults to `False`.

        Returns:
            [`CommitInfo`] or `Future`:
                Instance of [`CommitInfo`] containing information about the newly created commit (commit hash, commit
                url, pr url, commit message,...). If `run_as_future=True` is passed, returns a Future object which will
                contain the result when executed.

        Raises:
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If commit message is empty.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If parent commit is not a valid commit OID.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If the Hub API returns an HTTP 400 error (bad request)
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If `create_pr` is `True` and revision is neither `None` nor `"main"`.
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.

        <Tip warning={true}>

        `create_commit` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        <Tip warning={true}>

        `create_commit` is limited to 25k LFS files and a 1GB payload for regular files.

        </Tip>
        """
        _CREATE_COMMIT_NO_REPO_ERROR_MESSAGE = (
            "\nNote: Creating a commit assumes that the repo already exists on the"
            " Huggingface Hub. Please use `create_repo` if it's not the case."
        )

        if parent_commit is not None and not REGEX_COMMIT_OID.fullmatch(parent_commit):
            raise ValueError(
                f"`parent_commit` is not a valid commit OID. It must match the following regex: {REGEX_COMMIT_OID}"
            )

        if commit_message is None or len(commit_message) == 0:
            raise ValueError("`commit_message` can't be empty, please pass a value.")

        commit_description = commit_description if commit_description is not None else ""
        repo_type = repo_type if repo_type is not None else REPO_TYPE_MODEL
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        revision = quote(revision, safe="") if revision is not None else DEFAULT_REVISION
        create_pr = create_pr if create_pr is not None else False

        operations = list(operations)
        additions = [op for op in operations if isinstance(op, CommitOperationAdd)]
        copies = [op for op in operations if isinstance(op, CommitOperationCopy)]
        nb_additions = len(additions)
        nb_copies = len(copies)
        nb_deletions = len(operations) - nb_additions - nb_copies

        logger.debug(
            f"About to commit to the hub: {len(additions)} addition(s), {len(copies)} copie(s) and"
            f" {nb_deletions} deletion(s)."
        )

        # If updating twice the same file or update then delete a file in a single commit
        warn_on_overwriting_operations(operations)

        try:
            upload_modes = fetch_upload_modes(
                additions=additions,
                repo_type=repo_type,
                repo_id=repo_id,
                token=token or self.token,
                revision=revision,
                endpoint=self.endpoint,
                create_pr=create_pr,
            )
        except RepositoryNotFoundError as e:
            e.append_to_message(_CREATE_COMMIT_NO_REPO_ERROR_MESSAGE)
            raise
        files_to_copy = fetch_lfs_files_to_copy(
            copies=copies,
            repo_type=repo_type,
            repo_id=repo_id,
            token=token or self.token,
            revision=revision,
            endpoint=self.endpoint,
        )
        upload_lfs_files(
            additions=[addition for addition in additions if upload_modes[addition.path_in_repo] == "lfs"],
            repo_type=repo_type,
            repo_id=repo_id,
            token=token or self.token,
            endpoint=self.endpoint,
            num_threads=num_threads,
        )
        commit_payload = prepare_commit_payload(
            operations=operations,
            upload_modes=upload_modes,
            files_to_copy=files_to_copy,
            commit_message=commit_message,
            commit_description=commit_description,
            parent_commit=parent_commit,
        )
        commit_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"

        def _payload_as_ndjson() -> Iterable[bytes]:
            for item in commit_payload:
                yield json.dumps(item).encode()
                yield b"\n"

        headers = {
            # See https://github.com/huggingface/huggingface_hub/issues/1085#issuecomment-1265208073
            "Content-Type": "application/x-ndjson",
            **self._build_hf_headers(token=token, is_write_action=True),
        }
        data = b"".join(_payload_as_ndjson())
        params = {"create_pr": "1"} if create_pr else None

        try:
            commit_resp = get_session().post(url=commit_url, headers=headers, data=data, params=params)
            hf_raise_for_status(commit_resp, endpoint_name="commit")
        except RepositoryNotFoundError as e:
            e.append_to_message(_CREATE_COMMIT_NO_REPO_ERROR_MESSAGE)
            raise
        except EntryNotFoundError as e:
            if nb_deletions > 0 and "A file with this name doesn't exist" in str(e):
                e.append_to_message(
                    "\nMake sure to differentiate file and folder paths in delete"
                    " operations with a trailing '/' or using `is_folder=True/False`."
                )
            raise

        commit_data = commit_resp.json()
        return CommitInfo(
            commit_url=commit_data["commitUrl"],
            commit_message=commit_message,
            commit_description=commit_description,
            oid=commit_data["commitOid"],
            pr_url=commit_data["pullRequestUrl"] if create_pr else None,
        )

    @experimental
    @validate_hf_hub_args
    def create_commits_on_pr(
        self,
        *,
        repo_id: str,
        addition_commits: List[List[CommitOperationAdd]],
        deletion_commits: List[List[CommitOperationDelete]],
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        merge_pr: bool = True,
        num_threads: int = 5,  # TODO: use to multithread uploads
        verbose: bool = False,
    ) -> str:
        """Push changes to the Hub in multiple commits.

        Commits are pushed to a draft PR branch. If the upload fails or gets interrupted, it can be resumed. Progress
        is tracked in the PR description. At the end of the process, the PR is set as open and the title is updated to
        match the initial commit message. If `merge_pr=True` is passed, the PR is merged automatically.

        All deletion commits are pushed first, followed by the addition commits. The order of the commits is not
        guaranteed as we might implement parallel commits in the future. Be sure that your are not updating several
        times the same file.

        <Tip warning={true}>

        `create_commits_on_pr` is experimental.  Its API and behavior is subject to change in the future without prior notice.

        </Tip>

        Args:
            repo_id (`str`):
                The repository in which the commits will be pushed. Example: `"username/my-cool-model"`.

            addition_commits (`List` of `List` of [`~hf_api.CommitOperationAdd`]):
                A list containing lists of [`~hf_api.CommitOperationAdd`]. Each sublist will result in a commit on the
                PR.

            deletion_commits
                A list containing lists of [`~hf_api.CommitOperationDelete`]. Each sublist will result in a commit on
                the PR. Deletion commits are pushed before addition commits.

            commit_message (`str`):
                The summary (first line) of the commit that will be created. Will also be the title of the PR.

            commit_description (`str`, *optional*):
                The description of the commit that will be created. The description will be added to the PR.

            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"` if uploading to
                a model. Default is `None`.

            merge_pr (`bool`):
                If set to `True`, the Pull Request is merged at the end of the process. Defaults to `True`.

            num_threads (`int`, *optional*):
                Number of concurrent threads for uploading files. Defaults to 5.

            verbose (`bool`):
                If set to `True`, process will run on verbose mode i.e. print information about the ongoing tasks.
                Defaults to `False`.

        Returns:
            `str`: URL to the created PR.

        Example:
        ```python
        >>> from huggingface_hub import HfApi, plan_multi_commits
        >>> addition_commits, deletion_commits = plan_multi_commits(
        ...     operations=[
        ...          CommitOperationAdd(...),
        ...          CommitOperationAdd(...),
        ...          CommitOperationDelete(...),
        ...          CommitOperationDelete(...),
        ...          CommitOperationAdd(...),
        ...     ],
        ... )
        >>> HfApi().create_commits_on_pr(
        ...     repo_id="my-cool-model",
        ...     addition_commits=addition_commits,
        ...     deletion_commits=deletion_commits,
        ...     (...)
        ...     verbose=True,
        ... )
        ```

        Raises:
            [`MultiCommitException`]:
                If an unexpected issue occur in the process: empty commits, unexpected commits in a PR, unexpected PR
                description, etc.

        <Tip warning={true}>

        `create_commits_on_pr` assumes that the repo already exists on the Hub. If you get a Client error 404, please
        make sure you are authenticated and that `repo_id` and `repo_type` are set correctly. If repo does not exist,
        create it first using [`~hf_api.create_repo`].

        </Tip>
        """
        logger = logging.get_logger(__name__ + ".create_commits_on_pr")
        if verbose:
            logger.setLevel("INFO")

        # 1. Get strategy ID
        logger.info(
            f"Will create {len(deletion_commits)} deletion commit(s) and {len(addition_commits)} addition commit(s),"
            f" totalling {sum(len(ops) for ops in addition_commits+deletion_commits)} atomic operations."
        )
        strategy = MultiCommitStrategy(
            addition_commits=[MultiCommitStep(operations=operations) for operations in addition_commits],  # type: ignore
            deletion_commits=[MultiCommitStep(operations=operations) for operations in deletion_commits],  # type: ignore
        )
        logger.info(f"Multi-commits strategy with ID {strategy.id}.")

        # 2. Get or create a PR with this strategy ID
        for discussion in self.get_repo_discussions(repo_id=repo_id, repo_type=repo_type, token=token):
            # search for a draft PR with strategy ID
            if discussion.is_pull_request and discussion.status == "draft" and strategy.id in discussion.title:
                pr = self.get_discussion_details(
                    repo_id=repo_id, discussion_num=discussion.num, repo_type=repo_type, token=token
                )
                logger.info(f"PR already exists: {pr.url}. Will resume process where it stopped.")
                break
        else:
            # did not find a PR matching the strategy ID
            pr = multi_commit_create_pull_request(
                self,
                repo_id=repo_id,
                commit_message=commit_message,
                commit_description=commit_description,
                strategy=strategy,
                token=token,
                repo_type=repo_type,
            )
            logger.info(f"New PR created: {pr.url}")

        # 3. Parse PR description to check consistency with strategy (e.g. same commits are scheduled)
        for event in pr.events:
            if isinstance(event, DiscussionComment):
                pr_comment = event
                break
        else:
            raise MultiCommitException(f"PR #{pr.num} must have at least 1 comment")

        description_commits = multi_commit_parse_pr_description(pr_comment.content)
        if len(description_commits) != len(strategy.all_steps):
            raise MultiCommitException(
                f"Corrupted multi-commit PR #{pr.num}: got {len(description_commits)} steps in"
                f" description but {len(strategy.all_steps)} in strategy."
            )
        for step_id in strategy.all_steps:
            if step_id not in description_commits:
                raise MultiCommitException(
                    f"Corrupted multi-commit PR #{pr.num}: expected step {step_id} but didn't find"
                    f" it (have {', '.join(description_commits)})."
                )

        # 4. Retrieve commit history (and check consistency)
        commits_on_main_branch = {
            commit.commit_id
            for commit in self.list_repo_commits(
                repo_id=repo_id, repo_type=repo_type, token=token, revision=DEFAULT_REVISION
            )
        }
        pr_commits = [
            commit
            for commit in self.list_repo_commits(
                repo_id=repo_id, repo_type=repo_type, token=token, revision=pr.git_reference
            )
            if commit.commit_id not in commits_on_main_branch
        ]
        if len(pr_commits) > 0:
            logger.info(f"Found {len(pr_commits)} existing commits on the PR.")

        # At this point `pr_commits` is a list of commits pushed to the PR. We expect all of these commits (if any) to have
        # a step_id as title. We raise exception if an unexpected commit has been pushed.
        if len(pr_commits) > len(strategy.all_steps):
            raise MultiCommitException(
                f"Corrupted multi-commit PR #{pr.num}: scheduled {len(strategy.all_steps)} steps but"
                f" {len(pr_commits)} commits have already been pushed to the PR."
            )

        # Check which steps are already completed
        remaining_additions = {step.id: step for step in strategy.addition_commits}
        remaining_deletions = {step.id: step for step in strategy.deletion_commits}
        for commit in pr_commits:
            if commit.title in remaining_additions:
                step = remaining_additions.pop(commit.title)
                step.completed = True
            elif commit.title in remaining_deletions:
                step = remaining_deletions.pop(commit.title)
                step.completed = True

        if len(remaining_deletions) > 0 and len(remaining_additions) < len(strategy.addition_commits):
            raise MultiCommitException(
                f"Corrupted multi-commit PR #{pr.num}: some addition commits have already been pushed to the PR but"
                " deletion commits are not all completed yet."
            )
        nb_remaining = len(remaining_deletions) + len(remaining_additions)
        if len(pr_commits) > 0:
            logger.info(
                f"{nb_remaining} commits remaining ({len(remaining_deletions)} deletion commits and"
                f" {len(remaining_additions)} addition commits)"
            )

        # 5. Push remaining commits to the PR + update description
        # TODO: multi-thread this
        for step in list(remaining_deletions.values()) + list(remaining_additions.values()):
            # Push new commit
            self.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=step.id,
                revision=pr.git_reference,
                num_threads=num_threads,
                operations=step.operations,
                create_pr=False,
            )
            step.completed = True
            nb_remaining -= 1
            logger.info(f"  step {step.id} completed (still {nb_remaining} to go).")

            # Update PR description
            self.edit_discussion_comment(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                discussion_num=pr.num,
                comment_id=pr_comment.id,
                new_content=multi_commit_generate_comment(
                    commit_message=commit_message, commit_description=commit_description, strategy=strategy
                ),
            )
        logger.info("All commits have been pushed.")

        # 6. Update PR (and merge)
        self.rename_discussion(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            discussion_num=pr.num,
            new_title=commit_message,
        )
        self.change_discussion_status(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            discussion_num=pr.num,
            new_status="open",
            comment=MULTI_COMMIT_PR_COMPLETION_COMMENT_TEMPLATE,
        )
        logger.info("PR is now open for reviews.")

        if merge_pr:  # User don't want a PR => merge it
            try:
                self.merge_pull_request(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    discussion_num=pr.num,
                    comment=MULTI_COMMIT_PR_CLOSING_COMMENT_TEMPLATE,
                )
                logger.info("PR has been automatically merged (`merge_pr=True` was passed).")
            except BadRequestError as error:
                if error.server_message is not None and "no associated changes" in error.server_message:
                    # PR cannot be merged as no changes are associated. We close the PR without merging with a comment to
                    # explain.
                    self.change_discussion_status(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        discussion_num=pr.num,
                        comment=MULTI_COMMIT_PR_CLOSE_COMMENT_FAILURE_NO_CHANGES_TEMPLATE,
                        new_status="closed",
                    )
                    logger.warning("Couldn't merge the PR: no associated changes.")
                else:
                    # PR cannot be merged for another reason (conflicting files for example). We comment the PR to explain
                    # and re-raise the exception.
                    self.comment_discussion(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token,
                        discussion_num=pr.num,
                        comment=MULTI_COMMIT_PR_CLOSE_COMMENT_FAILURE_BAD_REQUEST_TEMPLATE.format(
                            error_message=error.server_message
                        ),
                    )
                    raise MultiCommitException(
                        f"Couldn't merge Pull Request in multi-commit: {error.server_message}"
                    ) from error

        return pr.url

    @overload
    def upload_file(  # type: ignore
        self,
        *,
        path_or_fileobj: Union[str, Path, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        run_as_future: Literal[False] = ...,
    ) -> str:
        ...

    @overload
    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, Path, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        run_as_future: Literal[True] = ...,
    ) -> Future[str]:
        ...

    @validate_hf_hub_args
    @future_compatible
    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, Path, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        run_as_future: bool = False,
    ) -> Union[str, Future[str]]:
        """
        Upload a local file (up to 50 GB) to the given repo. The upload is done
        through a HTTP post request, and doesn't require git or git-lfs to be
        installed.

        Args:
            path_or_fileobj (`str`, `Path`, `bytes`, or `IO`):
                Path to a file on the local machine or binary data stream /
                fileobj / buffer.
            path_in_repo (`str`):
                Relative filepath in the repo, for example:
                `"checkpoints/1fec34a/weights.bin"`
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit
            commit_description (`str` *optional*)
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
            run_as_future (`bool`, *optional*):
                Whether or not to run this method in the background. Background jobs are run sequentially without
                blocking the main thread. Passing `run_as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
                object. Defaults to `False`.


        Returns:
            `str` or `Future`: The URL to visualize the uploaded file on the hub. If `run_as_future=True` is passed,
            returns a Future object which will contain the result when executed.

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>

        <Tip warning={true}>

        `upload_file` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        Example:

        ```python
        >>> from huggingface_hub import upload_file

        >>> with open("./local/filepath", "rb") as fobj:
        ...     upload_file(
        ...         path_or_fileobj=fileobj,
        ...         path_in_repo="remote/file/path.h5",
        ...         repo_id="username/my-dataset",
        ...         repo_type="dataset",
        ...         token="my_token",
        ...     )
        "https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

        >>> upload_file(
        ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
        ...     path_in_repo="remote/file/path.h5",
        ...     repo_id="username/my-model",
        ...     token="my_token",
        ... )
        "https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"

        >>> upload_file(
        ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
        ...     path_in_repo="remote/file/path.h5",
        ...     repo_id="username/my-model",
        ...     token="my_token",
        ...     create_pr=True,
        ... )
        "https://huggingface.co/username/my-model/blob/refs%2Fpr%2F1/remote/file/path.h5"
        ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        commit_message = (
            commit_message if commit_message is not None else f"Upload {path_in_repo} with huggingface_hub"
        )
        operation = CommitOperationAdd(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
        )

        commit_info = self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

        if commit_info.pr_url is not None:
            revision = quote(_parse_revision_from_pr_url(commit_info.pr_url), safe="")
        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        revision = revision if revision is not None else DEFAULT_REVISION
        # Similar to `hf_hub_url` but it's "blob" instead of "resolve"
        return f"{self.endpoint}/{repo_id}/blob/{revision}/{path_in_repo}"

    @overload
    def upload_folder(  # type: ignore
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        multi_commits: bool = False,
        multi_commits_verbose: bool = False,
        run_as_future: Literal[False] = ...,
    ) -> str:
        ...

    @overload
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        multi_commits: bool = False,
        multi_commits_verbose: bool = False,
        run_as_future: Literal[True] = ...,
    ) -> Future[str]:
        ...

    @validate_hf_hub_args
    @future_compatible
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        multi_commits: bool = False,
        multi_commits_verbose: bool = False,
        run_as_future: bool = False,
    ) -> Union[str, Future[str]]:
        """
        Upload a local folder to the given repo. The upload is done through a HTTP requests, and doesn't require git or
        git-lfs to be installed.

        The structure of the folder will be preserved. Files with the same name already present in the repository will
        be overwritten. Others will be left untouched.

        Use the `allow_patterns` and `ignore_patterns` arguments to specify which files to upload. These parameters
        accept either a single pattern or a list of patterns. Patterns are Standard Wildcards (globbing patterns) as
        documented [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). If both `allow_patterns` and
        `ignore_patterns` are provided, both constraints apply. By default, all files from the folder are uploaded.

        Use the `delete_patterns` argument to specify remote files you want to delete. Input type is the same as for
        `allow_patterns` (see above). If `path_in_repo` is also provided, the patterns are matched against paths
        relative to this folder. For example, `upload_folder(..., path_in_repo="experiment", delete_patterns="logs/*")`
        will delete any remote file under `./experiment/logs/`. Note that the `.gitattributes` file will not be deleted
        even if it matches the patterns.

        Any `.git/` folder present in any subdirectory will be ignored. However, please be aware that the `.gitignore`
        file is not taken into account.

        Uses `HfApi.create_commit` under the hood.

        Args:
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            folder_path (`str` or `Path`):
                Path to the folder to upload on the local file system
            path_in_repo (`str`, *optional*):
                Relative path of the directory in the repo, for example:
                `"checkpoints/1fec34a/results"`. Will default to the root folder of the repository.
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to:
                `f"Upload {path_in_repo} with huggingface_hub"`
            commit_description (`str` *optional*):
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`. If `revision` is not
                set, PR is opened against the `"main"` branch. If `revision` is set and is a branch, PR is opened
                against this branch. If `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server. If both `multi_commits` and `create_pr` are True,
                the PR created in the multi-commit process is kept opened.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are uploaded.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not uploaded.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo while committing
                new files. This is useful if you don't know which files have already been uploaded.
                Note: to avoid discrepancies the `.gitattributes` file is not deleted even if it matches the pattern.
            multi_commits (`bool`):
                If True, changes are pushed to a PR using a multi-commit process. Defaults to `False`.
            multi_commits_verbose (`bool`):
                If True and `multi_commits` is used, more information will be displayed to the user.
            run_as_future (`bool`, *optional*):
                Whether or not to run this method in the background. Background jobs are run sequentially without
                blocking the main thread. Passing `run_as_future=True` will return a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
                object. Defaults to `False`.

        Returns:
            `str` or `Future[str]`: A URL to visualize the uploaded folder on the hub. If `run_as_future=True` is passed,
            returns a Future object which will contain the result when executed.

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
            if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            if some parameter value is invalid

        </Tip>

        <Tip warning={true}>

        `upload_folder` assumes that the repo already exists on the Hub. If you get a Client error 404, please make
        sure you are authenticated and that `repo_id` and `repo_type` are set correctly. If repo does not exist, create
        it first using [`~hf_api.create_repo`].

        </Tip>

        <Tip warning={true}>

        `multi_commits` is experimental. Its API and behavior is subject to change in the future without prior notice.

        </Tip>

        Example:

        ```python
        # Upload checkpoints folder except the log files
        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     ignore_patterns="**/logs/*.txt",
        ... )
        # "https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

        # Upload checkpoints folder including logs while deleting existing logs from the repo
        # Useful if you don't know exactly which log files have already being pushed
        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     delete_patterns="**/logs/*.txt",
        ... )
        "https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

        # Upload checkpoints folder while creating a PR
        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     create_pr=True,
        ... )
        "https://huggingface.co/datasets/username/my-dataset/tree/refs%2Fpr%2F1/remote/experiment/checkpoints"

        ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        if multi_commits:
            if revision is not None and revision != DEFAULT_REVISION:
                raise ValueError("Cannot use `multi_commit` to commit changes other than the main branch.")

        # By default, upload folder to the root directory in repo.
        if path_in_repo is None:
            path_in_repo = ""

        # Do not upload .git folder
        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        ignore_patterns += IGNORE_GIT_FOLDER_PATTERNS

        delete_operations = self._prepare_upload_folder_deletions(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=DEFAULT_REVISION if create_pr else revision,
            token=token,
            path_in_repo=path_in_repo,
            delete_patterns=delete_patterns,
        )
        add_operations = _prepare_upload_folder_additions(
            folder_path,
            path_in_repo,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        # Optimize operations: if some files will be overwritten, we don't need to delete them first
        if len(add_operations) > 0:
            added_paths = set(op.path_in_repo for op in add_operations)
            delete_operations = [
                delete_op for delete_op in delete_operations if delete_op.path_in_repo not in added_paths
            ]
        commit_operations = delete_operations + add_operations

        pr_url: Optional[str]
        commit_message = commit_message or "Upload folder using huggingface_hub"
        if multi_commits:
            addition_commits, deletion_commits = plan_multi_commits(operations=commit_operations)
            pr_url = self.create_commits_on_pr(
                repo_id=repo_id,
                repo_type=repo_type,
                addition_commits=addition_commits,
                deletion_commits=deletion_commits,
                commit_message=commit_message,
                commit_description=commit_description,
                token=token,
                merge_pr=not create_pr,
                verbose=multi_commits_verbose,
            )
        else:
            commit_info = self.create_commit(
                repo_type=repo_type,
                repo_id=repo_id,
                operations=commit_operations,
                commit_message=commit_message,
                commit_description=commit_description,
                token=token,
                revision=revision,
                create_pr=create_pr,
                parent_commit=parent_commit,
            )
            pr_url = commit_info.pr_url

        if create_pr and pr_url is not None:
            revision = quote(_parse_revision_from_pr_url(pr_url), safe="")
        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        revision = revision if revision is not None else DEFAULT_REVISION
        # Similar to `hf_hub_url` but it's "tree" instead of "resolve"
        return f"{self.endpoint}/{repo_id}/tree/{revision}/{path_in_repo}"

    @validate_hf_hub_args
    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ) -> CommitInfo:
        """
        Deletes a file in the given repo.

        Args:
            path_in_repo (`str`):
                Relative filepath in the repo, for example:
                `"checkpoints/1fec34a/weights.bin"`
            repo_id (`str`):
                The repository from which the file will be deleted, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if the file is in a dataset or
                space, `None` or `"model"` if in a model. Default is `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to
                `f"Delete {path_in_repo} with huggingface_hub"`.
            commit_description (`str` *optional*)
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.


        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.
            - [`~utils.EntryNotFoundError`]
              If the file to download cannot be found.

        </Tip>

        """
        commit_message = (
            commit_message if commit_message is not None else f"Delete {path_in_repo} with huggingface_hub"
        )

        operations = [CommitOperationDelete(path_in_repo=path_in_repo)]

        return self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            operations=operations,
            revision=revision,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    @validate_hf_hub_args
    def delete_folder(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ) -> CommitInfo:
        """
        Deletes a folder in the given repo.

        Simple wrapper around [`create_commit`] method.

        Args:
            path_in_repo (`str`):
                Relative folder path in the repo, for example: `"checkpoints/1fec34a"`.
            repo_id (`str`):
                The repository from which the folder will be deleted, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will default
                to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if the folder is in a dataset or
                space, `None` or `"model"` if in a model. Default is `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to
                `f"Delete folder {path_in_repo} with huggingface_hub"`.
            commit_description (`str` *optional*)
                The description of the generated commit.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request with that commit. Defaults to `False`.
                If `revision` is not set, PR is opened against the `"main"` branch. If
                `revision` is set and is a branch, PR is opened against this branch. If
                `revision` is set and is not a branch name (example: a commit oid), an
                `RevisionNotFoundError` is returned by the server.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
        """
        return self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            operations=[CommitOperationDelete(path_in_repo=path_in_repo, is_folder=True)],
            revision=revision,
            commit_message=(
                commit_message if commit_message is not None else f"Delete folder {path_in_repo} with huggingface_hub"
            ),
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    @validate_hf_hub_args
    def create_branch(
        self,
        repo_id: str,
        *,
        branch: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        exist_ok: bool = False,
    ) -> None:
        """
        Create a new branch for a repo on the Hub, starting from the specified revision (defaults to `main`).
        To find a revision suiting your needs, you can use [`list_repo_refs`] or [`list_repo_commits`].

        Args:
            repo_id (`str`):
                The repository in which the branch will be created.
                Example: `"user/my-cool-model"`.

            branch (`str`):
                The name of the branch to create.

            revision (`str`, *optional*):
                The git revision to create the branch from. It can be a branch name or
                the OID/SHA of a commit, as a hexadecimal string. Defaults to the head
                of the `"main"` branch.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if creating a branch on a dataset or
                space, `None` or `"model"` if tagging a model. Default is `None`.

            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if branch already exists.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.BadRequestError`]:
                If invalid reference for a branch. Ex: `refs/pr/5` or 'refs/foo/bar'.
            [`~utils.HfHubHTTPError`]:
                If the branch already exists on the repo (error 409) and `exist_ok` is
                set to `False`.
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        branch = quote(branch, safe="")

        # Prepare request
        branch_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/branch/{branch}"
        headers = self._build_hf_headers(token=token, is_write_action=True)
        payload = {}
        if revision is not None:
            payload["startingPoint"] = revision

        # Create branch
        response = get_session().post(url=branch_url, headers=headers, json=payload)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            if not (e.response.status_code == 409 and exist_ok):
                raise

    @validate_hf_hub_args
    def delete_branch(
        self,
        repo_id: str,
        *,
        branch: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Delete a branch from a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a branch will be deleted.
                Example: `"user/my-cool-model"`.

            branch (`str`):
                The name of the branch to delete.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if creating a branch on a dataset or
                space, `None` or `"model"` if tagging a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.HfHubHTTPError`]:
                If trying to delete a protected branch. Ex: `main` cannot be deleted.
            [`~utils.HfHubHTTPError`]:
                If trying to delete a branch that does not exist.

        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        branch = quote(branch, safe="")

        # Prepare request
        branch_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/branch/{branch}"
        headers = self._build_hf_headers(token=token, is_write_action=True)

        # Delete branch
        response = get_session().delete(url=branch_url, headers=headers)
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        tag_message: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        exist_ok: bool = False,
    ) -> None:
        """
        Tag a given commit of a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a commit will be tagged.
                Example: `"user/my-cool-model"`.

            tag (`str`):
                The name of the tag to create.

            tag_message (`str`, *optional*):
                The description of the tag to create.

            revision (`str`, *optional*):
                The git revision to tag. It can be a branch name or the OID/SHA of a
                commit, as a hexadecimal string. Shorthands (7 first characters) are
                also supported. Defaults to the head of the `"main"` branch.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if tagging a dataset or
                space, `None` or `"model"` if tagging a model. Default is
                `None`.

            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if tag already exists.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.
            [`~utils.HfHubHTTPError`]:
                If the branch already exists on the repo (error 409) and `exist_ok` is
                set to `False`.
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        revision = quote(revision, safe="") if revision is not None else DEFAULT_REVISION

        # Prepare request
        tag_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/tag/{revision}"
        headers = self._build_hf_headers(token=token, is_write_action=True)
        payload = {"tag": tag}
        if tag_message is not None:
            payload["message"] = tag_message

        # Tag
        response = get_session().post(url=tag_url, headers=headers, json=payload)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            if not (e.response.status_code == 409 and exist_ok):
                raise

    @validate_hf_hub_args
    def delete_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> None:
        """
        Delete a tag from a repo on the Hub.

        Args:
            repo_id (`str`):
                The repository in which a tag will be deleted.
                Example: `"user/my-cool-model"`.

            tag (`str`):
                The name of the tag to delete.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if tagging a dataset or space, `None` or
                `"model"` if tagging a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.
            [`~utils.RevisionNotFoundError`]:
                If tag is not found.
        """
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        tag = quote(tag, safe="")

        # Prepare request
        tag_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/tag/{tag}"
        headers = self._build_hf_headers(token=token, is_write_action=True)

        # Un-tag
        response = get_session().delete(url=tag_url, headers=headers)
        hf_raise_for_status(response)

    @validate_hf_hub_args
    def get_full_repo_name(
        self,
        model_id: str,
        *,
        organization: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
    ):
        """
        Returns the repository name for a given model ID and optional
        organization.

        Args:
            model_id (`str`):
                The name of the model.
            organization (`str`, *optional*):
                If passed, the repository name will be in the organization
                namespace instead of the user namespace.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `str`: The repository name in the user's namespace
            ({username}/{model_id}) if no organization is passed, and under the
            organization namespace ({organization}/{model_id}) otherwise.
        """
        if organization is None:
            if "/" in model_id:
                username = model_id.split("/")[0]
            else:
                username = self.whoami(token=token)["name"]  # type: ignore
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"

    @validate_hf_hub_args
    def get_repo_discussions(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Iterator[Discussion]:
        """
        Fetches Discussions and Pull Requests for the given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if fetching from a dataset or
                space, `None` or `"model"` if fetching from a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

        Returns:
            `Iterator[Discussion]`: An iterator of [`Discussion`] objects.

        Example:
            Collecting all discussions of a repo in a list:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
            ```

            Iterating over discussions of a repo:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> for discussion in get_repo_discussions(repo_id="bert-base-uncased"):
            ...     print(discussion.num, discussion.title)
            ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL

        headers = self._build_hf_headers(token=token)

        def _fetch_discussion_page(page_index: int):
            path = f"{self.endpoint}/api/{repo_type}s/{repo_id}/discussions?p={page_index}"
            resp = get_session().get(path, headers=headers)
            hf_raise_for_status(resp)
            paginated_discussions = resp.json()
            total = paginated_discussions["count"]
            start = paginated_discussions["start"]
            discussions = paginated_discussions["discussions"]
            has_next = (start + len(discussions)) < total
            return discussions, has_next

        has_next, page_index = True, 0

        while has_next:
            discussions, has_next = _fetch_discussion_page(page_index=page_index)
            for discussion in discussions:
                yield Discussion(
                    title=discussion["title"],
                    num=discussion["num"],
                    author=discussion.get("author", {}).get("name", "deleted"),
                    created_at=parse_datetime(discussion["createdAt"]),
                    status=discussion["status"],
                    repo_id=discussion["repo"]["name"],
                    repo_type=discussion["repo"]["type"],
                    is_pull_request=discussion["isPullRequest"],
                    endpoint=self.endpoint,
                )
            page_index = page_index + 1

    @validate_hf_hub_args
    def get_discussion_details(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> DiscussionWithDetails:
        """Fetches a Discussion's / Pull Request 's details from the Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if not isinstance(discussion_num, int) or discussion_num <= 0:
            raise ValueError("Invalid discussion_num, must be a positive integer")
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL

        path = f"{self.endpoint}/api/{repo_type}s/{repo_id}/discussions/{discussion_num}"
        headers = self._build_hf_headers(token=token)
        resp = get_session().get(path, params={"diff": "1"}, headers=headers)
        hf_raise_for_status(resp)

        discussion_details = resp.json()
        is_pull_request = discussion_details["isPullRequest"]

        target_branch = discussion_details["changes"]["base"] if is_pull_request else None
        conflicting_files = discussion_details["filesWithConflicts"] if is_pull_request else None
        merge_commit_oid = discussion_details["changes"].get("mergeCommitId", None) if is_pull_request else None

        return DiscussionWithDetails(
            title=discussion_details["title"],
            num=discussion_details["num"],
            author=discussion_details.get("author", {}).get("name", "deleted"),
            created_at=parse_datetime(discussion_details["createdAt"]),
            status=discussion_details["status"],
            repo_id=discussion_details["repo"]["name"],
            repo_type=discussion_details["repo"]["type"],
            is_pull_request=discussion_details["isPullRequest"],
            events=[deserialize_event(evt) for evt in discussion_details["events"]],
            conflicting_files=conflicting_files,
            target_branch=target_branch,
            merge_commit_oid=merge_commit_oid,
            diff=discussion_details.get("diff"),
            endpoint=self.endpoint,
        )

    @validate_hf_hub_args
    def create_discussion(
        self,
        repo_id: str,
        title: str,
        *,
        token: Optional[str] = None,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
        pull_request: bool = False,
    ) -> DiscussionWithDetails:
        """Creates a Discussion or Pull Request.

        Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            pull_request (`bool`, *optional*):
                Whether to create a Pull Request or discussion. If `True`, creates a Pull Request.
                If `False`, creates a discussion. Defaults to `False`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL

        if description is not None:
            description = description.strip()
        description = (
            description
            if description
            else (
                f"{'Pull Request' if pull_request else 'Discussion'} opened with the"
                " [huggingface_hub Python"
                " library](https://huggingface.co/docs/huggingface_hub)"
            )
        )

        headers = self._build_hf_headers(token=token, is_write_action=True)
        resp = get_session().post(
            f"{self.endpoint}/api/{repo_type}s/{repo_id}/discussions",
            json={
                "title": title.strip(),
                "description": description,
                "pullRequest": pull_request,
            },
            headers=headers,
        )
        hf_raise_for_status(resp)
        num = resp.json()["num"]
        return self.get_discussion_details(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=num,
            token=token,
        )

    @validate_hf_hub_args
    def create_pull_request(
        self,
        repo_id: str,
        title: str,
        *,
        token: Optional[str] = None,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionWithDetails:
        """Creates a Pull Request . Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`];

        This is a wrapper around [`HfApi.create_discussion`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
        return self.create_discussion(
            repo_id=repo_id,
            title=title,
            token=token,
            description=description,
            repo_type=repo_type,
            pull_request=True,
        )

    def _post_discussion_changes(
        self,
        *,
        repo_id: str,
        discussion_num: int,
        resource: str,
        body: Optional[dict] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> requests.Response:
        """Internal utility to POST changes to a Discussion or Pull Request"""
        if not isinstance(discussion_num, int) or discussion_num <= 0:
            raise ValueError("Invalid discussion_num, must be a positive integer")
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        repo_id = f"{repo_type}s/{repo_id}"

        path = f"{self.endpoint}/api/{repo_id}/discussions/{discussion_num}/{resource}"

        headers = self._build_hf_headers(token=token, is_write_action=True)
        resp = requests.post(path, headers=headers, json=body)
        hf_raise_for_status(resp)
        return resp

    @validate_hf_hub_args
    def comment_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        comment: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Creates a new comment on the given Discussion.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment (`str`):
                The content of the comment to create. Comments support markdown formatting.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the newly created comment


        Examples:
            ```python

            >>> comment = \"\"\"
            ... Hello @otheruser!
            ...
            ... # This is a title
            ...
            ... **This is bold**, *this is italic* and ~this is strikethrough~
            ... And [this](http://url) is a link
            ... \"\"\"

            >>> HfApi().comment_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     comment=comment
            ... )
            # DiscussionComment(id='deadbeef0000000', type='comment', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="comment",
            body={"comment": comment},
        )
        return deserialize_event(resp.json()["newMessage"])  # type: ignore

    @validate_hf_hub_args
    def rename_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        new_title: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionTitleChange:
        """Renames a Discussion.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            new_title (`str`):
                The new title for the discussion
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionTitleChange`]: the title change event


        Examples:
            ```python
            >>> new_title = "New title, fixing a typo"
            >>> HfApi().rename_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     new_title=new_title
            ... )
            # DiscussionTitleChange(id='deadbeef0000000', type='title-change', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="title",
            body={"title": new_title},
        )
        return deserialize_event(resp.json()["newTitle"])  # type: ignore

    @validate_hf_hub_args
    def change_discussion_status(
        self,
        repo_id: str,
        discussion_num: int,
        new_status: Literal["open", "closed"],
        *,
        token: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionStatusChange:
        """Closes or re-opens a Discussion or Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            new_status (`str`):
                The new status for the discussion, either `"open"` or `"closed"`.
            comment (`str`, *optional*):
                An optional comment to post with the status change.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionStatusChange`]: the status change event


        Examples:
            ```python
            >>> new_title = "New title, fixing a typo"
            >>> HfApi().rename_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     new_title=new_title
            ... )
            # DiscussionStatusChange(id='deadbeef0000000', type='status-change', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if new_status not in ["open", "closed"]:
            raise ValueError("Invalid status, valid statuses are: 'open' and 'closed'")
        body: Dict[str, str] = {"status": new_status}
        if comment and comment.strip():
            body["comment"] = comment.strip()
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="status",
            body=body,
        )
        return deserialize_event(resp.json()["newStatus"])  # type: ignore

    @validate_hf_hub_args
    def merge_pull_request(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        token: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """Merges a Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment (`str`, *optional*):
                An optional comment to post with the status change.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionStatusChange`]: the status change event

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="merge",
            body={"comment": comment.strip()} if comment and comment.strip() else None,
        )

    @validate_hf_hub_args
    def edit_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        new_content: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Edits a comment on a Discussion / Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            new_content (`str`):
                The new content of the comment. Comments support markdown formatting.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the edited comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource=f"comment/{comment_id.lower()}/edit",
            body={"content": new_content},
        )
        return deserialize_event(resp.json()["updatedComment"])  # type: ignore

    @validate_hf_hub_args
    def hide_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Hides a comment on a Discussion / Pull Request.

        <Tip warning={true}>
        Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.
        </Tip>

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the hidden comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        warnings.warn(
            "Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.",
            UserWarning,
        )
        resp = self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource=f"comment/{comment_id.lower()}/hide",
        )
        return deserialize_event(resp.json()["updatedComment"])  # type: ignore

    @validate_hf_hub_args
    def add_space_secret(self, repo_id: str, key: str, value: str, *, token: Optional[str] = None) -> None:
        """Adds or updates a secret in a Space.

        Secrets allow to set secret keys or tokens to a Space without hardcoding them.
        For more details, see https://huggingface.co/docs/hub/spaces-overview#managing-secrets.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            key (`str`):
                Secret key. Example: `"GITHUB_API_KEY"`
            value (`str`):
                Secret value. Example: `"your_github_api_key"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
        r = get_session().post(
            f"{self.endpoint}/api/spaces/{repo_id}/secrets",
            headers=self._build_hf_headers(token=token),
            json={"key": key, "value": value},
        )
        hf_raise_for_status(r)

    @validate_hf_hub_args
    def delete_space_secret(self, repo_id: str, key: str, *, token: Optional[str] = None) -> None:
        """Deletes a secret from a Space.

        Secrets allow to set secret keys or tokens to a Space without hardcoding them.
        For more details, see https://huggingface.co/docs/hub/spaces-overview#managing-secrets.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            key (`str`):
                Secret key. Example: `"GITHUB_API_KEY"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
        r = get_session().delete(
            f"{self.endpoint}/api/spaces/{repo_id}/secrets",
            headers=self._build_hf_headers(token=token),
            json={"key": key},
        )
        hf_raise_for_status(r)

    @validate_hf_hub_args
    def get_space_runtime(self, repo_id: str, *, token: Optional[str] = None) -> SpaceRuntime:
        """Gets runtime information about a Space.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if
                not provided.
        Returns:
            [`SpaceRuntime`]: Runtime information about a Space including Space stage and hardware.
        """
        r = get_session().get(
            f"{self.endpoint}/api/spaces/{repo_id}/runtime", headers=self._build_hf_headers(token=token)
        )
        hf_raise_for_status(r)
        return SpaceRuntime(r.json())

    @validate_hf_hub_args
    def request_space_hardware(
        self,
        repo_id: str,
        hardware: SpaceHardware,
        *,
        token: Optional[str] = None,
        sleep_time: Optional[int] = None,
    ) -> SpaceRuntime:
        """Request new hardware for a Space.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            hardware (`str` or [`SpaceHardware`]):
                Hardware on which to run the Space. Example: `"t4-medium"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
            sleep_time (`int`, *optional*):
                Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don't want
                your Space to sleep (default behavior for upgraded hardware). For free hardware, you can't configure
                the sleep time (value is fixed to 48 hours of inactivity).
                See https://huggingface.co/docs/hub/spaces-gpus#sleep-time for more details.
        Returns:
            [`SpaceRuntime`]: Runtime information about a Space including Space stage and hardware.

        <Tip>

        It is also possible to request hardware directly when creating the Space repo! See [`create_repo`] for details.

        </Tip>
        """
        if sleep_time is not None and hardware == SpaceHardware.CPU_BASIC:
            warnings.warn(
                (
                    "If your Space runs on the default 'cpu-basic' hardware, it will go to sleep if inactive for more"
                    " than 48 hours. This value is not configurable. If you don't want your Space to deactivate or if"
                    " you want to set a custom sleep time, you need to upgrade to a paid Hardware."
                ),
                UserWarning,
            )
        payload: Dict[str, Any] = {"flavor": hardware}
        if sleep_time is not None:
            payload["sleepTimeSeconds"] = sleep_time
        r = get_session().post(
            f"{self.endpoint}/api/spaces/{repo_id}/hardware",
            headers=self._build_hf_headers(token=token),
            json=payload,
        )
        hf_raise_for_status(r)
        return SpaceRuntime(r.json())

    @validate_hf_hub_args
    def set_space_sleep_time(self, repo_id: str, sleep_time: int, *, token: Optional[str] = None) -> SpaceRuntime:
        """Set a custom sleep time for a Space running on upgraded hardware..

        Your Space will go to sleep after X seconds of inactivity. You are not billed when your Space is in "sleep"
        mode. If a new visitor lands on your Space, it will "wake it up". Only upgraded hardware can have a
        configurable sleep time. To know more about the sleep stage, please refer to
        https://huggingface.co/docs/hub/spaces-gpus#sleep-time.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            sleep_time (`int`, *optional*):
                Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don't want
                your Space to pause (default behavior for upgraded hardware). For free hardware, you can't configure
                the sleep time (value is fixed to 48 hours of inactivity).
                See https://huggingface.co/docs/hub/spaces-gpus#sleep-time for more details.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
        Returns:
            [`SpaceRuntime`]: Runtime information about a Space including Space stage and hardware.

        <Tip>

        It is also possible to set a custom sleep time when requesting hardware with [`request_space_hardware`].

        </Tip>
        """
        r = get_session().post(
            f"{self.endpoint}/api/spaces/{repo_id}/sleeptime",
            headers=self._build_hf_headers(token=token),
            json={"seconds": sleep_time},
        )
        hf_raise_for_status(r)
        runtime = SpaceRuntime(r.json())

        hardware = runtime.requested_hardware or runtime.hardware
        if hardware == SpaceHardware.CPU_BASIC:
            warnings.warn(
                (
                    "If your Space runs on the default 'cpu-basic' hardware, it will go to sleep if inactive for more"
                    " than 48 hours. This value is not configurable. If you don't want your Space to deactivate or if"
                    " you want to set a custom sleep time, you need to upgrade to a paid Hardware."
                ),
                UserWarning,
            )
        return runtime

    @validate_hf_hub_args
    def pause_space(self, repo_id: str, *, token: Optional[str] = None) -> SpaceRuntime:
        """Pause your Space.

        A paused Space stops executing until manually restarted by its owner. This is different from the sleeping
        state in which free Spaces go after 48h of inactivity. Paused time is not billed to your account, no matter the
        hardware you've selected. To restart your Space, use [`restart_space`] and go to your Space settings page.

        For more details, please visit [the docs](https://huggingface.co/docs/hub/spaces-gpus#pause).

        Args:
            repo_id (`str`):
                ID of the Space to pause. Example: `"Salesforce/BLIP2"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Returns:
            [`SpaceRuntime`]: Runtime information about your Space including `stage=PAUSED` and requested hardware.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If your Space is not found (error 404). Most probably wrong repo_id or your space is private but you
                are not authenticated.
            [`~utils.HfHubHTTPError`]:
                403 Forbidden: only the owner of a Space can pause it. If you want to manage a Space that you don't
                own, either ask the owner by opening a Discussion or duplicate the Space.
            [`~utils.BadRequestError`]:
                If your Space is a static Space. Static Spaces are always running and never billed. If you want to hide
                a static Space, you can set it to private.
        """
        r = get_session().post(
            f"{self.endpoint}/api/spaces/{repo_id}/pause", headers=self._build_hf_headers(token=token)
        )
        hf_raise_for_status(r)
        return SpaceRuntime(r.json())

    @validate_hf_hub_args
    def restart_space(self, repo_id: str, *, token: Optional[str] = None) -> SpaceRuntime:
        """Restart your Space.

        This is the only way to programmatically restart a Space if you've put it on Pause (see [`pause_space`]). You
        must be the owner of the Space to restart it. If you are using an upgraded hardware, your account will be
        billed as soon as the Space is restarted. You can trigger a restart no matter the current state of a Space.

        For more details, please visit [the docs](https://huggingface.co/docs/hub/spaces-gpus#pause).

        Args:
            repo_id (`str`):
                ID of the Space to restart. Example: `"Salesforce/BLIP2"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Returns:
            [`SpaceRuntime`]: Runtime information about your Space.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If your Space is not found (error 404). Most probably wrong repo_id or your space is private but you
                are not authenticated.
            [`~utils.HfHubHTTPError`]:
                403 Forbidden: only the owner of a Space can restart it. If you want to restart a Space that you don't
                own, either ask the owner by opening a Discussion or duplicate the Space.
            [`~utils.BadRequestError`]:
                If your Space is a static Space. Static Spaces are always running and never billed. If you want to hide
                a static Space, you can set it to private.
        """
        r = get_session().post(
            f"{self.endpoint}/api/spaces/{repo_id}/restart", headers=self._build_hf_headers(token=token)
        )
        hf_raise_for_status(r)
        return SpaceRuntime(r.json())

    @validate_hf_hub_args
    def duplicate_space(
        self,
        from_id: str,
        to_id: Optional[str] = None,
        *,
        private: Optional[bool] = None,
        token: Optional[str] = None,
        exist_ok: bool = False,
    ) -> RepoUrl:
        """Duplicate a Space.

        Programmatically duplicate a Space. The new Space will be created in your account and will be in the same state
        as the original Space (running or paused). You can duplicate a Space no matter the current state of a Space.

        Args:
            from_id (`str`):
                ID of the Space to duplicate. Example: `"pharma/CLIP-Interrogator"`.
            to_id (`str`, *optional*):
                ID of the new Space. Example: `"dog/CLIP-Interrogator"`. If not provided, the new Space will have the same
                name as the original Space, but in your account.
            private (`bool`, *optional*):
                Whether the new Space should be private or not. Defaults to the same privacy as the original Space.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if repo already exists.

        Returns:
            [`RepoUrl`]: URL to the newly created repo. Value is a subclass of `str` containing
            attributes like `endpoint`, `repo_type` and `repo_id`.

        Raises:
            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`~utils.RepositoryNotFoundError`]
              If one of `from_id` or `to_id` cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        Example:
        ```python
        >>> from huggingface_hub import duplicate_space

        # Duplicate a Space to your account
        >>> duplicate_space("multimodalart/dreambooth-training")
        RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)

        # Can set custom destination id and visibility flag.
        >>> duplicate_space("multimodalart/dreambooth-training", to_id="my-dreambooth", private=True)
        RepoUrl('https://huggingface.co/spaces/nateraw/my-dreambooth',...)
        ```
        """
        # Parse to_id if provided
        parsed_to_id = RepoUrl(to_id) if to_id is not None else None

        # Infer target repo_id
        to_namespace = (  # set namespace manually or default to username
            parsed_to_id.namespace
            if parsed_to_id is not None and parsed_to_id.namespace is not None
            else self.whoami(token)["name"]
        )
        to_repo_name = parsed_to_id.repo_name if to_id is not None else RepoUrl(from_id).repo_name  # type: ignore

        # repository must be a valid repo_id (namespace/repo_name).
        payload: Dict[str, Any] = {"repository": f"{to_namespace}/{to_repo_name}"}

        # private is optional with this endpoint, with None defaulting to the original space's privacy.
        if private is not None:
            payload["private"] = private

        r = get_session().post(
            f"{self.endpoint}/api/spaces/{from_id}/duplicate",
            headers=self._build_hf_headers(token=token, is_write_action=True),
            json=payload,
        )

        try:
            hf_raise_for_status(r)
        except HTTPError as err:
            if exist_ok and err.response.status_code == 409:
                # Repo already exists and `exist_ok=True`
                pass
            else:
                raise

        return RepoUrl(r.json()["url"], endpoint=self.endpoint)

    def _build_hf_headers(
        self,
        token: Optional[Union[bool, str]] = None,
        is_write_action: bool = False,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
    ) -> Dict[str, str]:
        """
        Alias for [`build_hf_headers`] that uses the token from [`HfApi`] client
        when `token` is not provided.
        """
        if token is None:
            # Cannot do `token = token or self.token` as token can be `False`.
            token = self.token
        return build_hf_headers(
            token=token,
            is_write_action=is_write_action,
            library_name=library_name or self.library_name,
            library_version=library_version or self.library_version,
            user_agent=user_agent or self.user_agent,
        )

    def _prepare_upload_folder_deletions(
        self,
        repo_id: str,
        repo_type: Optional[str],
        revision: Optional[str],
        token: Optional[str],
        path_in_repo: str,
        delete_patterns: Optional[Union[List[str], str]],
    ) -> List[CommitOperationDelete]:
        """Generate the list of Delete operations for a commit to delete files from a repo.

        List remote files and match them against the `delete_patterns` constraints. Returns a list of [`CommitOperationDelete`]
        with the matching items.

        Note: `.gitattributes` file is essential to make a repo work properly on the Hub. This file will always be
              kept even if it matches the `delete_patterns` constraints.
        """
        if delete_patterns is None:
            # If no delete patterns, no need to list and filter remote files
            return []

        # List remote files
        filenames = self.list_repo_files(repo_id=repo_id, revision=revision, repo_type=repo_type, token=token)

        # Compute relative path in repo
        if path_in_repo:
            path_in_repo = path_in_repo.strip("/") + "/"  # harmonize
            relpath_to_abspath = {
                file[len(path_in_repo) :]: file for file in filenames if file.startswith(path_in_repo)
            }
        else:
            relpath_to_abspath = {file: file for file in filenames}

        # Apply filter on relative paths and return
        return [
            CommitOperationDelete(path_in_repo=relpath_to_abspath[relpath], is_folder=False)
            for relpath in filter_repo_objects(relpath_to_abspath.keys(), allow_patterns=delete_patterns)
            if relpath_to_abspath[relpath] != ".gitattributes"
        ]


def _prepare_upload_folder_additions(
    folder_path: Union[str, Path],
    path_in_repo: str,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
) -> List[CommitOperationAdd]:
    """Generate the list of Add operations for a commit to upload a folder.

    Files not matching the `allow_patterns` (allowlist) and `ignore_patterns` (denylist)
    constraints are discarded.
    """
    folder_path = Path(folder_path).expanduser().resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")

    # List files from folder
    relpath_to_abspath = {
        path.relative_to(folder_path).as_posix(): path
        for path in sorted(folder_path.glob("**/*"))  # sorted to be deterministic
        if path.is_file()
    }

    # Filter files and return
    # Patterns are applied on the path relative to `folder_path`. `path_in_repo` is prefixed after the filtering.
    prefix = f"{path_in_repo.strip('/')}/" if path_in_repo else ""
    return [
        CommitOperationAdd(
            path_or_fileobj=relpath_to_abspath[relpath],  # absolute path on disk
            path_in_repo=prefix + relpath,  # "absolute" path in repo
        )
        for relpath in filter_repo_objects(
            relpath_to_abspath.keys(), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )
    ]


def _parse_revision_from_pr_url(pr_url: str) -> str:
    """Safely parse revision number from a PR url.

    Example:
    ```py
    >>> _parse_revision_from_pr_url("https://huggingface.co/bigscience/bloom/discussions/2")
    "refs/pr/2"
    ```
    """
    re_match = re.match(_REGEX_DISCUSSION_URL, pr_url)
    if re_match is None:
        raise RuntimeError(f"Unexpected response from the hub, expected a Pull Request URL but got: '{pr_url}'")
    return f"refs/pr/{re_match[1]}"


api = HfApi()

whoami = api.whoami
get_token_permission = api.get_token_permission

list_models = api.list_models
model_info = api.model_info

list_datasets = api.list_datasets
dataset_info = api.dataset_info

list_spaces = api.list_spaces
space_info = api.space_info

repo_info = api.repo_info
list_repo_files = api.list_repo_files
list_repo_refs = api.list_repo_refs
list_repo_commits = api.list_repo_commits
list_files_info = api.list_files_info

list_metrics = api.list_metrics

get_model_tags = api.get_model_tags
get_dataset_tags = api.get_dataset_tags

create_commit = api.create_commit
create_repo = api.create_repo
delete_repo = api.delete_repo
update_repo_visibility = api.update_repo_visibility
move_repo = api.move_repo
upload_file = api.upload_file
upload_folder = api.upload_folder
delete_file = api.delete_file
delete_folder = api.delete_folder
create_commits_on_pr = api.create_commits_on_pr
create_branch = api.create_branch
delete_branch = api.delete_branch
create_tag = api.create_tag
delete_tag = api.delete_tag
get_full_repo_name = api.get_full_repo_name

# Background jobs
run_as_future = api.run_as_future

# Activity API
list_liked_repos = api.list_liked_repos
like = api.like
unlike = api.unlike

# Community API
get_discussion_details = api.get_discussion_details
get_repo_discussions = api.get_repo_discussions
create_discussion = api.create_discussion
create_pull_request = api.create_pull_request
change_discussion_status = api.change_discussion_status
comment_discussion = api.comment_discussion
edit_discussion_comment = api.edit_discussion_comment
rename_discussion = api.rename_discussion
merge_pull_request = api.merge_pull_request

# Space API
add_space_secret = api.add_space_secret
delete_space_secret = api.delete_space_secret
get_space_runtime = api.get_space_runtime
request_space_hardware = api.request_space_hardware
set_space_sleep_time = api.set_space_sleep_time
pause_space = api.pause_space
restart_space = api.restart_space
duplicate_space = api.duplicate_space
