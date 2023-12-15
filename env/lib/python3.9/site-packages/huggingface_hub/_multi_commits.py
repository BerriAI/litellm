# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
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
"""Contains utilities to multi-commits (i.e. push changes iteratively on a PR)."""
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple, Union

from ._commit_api import CommitOperationAdd, CommitOperationDelete
from .community import DiscussionWithDetails
from .utils import experimental
from .utils._cache_manager import _format_size


if TYPE_CHECKING:
    from .hf_api import HfApi


class MultiCommitException(Exception):
    """Base exception for any exception happening while doing a multi-commit."""


MULTI_COMMIT_PR_DESCRIPTION_TEMPLATE = """
## {commit_message}

{commit_description}

**Multi commit ID:** {multi_commit_id}

Scheduled commits:

{multi_commit_strategy}

_This is a PR opened using the `huggingface_hub` library in the context of a multi-commit. PR can be commented as a usual PR. However, please be aware that manually updating the PR description, changing the PR status, or pushing new commits, is not recommended as it might corrupt the commit process. Learn more about multi-commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

MULTI_COMMIT_PR_COMPLETION_COMMENT_TEMPLATE = """
Multi-commit is now completed! You can ping the repo owner to review the changes. This PR can now be commented or modified without risking to corrupt it.

_This is a comment posted using the `huggingface_hub` library in the context of a multi-commit. Learn more about multi-commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

MULTI_COMMIT_PR_CLOSING_COMMENT_TEMPLATE = """
`create_pr=False` has been passed so PR is automatically merged.

_This is a comment posted using the `huggingface_hub` library in the context of a multi-commit. Learn more about multi-commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

MULTI_COMMIT_PR_CLOSE_COMMENT_FAILURE_NO_CHANGES_TEMPLATE = """
Cannot merge Pull Requests as no changes are associated. This PR will be closed automatically.

_This is a comment posted using the `huggingface_hub` library in the context of a multi-commit. Learn more about multi-commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""

MULTI_COMMIT_PR_CLOSE_COMMENT_FAILURE_BAD_REQUEST_TEMPLATE = """
An error occurred while trying to merge the Pull Request: `{error_message}`.

_This is a comment posted using the `huggingface_hub` library in the context of a multi-commit. Learn more about multi-commits [in this guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)._
"""


STEP_ID_REGEX = re.compile(r"- \[(?P<completed>[ |x])\].*(?P<step_id>[a-fA-F0-9]{64})", flags=re.MULTILINE)


@experimental
def plan_multi_commits(
    operations: Iterable[Union[CommitOperationAdd, CommitOperationDelete]],
    max_operations_per_commit: int = 50,
    max_upload_size_per_commit: int = 2 * 1024 * 1024 * 1024,
) -> Tuple[List[List[CommitOperationAdd]], List[List[CommitOperationDelete]]]:
    """Split a list of operations in a list of commits to perform.

    Implementation follows a sub-optimal (yet simple) algorithm:
    1. Delete operations are grouped together by commits of maximum `max_operations_per_commits` operations.
    2. All additions exceeding `max_upload_size_per_commit` are committed 1 by 1.
    3. All remaining additions are grouped together and split each time the `max_operations_per_commit` or the
       `max_upload_size_per_commit` limit is reached.

    We do not try to optimize the splitting to get the lowest number of commits as this is a NP-hard problem (see
    [bin packing problem](https://en.wikipedia.org/wiki/Bin_packing_problem)). For our use case, it is not problematic
    to use a sub-optimal solution so we favored an easy-to-explain implementation.

    Args:
        operations (`List` of [`~hf_api.CommitOperation`]):
            The list of operations to split into commits.
        max_operations_per_commit (`int`):
            Maximum number of operations in a single commit. Defaults to 50.
        max_upload_size_per_commit (`int`):
            Maximum size to upload (in bytes) in a single commit. Defaults to 2GB. Files bigger than this limit are
            uploaded, 1 per commit.

    Returns:
        `Tuple[List[List[CommitOperationAdd]], List[List[CommitOperationDelete]]]`: a tuple. First item is a list of
        lists of [`CommitOperationAdd`] representing the addition commits to push. The second item is a list of lists
        of [`CommitOperationDelete`] representing the deletion commits.

    <Tip warning={true}>

    `plan_multi_commits` is experimental. Its API and behavior is subject to change in the future without prior notice.

    </Tip>

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

    <Tip warning={true}>

    The initial order of the operations is not guaranteed! All deletions will be performed before additions. If you are
    not updating multiple times the same file, you are fine.

    </Tip>
    """
    addition_commits: List[List[CommitOperationAdd]] = []
    deletion_commits: List[List[CommitOperationDelete]] = []

    additions: List[CommitOperationAdd] = []
    additions_size = 0
    deletions: List[CommitOperationDelete] = []
    for op in operations:
        if isinstance(op, CommitOperationDelete):
            # Group delete operations together
            deletions.append(op)
            if len(deletions) >= max_operations_per_commit:
                deletion_commits.append(deletions)
                deletions = []

        elif op.upload_info.size >= max_upload_size_per_commit:
            # Upload huge files 1 by 1
            addition_commits.append([op])

        elif additions_size + op.upload_info.size < max_upload_size_per_commit:
            # Group other additions and split if size limit is reached (either max_nb_files or max_upload_size)
            additions.append(op)
            additions_size += op.upload_info.size

        else:
            addition_commits.append(additions)
            additions = [op]
            additions_size = op.upload_info.size

        if len(additions) >= max_operations_per_commit:
            addition_commits.append(additions)
            additions = []
            additions_size = 0

    if len(additions) > 0:
        addition_commits.append(additions)
    if len(deletions) > 0:
        deletion_commits.append(deletions)

    return addition_commits, deletion_commits


@dataclass
class MultiCommitStep:
    """Dataclass containing a list of CommitOperation to commit at once.

    A [`MultiCommitStep`] is one atomic part of a [`MultiCommitStrategy`]. Each step is identified by its own
    deterministic ID based on the list of commit operations (hexadecimal sha256). ID is persistent between re-runs if
    the list of commits is kept the same.
    """

    operations: List[Union[CommitOperationAdd, CommitOperationDelete]]

    id: str = field(init=False)
    completed: bool = False

    def __post_init__(self) -> None:
        if len(self.operations) == 0:
            raise ValueError("A MultiCommitStep must have at least 1 commit operation, got 0.")

        # Generate commit id
        sha = sha256()
        for op in self.operations:
            if isinstance(op, CommitOperationAdd):
                sha.update(b"ADD")
                sha.update(op.path_in_repo.encode())
                sha.update(op.upload_info.sha256)
            elif isinstance(op, CommitOperationDelete):
                sha.update(b"DELETE")
                sha.update(op.path_in_repo.encode())
                sha.update(str(op.is_folder).encode())
            else:
                NotImplementedError()
        self.id = sha.hexdigest()

    def __str__(self) -> str:
        """Format a step for PR description.

        Formatting can be changed in the future as long as it is single line, starts with `- [ ]`/`- [x]` and contains
        `self.id`. Must be able to match `STEP_ID_REGEX`.
        """
        additions = [op for op in self.operations if isinstance(op, CommitOperationAdd)]
        file_deletions = [op for op in self.operations if isinstance(op, CommitOperationDelete) and not op.is_folder]
        folder_deletions = [op for op in self.operations if isinstance(op, CommitOperationDelete) and op.is_folder]
        if len(additions) > 0:
            return (
                f"- [{'x' if self.completed else ' '}] Upload {len(additions)} file(s) "
                f"totalling {_format_size(sum(add.upload_info.size for add in additions))}"
                f" ({self.id})"
            )
        else:
            return (
                f"- [{'x' if self.completed else ' '}] Delete {len(file_deletions)} file(s) and"
                f" {len(folder_deletions)} folder(s) ({self.id})"
            )


@dataclass
class MultiCommitStrategy:
    """Dataclass containing a list of [`MultiCommitStep`] to commit iteratively.

    A strategy is identified by its own deterministic ID based on the list of its steps (hexadecimal sha256). ID is
    persistent between re-runs if the list of commits is kept the same.
    """

    addition_commits: List[MultiCommitStep]
    deletion_commits: List[MultiCommitStep]

    id: str = field(init=False)
    all_steps: Set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.all_steps = {step.id for step in self.addition_commits + self.deletion_commits}
        if len(self.all_steps) < len(self.addition_commits) + len(self.deletion_commits):
            raise ValueError("Got duplicate commits in MultiCommitStrategy. All commits must be unique.")

        if len(self.all_steps) == 0:
            raise ValueError("A MultiCommitStrategy must have at least 1 commit, got 0.")

        # Generate strategy id
        sha = sha256()
        for step in self.addition_commits + self.deletion_commits:
            sha.update("new step".encode())
            sha.update(step.id.encode())
        self.id = sha.hexdigest()


def multi_commit_create_pull_request(
    api: "HfApi",
    repo_id: str,
    commit_message: str,
    commit_description: Optional[str],
    strategy: MultiCommitStrategy,
    token: Optional[str],
    repo_type: Optional[str],
) -> DiscussionWithDetails:
    return api.create_pull_request(
        repo_id=repo_id,
        title=f"[WIP] {commit_message} (multi-commit {strategy.id})",
        description=multi_commit_generate_comment(
            commit_message=commit_message, commit_description=commit_description, strategy=strategy
        ),
        token=token,
        repo_type=repo_type,
    )


def multi_commit_generate_comment(
    commit_message: str,
    commit_description: Optional[str],
    strategy: MultiCommitStrategy,
) -> str:
    return MULTI_COMMIT_PR_DESCRIPTION_TEMPLATE.format(
        commit_message=commit_message,
        commit_description=commit_description or "",
        multi_commit_id=strategy.id,
        multi_commit_strategy="\n".join(
            str(commit) for commit in strategy.deletion_commits + strategy.addition_commits
        ),
    )


def multi_commit_parse_pr_description(description: str) -> Set[str]:
    return {match[1] for match in STEP_ID_REGEX.findall(description)}
