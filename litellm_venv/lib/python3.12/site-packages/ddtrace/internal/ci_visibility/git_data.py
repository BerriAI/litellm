import dataclasses
import typing as t

from ddtrace.ext import ci


@dataclasses.dataclass(frozen=True)
class GitData:
    repository_url: t.Optional[str]
    branch: t.Optional[str]
    commit_sha: t.Optional[str]


def get_git_data_from_tags(tags: t.Dict[str, t.Any]) -> GitData:
    return GitData(
        tags.get(ci.git.REPOSITORY_URL),
        tags.get(ci.git.BRANCH),
        tags.get(ci.git.COMMIT_SHA),
    )
