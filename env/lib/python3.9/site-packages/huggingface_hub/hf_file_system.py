import itertools
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from glob import has_magic
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, unquote

import fsspec

from ._commit_api import CommitOperationCopy, CommitOperationDelete
from .constants import DEFAULT_REVISION, ENDPOINT, REPO_TYPE_MODEL, REPO_TYPES_MAPPING, REPO_TYPES_URL_PREFIXES
from .hf_api import HfApi
from .utils import (
    EntryNotFoundError,
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    hf_raise_for_status,
    http_backoff,
    paginate,
    parse_datetime,
)


@dataclass
class HfFileSystemResolvedPath:
    """Data structure containing information about a resolved Hugging Face file system path."""

    repo_type: str
    repo_id: str
    revision: str
    path_in_repo: str

    def unresolve(self) -> str:
        return (
            f"{REPO_TYPES_URL_PREFIXES.get(self.repo_type, '') + self.repo_id}@{safe_quote(self.revision)}/{self.path_in_repo}"
            .rstrip("/")
        )


class HfFileSystem(fsspec.AbstractFileSystem):
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    Args:
        endpoint (`str`, *optional*):
            The endpoint to use. If not provided, the default one (https://huggingface.co) is used.
        token (`str`, *optional*):
            Authentication token, obtained with [`HfApi.login`] method. Will default to the stored token.

    Usage:

    ```python
    >>> from huggingface_hub import HfFileSystem

    >>> fs = HfFileSystem()

    >>> # List files
    >>> fs.glob("my-username/my-model/*.bin")
    ['my-username/my-model/pytorch_model.bin']
    >>> fs.ls("datasets/my-username/my-dataset", detail=False)
    ['datasets/my-username/my-dataset/.gitattributes', 'datasets/my-username/my-dataset/README.md', 'datasets/my-username/my-dataset/data.json']

    >>> # Read/write files
    >>> with fs.open("my-username/my-model/pytorch_model.bin") as f:
    ...     data = f.read()
    >>> with fs.open("my-username/my-model/pytorch_model.bin", "wb") as f:
    ...     f.write(data)
    ```
    """

    root_marker = ""
    protocol = "hf"

    def __init__(
        self,
        *args,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        **storage_options,
    ):
        super().__init__(*args, **storage_options)
        self.endpoint = endpoint or ENDPOINT
        self.token = token
        self._api = HfApi(endpoint=endpoint, token=token)
        # Maps (repo_type, repo_id, revision) to a 2-tuple with:
        #  * the 1st element indicating whether the repositoy and the revision exist
        #  * the 2nd element being the exception raised if the repository or revision doesn't exist
        self._repo_and_revision_exists_cache: Dict[
            Tuple[str, str, Optional[str]], Tuple[bool, Optional[Exception]]
        ] = {}

    def _repo_and_revision_exist(
        self, repo_type: str, repo_id: str, revision: Optional[str]
    ) -> Tuple[bool, Optional[Exception]]:
        if (repo_type, repo_id, revision) not in self._repo_and_revision_exists_cache:
            try:
                self._api.repo_info(repo_id, revision=revision, repo_type=repo_type)
            except (RepositoryNotFoundError, HFValidationError) as e:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = False, e
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = False, e
            except RevisionNotFoundError as e:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = False, e
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = True, None
            else:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = True, None
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = True, None
        return self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)]

    def resolve_path(self, path: str, revision: Optional[str] = None) -> HfFileSystemResolvedPath:
        def _align_revision_in_path_with_revision(
            revision_in_path: Optional[str], revision: Optional[str]
        ) -> Optional[str]:
            if revision is not None:
                if revision_in_path is not None and revision_in_path != revision:
                    raise ValueError(
                        f'Revision specified in path ("{revision_in_path}") and in `revision` argument ("{revision}")'
                        " are not the same."
                    )
            else:
                revision = revision_in_path
            return revision

        path = self._strip_protocol(path)
        if not path:
            # can't list repositories at root
            raise NotImplementedError("Access to repositories lists is not implemented.")
        elif path.split("/")[0] + "/" in REPO_TYPES_URL_PREFIXES.values():
            if "/" not in path:
                # can't list repositories at the repository type level
                raise NotImplementedError("Acces to repositories lists is not implemented.")
            repo_type, path = path.split("/", 1)
            repo_type = REPO_TYPES_MAPPING[repo_type]
        else:
            repo_type = REPO_TYPE_MODEL
        if path.count("/") > 0:
            if "@" in path:
                repo_id, revision_in_path = path.split("@", 1)
                if "/" in revision_in_path:
                    revision_in_path, path_in_repo = revision_in_path.split("/", 1)
                else:
                    path_in_repo = ""
                revision_in_path = unquote(revision_in_path)
                revision = _align_revision_in_path_with_revision(revision_in_path, revision)
                repo_and_revision_exist, err = self._repo_and_revision_exist(repo_type, repo_id, revision)
                if not repo_and_revision_exist:
                    raise FileNotFoundError(path) from err
            else:
                repo_id_with_namespace = "/".join(path.split("/")[:2])
                path_in_repo_with_namespace = "/".join(path.split("/")[2:])
                repo_id_without_namespace = path.split("/")[0]
                path_in_repo_without_namespace = "/".join(path.split("/")[1:])
                repo_id = repo_id_with_namespace
                path_in_repo = path_in_repo_with_namespace
                repo_and_revision_exist, err = self._repo_and_revision_exist(repo_type, repo_id, revision)
                if not repo_and_revision_exist:
                    if isinstance(err, (RepositoryNotFoundError, HFValidationError)):
                        repo_id = repo_id_without_namespace
                        path_in_repo = path_in_repo_without_namespace
                        repo_and_revision_exist, _ = self._repo_and_revision_exist(repo_type, repo_id, revision)
                        if not repo_and_revision_exist:
                            raise FileNotFoundError(path) from err
                    else:
                        raise FileNotFoundError(path) from err
        else:
            repo_id = path
            path_in_repo = ""
            if "@" in path:
                repo_id, revision_in_path = path.split("@", 1)
                revision_in_path = unquote(revision_in_path)
                revision = _align_revision_in_path_with_revision(revision_in_path, revision)
            repo_and_revision_exist, _ = self._repo_and_revision_exist(repo_type, repo_id, revision)
            if not repo_and_revision_exist:
                raise NotImplementedError("Acces to repositories lists is not implemented.")

        revision = revision if revision is not None else DEFAULT_REVISION
        return HfFileSystemResolvedPath(repo_type, repo_id, revision, path_in_repo)

    def invalidate_cache(self, path: Optional[str] = None) -> None:
        if not path:
            self.dircache.clear()
            self._repository_type_and_id_exists_cache.clear()
        else:
            path = self.resolve_path(path).unresolve()
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        revision: Optional[str] = None,
        **kwargs,
    ) -> "HfFileSystemFile":
        if mode == "ab":
            raise NotImplementedError("Appending to remote files is not yet supported.")
        return HfFileSystemFile(self, path, mode=mode, revision=revision, **kwargs)

    def _rm(self, path: str, revision: Optional[str] = None, **kwargs) -> None:
        resolved_path = self.resolve_path(path, revision=revision)
        self._api.delete_file(
            path_in_repo=resolved_path.path_in_repo,
            repo_id=resolved_path.repo_id,
            token=self.token,
            repo_type=resolved_path.repo_type,
            revision=resolved_path.revision,
            commit_message=kwargs.get("commit_message"),
            commit_description=kwargs.get("commit_description"),
        )
        self.invalidate_cache(path=resolved_path.unresolve())

    def rm(
        self,
        path: str,
        recursive: bool = False,
        maxdepth: Optional[int] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> None:
        resolved_path = self.resolve_path(path, revision=revision)
        root_path = REPO_TYPES_URL_PREFIXES.get(resolved_path.repo_type, "") + resolved_path.repo_id
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth, revision=resolved_path.revision)
        paths_in_repo = [path[len(root_path) + 1 :] for path in paths if not self.isdir(path)]
        operations = [CommitOperationDelete(path_in_repo=path_in_repo) for path_in_repo in paths_in_repo]
        commit_message = f"Delete {path} "
        commit_message += "recursively " if recursive else ""
        commit_message += f"up to depth {maxdepth} " if maxdepth is not None else ""
        # TODO: use `commit_description` to list all the deleted paths?
        self._api.create_commit(
            repo_id=resolved_path.repo_id,
            repo_type=resolved_path.repo_type,
            token=self.token,
            operations=operations,
            revision=resolved_path.revision,
            commit_message=kwargs.get("commit_message", commit_message),
            commit_description=kwargs.get("commit_description"),
        )
        self.invalidate_cache(path=resolved_path.unresolve())

    def ls(
        self, path: str, detail: bool = True, refresh: bool = False, revision: Optional[str] = None, **kwargs
    ) -> List[Union[str, Dict[str, Any]]]:
        """List the contents of a directory."""
        resolved_path = self.resolve_path(path, revision=revision)
        revision_in_path = "@" + safe_quote(resolved_path.revision)
        has_revision_in_path = revision_in_path in path
        path = resolved_path.unresolve()
        if path not in self.dircache or refresh:
            path_prefix = (
                HfFileSystemResolvedPath(
                    resolved_path.repo_type, resolved_path.repo_id, resolved_path.revision, ""
                ).unresolve()
                + "/"
            )
            tree_path = path
            tree_iter = self._iter_tree(tree_path, revision=resolved_path.revision)
            try:
                tree_item = next(tree_iter)
            except EntryNotFoundError:
                if "/" in resolved_path.path_in_repo:
                    tree_path = self._parent(path)
                    tree_iter = self._iter_tree(tree_path, revision=resolved_path.revision)
                else:
                    raise
            else:
                tree_iter = itertools.chain([tree_item], tree_iter)
            child_infos = []
            for tree_item in tree_iter:
                child_info = {
                    "name": path_prefix + tree_item["path"],
                    "size": tree_item["size"],
                    "type": tree_item["type"],
                }
                if tree_item["type"] == "file":
                    child_info.update(
                        {
                            "blob_id": tree_item["oid"],
                            "lfs": tree_item.get("lfs"),
                            "last_modified": parse_datetime(tree_item["lastCommit"]["date"]),
                        },
                    )
                child_infos.append(child_info)
            self.dircache[tree_path] = child_infos
        out = self._ls_from_cache(path)
        if not has_revision_in_path:
            out = [{**o, "name": o["name"].replace(revision_in_path, "", 1)} for o in out]
        return out if detail else [o["name"] for o in out]

    def _iter_tree(self, path: str, revision: Optional[str] = None):
        # TODO: use HfApi.list_files_info instead when it supports "lastCommit" and "expand=True"
        # See https://github.com/huggingface/moon-landing/issues/5993
        resolved_path = self.resolve_path(path, revision=revision)
        path = f"{self._api.endpoint}/api/{resolved_path.repo_type}s/{resolved_path.repo_id}/tree/{safe_quote(resolved_path.revision)}/{resolved_path.path_in_repo}".rstrip(
            "/"
        )
        headers = self._api._build_hf_headers()
        yield from paginate(path, params={"expand": True}, headers=headers)

    def cp_file(self, path1: str, path2: str, revision: Optional[str] = None, **kwargs) -> None:
        resolved_path1 = self.resolve_path(path1, revision=revision)
        resolved_path2 = self.resolve_path(path2, revision=revision)

        same_repo = (
            resolved_path1.repo_type == resolved_path2.repo_type and resolved_path1.repo_id == resolved_path2.repo_id
        )

        # TODO: Wait for https://github.com/huggingface/huggingface_hub/issues/1083 to be resolved to simplify this logic
        if same_repo and self.info(path1, revision=resolved_path1.revision)["lfs"] is not None:
            commit_message = f"Copy {path1} to {path2}"
            self._api.create_commit(
                repo_id=resolved_path1.repo_id,
                repo_type=resolved_path1.repo_type,
                revision=resolved_path2.revision,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description", ""),
                operations=[
                    CommitOperationCopy(
                        src_path_in_repo=resolved_path1.path_in_repo,
                        path_in_repo=resolved_path2.path_in_repo,
                        src_revision=resolved_path1.revision,
                    )
                ],
            )
        else:
            with self.open(path1, "rb", revision=resolved_path1.revision) as f:
                content = f.read()
            commit_message = f"Copy {path1} to {path2}"
            self._api.upload_file(
                path_or_fileobj=content,
                path_in_repo=resolved_path2.path_in_repo,
                repo_id=resolved_path2.repo_id,
                token=self.token,
                repo_type=resolved_path2.repo_type,
                revision=resolved_path2.revision,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description"),
            )
        self.invalidate_cache(path=resolved_path1.unresolve())
        self.invalidate_cache(path=resolved_path2.unresolve())

    def modified(self, path: str, **kwargs) -> datetime:
        info = self.info(path, **kwargs)
        if "last_modified" not in info:
            raise IsADirectoryError(path)
        return info["last_modified"]

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        resolved_path = self.resolve_path(path)
        if not resolved_path.path_in_repo:
            revision_in_path = "@" + safe_quote(resolved_path.revision)
            has_revision_in_path = revision_in_path in path
            name = resolved_path.unresolve()
            name = name.replace(revision_in_path, "", 1) if not has_revision_in_path else name
            return {"name": name, "size": 0, "type": "directory"}
        return super().info(path, **kwargs)

    def expand_path(
        self, path: Union[str, List[str]], recursive: bool = False, maxdepth: Optional[int] = None, **kwargs
    ) -> List[str]:
        # The default implementation does not allow passing custom kwargs (e.g., we use these kwargs to propagate the `revision`)
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        if isinstance(path, str):
            return self.expand_path([path], recursive, maxdepth)

        out = set()
        path = [self._strip_protocol(p) for p in path]
        for p in path:
            if has_magic(p):
                bit = set(self.glob(p, **kwargs))
                out |= bit
                if recursive:
                    out |= set(self.expand_path(list(bit), recursive=recursive, maxdepth=maxdepth, **kwargs))
                continue
            elif recursive:
                rec = set(self.find(p, maxdepth=maxdepth, withdirs=True, detail=False, **kwargs))
                out |= rec
            if p not in out and (recursive is False or self.exists(p)):
                # should only check once, for the root
                out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return list(sorted(out))


class HfFileSystemFile(fsspec.spec.AbstractBufferedFile):
    def __init__(self, fs: HfFileSystem, path: str, revision: Optional[str] = None, **kwargs):
        super().__init__(fs, path, **kwargs)
        self.fs: HfFileSystem
        self.resolved_path = fs.resolve_path(path, revision=revision)

    def _fetch_range(self, start: int, end: int) -> bytes:
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        url = (
            f"{self.fs.endpoint}/{REPO_TYPES_URL_PREFIXES.get(self.resolved_path.repo_type, '') + self.resolved_path.repo_id}/resolve/{safe_quote(self.resolved_path.revision)}/{safe_quote(self.resolved_path.path_in_repo)}"
        )
        r = http_backoff("GET", url, headers=headers)
        hf_raise_for_status(r)
        return r.content

    def _initiate_upload(self) -> None:
        self.temp_file = tempfile.NamedTemporaryFile(prefix="hffs-", delete=False)

    def _upload_chunk(self, final: bool = False) -> None:
        self.buffer.seek(0)
        block = self.buffer.read()
        self.temp_file.write(block)
        if final:
            self.temp_file.close()
            self.fs._api.upload_file(
                path_or_fileobj=self.temp_file.name,
                path_in_repo=self.resolved_path.path_in_repo,
                repo_id=self.resolved_path.repo_id,
                token=self.fs.token,
                repo_type=self.resolved_path.repo_type,
                revision=self.resolved_path.revision,
                commit_message=self.kwargs.get("commit_message"),
                commit_description=self.kwargs.get("commit_description"),
            )
            os.remove(self.temp_file.name)
            self.fs.invalidate_cache(
                path=self.resolved_path.unresolve(),
            )


def safe_quote(s: str) -> str:
    return quote(s, safe="")
