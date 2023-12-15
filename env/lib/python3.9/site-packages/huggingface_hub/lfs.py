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
"""Git LFS related type definitions and utilities"""
import io
import os
import re
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Tuple

from requests.auth import HTTPBasicAuth

from huggingface_hub.constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER, REPO_TYPES_URL_PREFIXES
from huggingface_hub.utils import get_session

from .utils import get_token_to_send, hf_raise_for_status, http_backoff, logging, validate_hf_hub_args
from .utils._typing import TypedDict
from .utils.sha import sha256, sha_fileobj


if TYPE_CHECKING:
    from ._commit_api import CommitOperationAdd

logger = logging.get_logger(__name__)

OID_REGEX = re.compile(r"^[0-9a-f]{40}$")

LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"

LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


@dataclass
class UploadInfo:
    """
    Dataclass holding required information to determine whether a blob
    should be uploaded to the hub using the LFS protocol or the regular protocol

    Args:
        sha256 (`bytes`):
            SHA256 hash of the blob
        size (`int`):
            Size in bytes of the blob
        sample (`bytes`):
            First 512 bytes of the blob
    """

    sha256: bytes
    size: int
    sample: bytes

    @classmethod
    def from_path(cls, path: str):
        size = getsize(path)
        with io.open(path, "rb") as file:
            sample = file.peek(512)[:512]
            sha = sha_fileobj(file)
        return cls(size=size, sha256=sha, sample=sample)

    @classmethod
    def from_bytes(cls, data: bytes):
        sha = sha256(data).digest()
        return cls(size=len(data), sample=data[:512], sha256=sha)

    @classmethod
    def from_fileobj(cls, fileobj: BinaryIO):
        sample = fileobj.read(512)
        fileobj.seek(0, io.SEEK_SET)
        sha = sha_fileobj(fileobj)
        size = fileobj.tell()
        fileobj.seek(0, io.SEEK_SET)
        return cls(size=size, sha256=sha, sample=sample)


@validate_hf_hub_args
def post_lfs_batch_info(
    upload_infos: Iterable[UploadInfo],
    token: Optional[str],
    repo_type: str,
    repo_id: str,
    endpoint: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Requests the LFS batch endpoint to retrieve upload instructions

    Learn more: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    Args:
        upload_infos (`Iterable` of `UploadInfo`):
            `UploadInfo` for the files that are being uploaded, typically obtained
            from `CommitOperationAdd.upload_info`
        repo_type (`str`):
            Type of the repo to upload to: `"model"`, `"dataset"` or `"space"`.
        repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        token (`str`, *optional*):
            An authentication token ( See https://huggingface.co/settings/tokens )

    Returns:
        `LfsBatchInfo`: 2-tuple:
            - First element is the list of upload instructions from the server
            - Second element is an list of errors, if any

    Raises:
        `ValueError`: If an argument is invalid or the server response is malformed

        `HTTPError`: If the server returned an error
    """
    endpoint = endpoint if endpoint is not None else ENDPOINT
    url_prefix = ""
    if repo_type in REPO_TYPES_URL_PREFIXES:
        url_prefix = REPO_TYPES_URL_PREFIXES[repo_type]
    batch_url = f"{endpoint}/{url_prefix}{repo_id}.git/info/lfs/objects/batch"
    resp = get_session().post(
        batch_url,
        headers=LFS_HEADERS,
        json={
            "operation": "upload",
            "transfers": ["basic", "multipart"],
            "objects": [
                {
                    "oid": upload.sha256.hex(),
                    "size": upload.size,
                }
                for upload in upload_infos
            ],
            "hash_algo": "sha256",
        },
        auth=HTTPBasicAuth(
            "access_token",
            get_token_to_send(token or True),  # type: ignore  # Token must be provided or retrieved
        ),
    )
    hf_raise_for_status(resp)
    batch_info = resp.json()

    objects = batch_info.get("objects", None)
    if not isinstance(objects, list):
        raise ValueError("Malformed response from server")

    return (
        [_validate_batch_actions(obj) for obj in objects if "error" not in obj],
        [_validate_batch_error(obj) for obj in objects if "error" in obj],
    )


class PayloadPartT(TypedDict):
    partNumber: int
    etag: str


class CompletionPayloadT(TypedDict):
    """Payload that will be sent to the Hub when uploading multi-part."""

    oid: str
    parts: List[PayloadPartT]


def lfs_upload(operation: "CommitOperationAdd", lfs_batch_action: Dict, token: Optional[str]) -> None:
    """
    Handles uploading a given object to the Hub with the LFS protocol.

    Can be a No-op if the content of the file is already present on the hub large file storage.

    Args:
        operation (`CommitOperationAdd`):
            The add operation triggering this upload.
        lfs_batch_action (`dict`):
            Upload instructions from the LFS batch endpoint for this object. See [`~utils.lfs.post_lfs_batch_info`] for
            more details.
        token (`str`, *optional*):
            A [user access token](https://hf.co/settings/tokens) to authenticate requests against the Hub

    Raises:
        - `ValueError` if `lfs_batch_action` is improperly formatted
        - `HTTPError` if the upload resulted in an error
    """
    # 0. If LFS file is already present, skip upload
    _validate_batch_actions(lfs_batch_action)
    actions = lfs_batch_action.get("actions")
    if actions is None:
        # The file was already uploaded
        logger.debug(f"Content of file {operation.path_in_repo} is already present upstream - skipping upload")
        return

    # 1. Validate server response (check required keys in dict)
    upload_action = lfs_batch_action["actions"]["upload"]
    _validate_lfs_action(upload_action)
    verify_action = lfs_batch_action["actions"].get("verify")
    if verify_action is not None:
        _validate_lfs_action(verify_action)

    # 2. Upload file (either single part or multi-part)
    header = upload_action.get("header", {})
    chunk_size = header.get("chunk_size")
    if chunk_size is not None:
        try:
            chunk_size = int(chunk_size)
        except (ValueError, TypeError):
            raise ValueError(
                f"Malformed response from LFS batch endpoint: `chunk_size` should be an integer. Got '{chunk_size}'."
            )
        _upload_multi_part(operation=operation, header=header, chunk_size=chunk_size, upload_url=upload_action["href"])
    else:
        _upload_single_part(operation=operation, upload_url=upload_action["href"])

    # 3. Verify upload went well
    if verify_action is not None:
        _validate_lfs_action(verify_action)
        verify_resp = get_session().post(
            verify_action["href"],
            auth=HTTPBasicAuth(username="USER", password=get_token_to_send(token or True)),  # type: ignore
            json={"oid": operation.upload_info.sha256.hex(), "size": operation.upload_info.size},
        )
        hf_raise_for_status(verify_resp)
    logger.debug(f"{operation.path_in_repo}: Upload successful")


def _validate_lfs_action(lfs_action: dict):
    """validates response from the LFS batch endpoint"""
    if not (
        isinstance(lfs_action.get("href"), str)
        and (lfs_action.get("header") is None or isinstance(lfs_action.get("header"), dict))
    ):
        raise ValueError("lfs_action is improperly formatted")
    return lfs_action


def _validate_batch_actions(lfs_batch_actions: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_actions.get("oid"), str) and isinstance(lfs_batch_actions.get("size"), int)):
        raise ValueError("lfs_batch_actions is improperly formatted")

    upload_action = lfs_batch_actions.get("actions", {}).get("upload")
    verify_action = lfs_batch_actions.get("actions", {}).get("verify")
    if upload_action is not None:
        _validate_lfs_action(upload_action)
    if verify_action is not None:
        _validate_lfs_action(verify_action)
    return lfs_batch_actions


def _validate_batch_error(lfs_batch_error: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_error.get("oid"), str) and isinstance(lfs_batch_error.get("size"), int)):
        raise ValueError("lfs_batch_error is improperly formatted")
    error_info = lfs_batch_error.get("error")
    if not (
        isinstance(error_info, dict)
        and isinstance(error_info.get("message"), str)
        and isinstance(error_info.get("code"), int)
    ):
        raise ValueError("lfs_batch_error is improperly formatted")
    return lfs_batch_error


def _upload_single_part(operation: "CommitOperationAdd", upload_url: str) -> None:
    """
    Uploads `fileobj` as a single PUT HTTP request (basic LFS transfer protocol)

    Args:
        upload_url (`str`):
            The URL to PUT the file to.
        fileobj:
            The file-like object holding the data to upload.

    Returns: `requests.Response`

    Raises: `requests.HTTPError` if the upload resulted in an error
    """
    with operation.as_file(with_tqdm=True) as fileobj:
        response = http_backoff("PUT", upload_url, data=fileobj)
        hf_raise_for_status(response)


def _upload_multi_part(operation: "CommitOperationAdd", header: Dict, chunk_size: int, upload_url: str) -> None:
    """
    Uploads file using HF multipart LFS transfer protocol.
    """
    # 1. Get upload URLs for each part
    sorted_parts_urls = _get_sorted_parts_urls(header=header, upload_info=operation.upload_info, chunk_size=chunk_size)

    # 2. Upload parts (either with hf_transfer or in pure Python)
    use_hf_transfer = HF_HUB_ENABLE_HF_TRANSFER
    if (
        HF_HUB_ENABLE_HF_TRANSFER
        and not isinstance(operation.path_or_fileobj, str)
        and not isinstance(operation.path_or_fileobj, Path)
    ):
        warnings.warn(
            "hf_transfer is enabled but does not support uploading from bytes or BinaryIO, falling back to regular"
            " upload"
        )
        use_hf_transfer = False

    response_headers = (
        _upload_parts_hf_transfer(operation=operation, sorted_parts_urls=sorted_parts_urls, chunk_size=chunk_size)
        if use_hf_transfer
        else _upload_parts_iteratively(operation=operation, sorted_parts_urls=sorted_parts_urls, chunk_size=chunk_size)
    )

    # 3. Send completion request
    completion_res = get_session().post(
        upload_url,
        json=_get_completion_payload(response_headers, operation.upload_info.sha256.hex()),
        headers=LFS_HEADERS,
    )
    hf_raise_for_status(completion_res)


def _get_sorted_parts_urls(header: Dict, upload_info: UploadInfo, chunk_size: int) -> List[str]:
    sorted_part_upload_urls = [
        upload_url
        for _, upload_url in sorted(
            [
                (int(part_num, 10), upload_url)
                for part_num, upload_url in header.items()
                if part_num.isdigit() and len(part_num) > 0
            ],
            key=lambda t: t[0],
        )
    ]
    num_parts = len(sorted_part_upload_urls)
    if num_parts != ceil(upload_info.size / chunk_size):
        raise ValueError("Invalid server response to upload large LFS file")
    return sorted_part_upload_urls


def _get_completion_payload(response_headers: List[Dict], oid: str) -> CompletionPayloadT:
    parts: List[PayloadPartT] = []
    for part_number, header in enumerate(response_headers):
        etag = header.get("etag")
        if etag is None or etag == "":
            raise ValueError(f"Invalid etag (`{etag}`) returned for part {part_number + 1}")
        parts.append(
            {
                "partNumber": part_number + 1,
                "etag": etag,
            }
        )
    return {"oid": oid, "parts": parts}


def _upload_parts_iteratively(
    operation: "CommitOperationAdd", sorted_parts_urls: List[str], chunk_size: int
) -> List[Dict]:
    headers = []
    with operation.as_file(with_tqdm=True) as fileobj:
        for part_idx, part_upload_url in enumerate(sorted_parts_urls):
            with SliceFileObj(
                fileobj,
                seek_from=chunk_size * part_idx,
                read_limit=chunk_size,
            ) as fileobj_slice:
                part_upload_res = http_backoff("PUT", part_upload_url, data=fileobj_slice)
                hf_raise_for_status(part_upload_res)
                headers.append(part_upload_res.headers)
    return headers  # type: ignore


def _upload_parts_hf_transfer(
    operation: "CommitOperationAdd", sorted_parts_urls: List[str], chunk_size: int
) -> List[Dict]:
    # Upload file using an external Rust-based package. Upload is faster but support less features (no progress bars).
    try:
        from hf_transfer import multipart_upload
    except ImportError:
        raise ValueError(
            "Fast uploading using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is"
            " not available in your environment. Try `pip install hf_transfer`."
        )

    try:
        return multipart_upload(
            file_path=operation.path_or_fileobj,
            parts_urls=sorted_parts_urls,
            chunk_size=chunk_size,
            max_files=128,
            parallel_failures=127,  # could be removed
            max_retries=5,
        )
    except Exception as e:
        raise RuntimeError(
            "An error occurred while uploading using `hf_transfer`. Consider disabling HF_HUB_ENABLE_HF_TRANSFER for"
            " better error handling."
        ) from e


class SliceFileObj(AbstractContextManager):
    """
    Utility context manager to read a *slice* of a seekable file-like object as a seekable, file-like object.

    This is NOT thread safe

    Inspired by stackoverflow.com/a/29838711/593036

    Credits to @julien-c

    Args:
        fileobj (`BinaryIO`):
            A file-like object to slice. MUST implement `tell()` and `seek()` (and `read()` of course).
            `fileobj` will be reset to its original position when exiting the context manager.
        seek_from (`int`):
            The start of the slice (offset from position 0 in bytes).
        read_limit (`int`):
            The maximum number of bytes to read from the slice.

    Attributes:
        previous_position (`int`):
            The previous position

    Examples:

    Reading 200 bytes with an offset of 128 bytes from a file (ie bytes 128 to 327):
    ```python
    >>> with open("path/to/file", "rb") as file:
    ...     with SliceFileObj(file, seek_from=128, read_limit=200) as fslice:
    ...         fslice.read(...)
    ```

    Reading a file in chunks of 512 bytes
    ```python
    >>> import os
    >>> chunk_size = 512
    >>> file_size = os.getsize("path/to/file")
    >>> with open("path/to/file", "rb") as file:
    ...     for chunk_idx in range(ceil(file_size / chunk_size)):
    ...         with SliceFileObj(file, seek_from=chunk_idx * chunk_size, read_limit=chunk_size) as fslice:
    ...             chunk = fslice.read(...)

    ```
    """

    def __init__(self, fileobj: BinaryIO, seek_from: int, read_limit: int):
        self.fileobj = fileobj
        self.seek_from = seek_from
        self.read_limit = read_limit

    def __enter__(self):
        self._previous_position = self.fileobj.tell()
        end_of_stream = self.fileobj.seek(0, os.SEEK_END)
        self._len = min(self.read_limit, end_of_stream - self.seek_from)
        # ^^ The actual number of bytes that can be read from the slice
        self.fileobj.seek(self.seek_from, io.SEEK_SET)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fileobj.seek(self._previous_position, io.SEEK_SET)

    def read(self, n: int = -1):
        pos = self.tell()
        if pos >= self._len:
            return b""
        remaining_amount = self._len - pos
        data = self.fileobj.read(remaining_amount if n < 0 else min(n, remaining_amount))
        return data

    def tell(self) -> int:
        return self.fileobj.tell() - self.seek_from

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        start = self.seek_from
        end = start + self._len
        if whence in (os.SEEK_SET, os.SEEK_END):
            offset = start + offset if whence == os.SEEK_SET else end + offset
            offset = max(start, min(offset, end))
            whence = os.SEEK_SET
        elif whence == os.SEEK_CUR:
            cur_pos = self.fileobj.tell()
            offset = max(start - cur_pos, min(offset, end - cur_pos))
        else:
            raise ValueError(f"whence value {whence} is not supported")
        return self.fileobj.seek(offset, whence) - self.seek_from

    def __iter__(self):
        yield self.read(n=4 * 1024 * 1024)
