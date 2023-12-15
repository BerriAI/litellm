"""Utilities to efficiently compute the SHA 256 hash of a bunch of bytes."""
from hashlib import sha256
from typing import BinaryIO, Optional


def sha_fileobj(fileobj: BinaryIO, chunk_size: Optional[int] = None) -> bytes:
    """
    Computes the sha256 hash of the given file object, by chunks of size `chunk_size`.

    Args:
        fileobj (file-like object):
            The File object to compute sha256 for, typically obtained with `open(path, "rb")`
        chunk_size (`int`, *optional*):
            The number of bytes to read from `fileobj` at once, defaults to 1MB.

    Returns:
        `bytes`: `fileobj`'s sha256 hash as bytes
    """
    chunk_size = chunk_size if chunk_size is not None else 1024 * 1024

    sha = sha256()
    while True:
        chunk = fileobj.read(chunk_size)
        sha.update(chunk)
        if not chunk:
            break
    return sha.digest()
