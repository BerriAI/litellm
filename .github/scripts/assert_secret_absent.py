import base64
import io
import os
import sys
import urllib.parse
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, TextIO

SECRET_VARIABLE: Final = "SCANNED_SECRET"
MIN_ENCODED_LENGTH: Final = 12
MAX_ARCHIVE_DEPTH: Final = 3
MASK: Final = "***"

Forms = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Hit:
    location: str
    form: str


def _aligned_encodings(raw: bytes, name: str, encode: Callable[[bytes], bytes]) -> Iterator[tuple[str, str]]:
    for offset in range(3):
        chunk: Final = raw[offset:]
        yield f"{name} offset {offset}", encode(chunk[: len(chunk) - len(chunk) % 3]).decode()


def secret_forms(secret: str) -> Forms:
    encoded: Final = (
        ("url-encoded", urllib.parse.quote(secret, safe="")),
        *_aligned_encodings(secret.encode(), "base64", base64.b64encode),
        *_aligned_encodings(secret.encode(), "base64url", base64.urlsafe_b64encode),
    )
    by_value: Final = MappingProxyType(
        {value: name for name, value in encoded if value != secret and len(value) >= MIN_ENCODED_LENGTH}
    )
    return (("raw", secret), *((name, value) for value, name in by_value.items()))


def mask(text: str, forms: Forms) -> str:
    return reduce(lambda masked, form: masked.replace(form[1], MASK), forms, text)


def _name_hits(location: str, name: str, forms: Forms) -> Iterator[Hit]:
    for form, value in forms:
        if value in name:
            yield Hit(mask(location, forms), f"{form} in name")


def _content_hits(location: str, data: bytes, forms: Forms, depth: int) -> Iterator[Hit]:
    for form, value in forms:
        if value.encode() in data:
            yield Hit(mask(location, forms), form)
    if depth >= MAX_ARCHIVE_DEPTH or not zipfile.is_zipfile(io.BytesIO(data)):
        return
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            yield from _content_hits(f"{location}!{member.filename}", archive.read(member), forms, depth + 1)


def _walk(root: Path) -> Iterator[tuple[Path, bool]]:
    for directory, subdirectories, files in os.walk(root):
        yield from ((Path(directory, name), True) for name in subdirectories if Path(directory, name).is_symlink())
        yield from ((Path(directory, name), False) for name in files)


def _path_hits(root: Path, path: Path, linked_directory: bool, forms: Forms) -> Iterator[Hit]:
    location: Final = str(path.relative_to(root))
    if linked_directory:
        yield Hit(mask(location, forms), "directory symlink, not scanned")
        return
    yield from _name_hits(location, location, forms)
    yield from _content_hits(location, path.read_bytes(), forms, 0)


def scan(root: Path, secret: str) -> tuple[int, tuple[Hit, ...]]:
    forms: Final = secret_forms(secret)
    entries: Final = tuple(_walk(root))
    hits: Final = tuple(chain.from_iterable(_path_hits(root, path, linked, forms) for path, linked in entries))
    return len(entries), hits


def main(argv: tuple[str, ...], environ: Mapping[str, str], out: TextIO) -> int:
    if len(argv) != 1:
        print(f"usage: assert_secret_absent.py <directory>, with the secret in ${SECRET_VARIABLE}", file=out)
        return 2
    secret: Final = environ.get(SECRET_VARIABLE, "")
    root: Final = Path(argv[0])
    if not secret:
        print(f"{SECRET_VARIABLE} is empty, nothing to scan for", file=out)
        return 0
    if not root.is_dir():
        print(f"{root} is not a directory, nothing to scan", file=out)
        return 0
    try:
        scanned, hits = scan(root, secret)
    except Exception as error:
        print(f"scan failed with {type(error).__name__}, treating the results as unsafe", file=out)
        return 1
    for hit in hits:
        print(f"secret found: {hit.location} ({hit.form})", file=out)
    if hits:
        print(f"{len(hits)} copies of the secret under {root}, refusing to upload", file=out)
        return 1
    print(f"no copy of the secret in {scanned} files under {root}", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:]), os.environ, sys.stdout))
