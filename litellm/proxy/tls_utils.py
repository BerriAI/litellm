import os
import tempfile
from collections.abc import Mapping

PEM_FILE_MODE = 0o600


def materialize_pem_files(files: Mapping[str, str], *, directory_prefix: str) -> dict[str, str]:
    directory = tempfile.mkdtemp(prefix=directory_prefix)
    paths: dict[str, str] = {}
    for filename, pem in files.items():
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(pem if pem.endswith("\n") else f"{pem}\n")
        os.chmod(path, PEM_FILE_MODE)
        paths[filename] = path
    return paths
