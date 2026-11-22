from typing import Final
from urllib.parse import urlparse


def gemini_file_id(file_id: str) -> str:
    path: Final = urlparse(file_id).path.lstrip("/") if file_id.startswith(("http://", "https://")) else file_id
    files_index: Final = path.find("files/")
    return (path[files_index:] if files_index != -1 else path).strip("/").removeprefix("files/")
