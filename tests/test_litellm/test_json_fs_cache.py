from pathlib import Path
from typing import Final

from tests.test_litellm._json_fs_cache import JsonFileCache


def test_json_file_cache_is_content_addressed_and_recursive(tmp_path: Path) -> None:
    key: Final = {"method": "POST", "body": {"model": "test-model", "pages": [0]}}
    reordered_key: Final = {"body": {"pages": [0], "model": "test-model"}, "method": "POST"}
    value: Final = {"request": key, "response": {"status_code": 200}}
    cache: Final = JsonFileCache(tmp_path / "provider")

    stored_path: Final = cache.put(key, value)

    assert stored_path.name == cache.path_for(reordered_key).name
    assert cache.get(reordered_key) == value
    assert JsonFileCache(tmp_path).values() == (value,)
