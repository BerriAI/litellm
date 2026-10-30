from __future__ import annotations

import threading
from itertools import chain
from pathlib import Path
from typing import Final

from client_apps.codex.columns import column_lock, column_models, load_columns
from client_apps.compat_collector import CLIENT_APPS_DIR, client_manifest


def _acquire_in_thread(provider: str, lock_dir: Path) -> threading.Event:
    acquired: Final = threading.Event()

    def hold() -> None:
        with column_lock(provider, lock_dir):
            acquired.set()

    threading.Thread(target=hold, daemon=True).start()
    return acquired


def test_column_lock_lets_one_cell_per_column_run_at_a_time(tmp_path: Path) -> None:
    with column_lock("azure", tmp_path):
        same_column: Final = _acquire_in_thread("azure", tmp_path)
        other_column: Final = _acquire_in_thread("openai", tmp_path)

        assert other_column.wait(5)
        assert not same_column.wait(0.3)

    assert same_column.wait(5)


def test_every_manifest_column_has_deployments_with_unique_names() -> None:
    manifest: Final = client_manifest(CLIENT_APPS_DIR / "codex")
    names: Final = tuple(chain.from_iterable(column_models(provider) for provider in manifest.providers))

    assert set(load_columns()) == set(manifest.providers)
    assert all(column_models(provider) for provider in manifest.providers)
    assert len(set(names)) == len(names)
