from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.workers import worker_services


@pytest.fixture(scope="session", autouse=True)
def isolated_security_worker(worker_id: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    if worker_id == "master":
        yield
        return
    directory: Final = tmp_path_factory.mktemp("security-worker")
    results: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory))) / worker_id
    results.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("INTEGRATION_RESULTS_DIR", str(results))
        with worker_services(directory) as services:
            for name, value in services.items():
                environment.setenv(name, value)
            yield
