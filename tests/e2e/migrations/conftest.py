import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from _pytest.fixtures import SubRequest

from .containers import Containers, docker, ready
from .database import Database, Databases


@pytest.fixture(scope="session")
def migration_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    configured: Final = os.environ.get("LITELLM_MIGRATION_TEST_IMAGE")
    assert configured, "LITELLM_MIGRATION_TEST_IMAGE must name the built candidate image"
    image: Final = docker("image", "inspect", configured, "--format", "{{.Id}}")
    assert image.startswith("sha256:"), "Unable to identify the candidate image"
    output: Final = Path(os.environ.get("MIGRATION_TEST_OUTPUT", str(tmp_path_factory.getbasetemp())))
    output.mkdir(parents=True, exist_ok=True)
    (output / "image.json").write_text(json.dumps({"requested": configured, "image_id": image}))
    return image


@pytest.fixture(scope="session")
def databases() -> Databases:
    admin: Final = os.environ.get("MIGRATION_TEST_ADMIN_URL", "")
    parsed: Final = urlsplit(admin)
    assert parsed.hostname in ("127.0.0.1", "localhost"), "Use an isolated loopback PostgreSQL test cluster"
    assert parsed.port and parsed.path and not parsed.query, "Supply the test cluster port and admin database"
    container_admin: Final = os.environ.get(
        "MIGRATION_TEST_CONTAINER_ADMIN_URL",
        admin.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal"),
    )
    return Databases(admin, container_admin)


@pytest.fixture(scope="session")
def migrated_template(
    databases: Databases, migration_image: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Database]:
    output: Final = Path(os.environ.get("MIGRATION_TEST_OUTPUT", str(tmp_path_factory.getbasetemp()))) / "seed"
    with databases.create() as database:
        with Containers(migration_image, output).start(database) as replica:
            ready((replica,), database)
        yield database


@pytest.fixture
def database(databases: Databases, migrated_template: Database) -> Iterator[Database]:
    with databases.create(migrated_template) as database:
        yield database


@pytest.fixture
def containers(migration_image: str, tmp_path: Path, request: SubRequest) -> Containers:
    configured: Final = os.environ.get("MIGRATION_TEST_OUTPUT")
    output: Final = Path(configured) / request.node.name if configured else tmp_path
    output.mkdir(parents=True, exist_ok=True)
    return Containers(migration_image, output)
