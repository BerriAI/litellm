from collections.abc import Iterator
from typing import cast

import pytest
from pydantic import BaseModel

from litellm.repositories import rust_database
from litellm.rust_bridge._native import NativeDatabase


class _Repository:
    def __init__(self, database: NativeDatabase) -> None:
        self.database = database


def _database() -> NativeDatabase:
    return cast(NativeDatabase, object())


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    yield
    database = rust_database.current()
    if database is not None:
        rust_database.clear(database)


def test_repository_is_none_without_a_database() -> None:
    assert rust_database.current() is None
    assert rust_database.repository(_Repository) is None


def test_repository_is_built_once_per_installed_database() -> None:
    database = _database()
    rust_database.install(database)

    first = rust_database.repository(_Repository)

    assert rust_database.current() is database
    assert first is not None and first.database is database
    assert rust_database.repository(_Repository) is first


def test_install_drops_repositories_of_the_previous_database() -> None:
    old = _database()
    rust_database.install(old)
    old_repository = rust_database.repository(_Repository)
    new = _database()

    rust_database.install(new)
    new_repository = rust_database.repository(_Repository)

    assert new_repository is not old_repository
    assert new_repository is not None and new_repository.database is new


def test_clear_forgets_the_database_and_its_repositories() -> None:
    database = _database()
    rust_database.install(database)
    rust_database.repository(_Repository)

    rust_database.clear(database)

    assert rust_database.current() is None
    assert rust_database.repository(_Repository) is None


def test_clear_keeps_a_newer_database() -> None:
    old = _database()
    new = _database()
    rust_database.install(old)
    rust_database.install(new)
    kept = rust_database.repository(_Repository)

    rust_database.clear(old)

    assert rust_database.current() is new
    assert rust_database.repository(_Repository) is kept


def test_required_repository_raises_no_db_connected_without_a_database() -> None:
    with pytest.raises(Exception, match="No db connected"):
        rust_database.required_repository(_Repository)


def test_required_repository_returns_the_installed_repository() -> None:
    database = _database()
    rust_database.install(database)

    assert rust_database.required_repository(_Repository) is rust_database.repository(_Repository)


class _Row(BaseModel):
    token: str


def test_decode_turns_a_json_answer_into_the_model() -> None:
    assert rust_database.decode(_Row, '{"token": "hashed"}') == _Row(token="hashed")


def test_decode_answers_none_for_a_missing_row() -> None:
    assert rust_database.decode(_Row, None) is None
