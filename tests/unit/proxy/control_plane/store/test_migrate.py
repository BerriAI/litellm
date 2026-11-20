from typing import Final

from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import SecretStr

from litellm.proxy.control_plane.store.migrate import database_url, head_revision, migrate, migration_config


def test_migrator_dispatches_upgrade_head_using_packaged_revisions_and_the_dedicated_url() -> None:
    calls: Final[list[tuple[Config, str]]] = []  # mutable-ok: records the Alembic command boundary

    def upgrade(config: Config, revision: str) -> None:
        calls.append((config, revision))

    url: Final = SecretStr("postgresql://dedicated/control")
    migrate(url, upgrade=upgrade)
    [(config, revision)] = calls
    assert revision == "head"
    assert config.attributes["database_url"] == url
    script: Final = ScriptDirectory.from_config(config)
    assert script.get_current_head() == head_revision()
    assert tuple(rev.revision for rev in script.walk_revisions()) == (head_revision(),)


def test_plain_postgres_urls_use_psycopg_without_modifying_escaped_credentials() -> None:
    assert (
        database_url(SecretStr("postgresql://user:p%25ss@local/control"))
        == "postgresql+psycopg://user:p%25ss@local/control"
    )
    assert database_url(SecretStr("postgresql+psycopg://local/control")) == "postgresql+psycopg://local/control"
