from __future__ import annotations

import os
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg

DATABASE_URL: Final = os.environ["DATABASE_URL"]


def postgres_url() -> str:
    parsed: Final = urlsplit(DATABASE_URL)
    return urlunsplit(parsed._replace(path="/postgres"))


def main() -> None:
    with psycopg.connect(postgres_url(), autocommit=True) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        admin.execute("CREATE ROLE litellm_writer LOGIN PASSWORD 'litellm-writer' NOSUPERUSER")
        admin.execute("CREATE ROLE litellm_reader LOGIN PASSWORD 'litellm-reader' NOSUPERUSER NOINHERIT")
        admin.execute("ALTER ROLE litellm_reader SET default_transaction_read_only = on")
        admin.execute("ALTER DATABASE circle_test OWNER TO litellm_writer")
        admin.execute("GRANT CONNECT ON DATABASE circle_test TO litellm_reader")
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute("GRANT USAGE ON SCHEMA public TO litellm_reader")
        admin.execute(
            "ALTER DEFAULT PRIVILEGES FOR ROLE litellm_writer IN SCHEMA public GRANT SELECT ON TABLES TO litellm_reader"
        )
        admin.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO litellm_reader")

    parsed: Final = urlsplit(DATABASE_URL)
    reader_url: Final = urlunsplit(
        parsed._replace(netloc=f"litellm_reader:litellm-reader@{parsed.hostname}:{parsed.port}")
    )
    writer_url: Final = urlunsplit(
        parsed._replace(netloc=f"litellm_writer:litellm-writer@{parsed.hostname}:{parsed.port}")
    )
    with psycopg.connect(reader_url, autocommit=True) as reader:
        assert reader.execute("SHOW transaction_read_only").fetchone() == ("on",)
        try:
            reader.execute("CREATE TABLE integration_readonly_probe (id int)")
        except psycopg.errors.ReadOnlySqlTransaction:
            pass
        else:
            raise AssertionError("litellm_reader executed a write statement")
    with psycopg.connect(writer_url, autocommit=True) as writer:
        assert writer.execute("SELECT current_user").fetchone() == ("litellm_writer",)


if __name__ == "__main__":
    main()
