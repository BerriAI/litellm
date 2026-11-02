"""FIPS repository for the LiteLLM_Config row that marks a database as set up in FIPS mode."""

from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # Prisma delegates query_raw and execute_raw dynamically

from litellm.proxy.db.create_views import SupportsRawQueries
from litellm.proxy.db.exception_handler import call_with_db_reconnect_retry

FIPS_MARKER: Final = "fips_mode"
_IS_MARKED: Final = 'SELECT 1 FROM "LiteLLM_Config" WHERE param_name = $1'
_OTHER_TABLES: Final = (
    "SELECT table_name FROM information_schema.tables WHERE table_schema = ANY (current_schemas(false)) "
    "AND table_type = 'BASE TABLE' AND table_name LIKE 'LiteLLM\\_%' AND table_name <> 'LiteLLM_Config'"
)
_MARK: Final = (
    "INSERT INTO \"LiteLLM_Config\" (param_name, param_value) VALUES ($1, 'true'::jsonb) "
    "ON CONFLICT (param_name) DO NOTHING"
)


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class _WithWriter(Protocol):
    @property
    def writer_db(self) -> object: ...

    async def attempt_db_reconnect(
        self, *, reason: str, timeout_seconds: float | None = None, lock_timeout_seconds: float | None = None
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class FipsDatabaseState:
    marked_fips: bool
    holds_rows: bool


class FipsRepository:
    """Reads and writes go to the writer, since a replica may not have the marker yet. A read that loses its
    connection reconnects and runs once more."""

    def __init__(self, prisma_client: _WithWriter) -> None:
        self._prisma_client: Final = prisma_client

    @property
    def _writer(self) -> SupportsRawQueries:
        writer: Final = self._prisma_client.writer_db
        return cast(SupportsRawQueries, writer)  # cast-ok: Prisma delegates database methods dynamically

    async def is_marked_fips(self) -> bool:
        return await call_with_db_reconnect_retry(
            self._prisma_client, self._read_marker, reason="fips_startup_read_marker_failure"
        )

    async def state(self) -> FipsDatabaseState:
        """The marker and whether any other LiteLLM row exists, read in one statement so they agree."""
        return await call_with_db_reconnect_retry(
            self._prisma_client, self._read_state, reason="fips_startup_read_state_failure"
        )

    async def _read_marker(self) -> bool:
        return bool(await self._writer.query_raw(_IS_MARKED, FIPS_MARKER))

    async def _read_state(self) -> FipsDatabaseState:
        writer: Final = self._writer
        tables: Final = tuple(str(row["table_name"]) for row in await writer.query_raw(_OTHER_TABLES))
        any_row_in_other_tables: Final = "".join(f" OR EXISTS (SELECT 1 FROM {_quoted(table)})" for table in tables)
        row: Final = (
            await writer.query_raw(
                'SELECT EXISTS (SELECT 1 FROM "LiteLLM_Config" WHERE param_name = $1) AS marked_fips, '
                f'(EXISTS (SELECT 1 FROM "LiteLLM_Config" WHERE param_name <> $1){any_row_in_other_tables}) '
                "AS holds_rows",
                FIPS_MARKER,
            )
        )[0]
        return FipsDatabaseState(marked_fips=row["marked_fips"] is True, holds_rows=row["holds_rows"] is True)

    async def mark_fips(self) -> None:
        await self._writer.execute_raw(_MARK, FIPS_MARKER)
