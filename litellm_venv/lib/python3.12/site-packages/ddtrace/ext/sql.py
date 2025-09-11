from typing import Dict  # noqa:F401

from ddtrace.internal.logger import get_logger
from ddtrace.internal.module import ModuleWatchdog


log = get_logger(__name__)

# tags
DB = "sql.db"  # the name of the database


def normalize_vendor(vendor):
    # type: (str) -> str
    """Return a canonical name for a type of database."""
    if not vendor:
        return "db"  # should this ever happen?
    elif "sqlite" in vendor:
        return "sqlite"
    elif "postgres" in vendor or vendor == "psycopg2":
        return "postgres"
    else:
        return vendor


def _dd_parse_pg_dsn(dsn):
    # type: (str) -> Dict[str, str]
    """
    Return a dictionary of the components of a postgres DSN.
    >>> parse_pg_dsn('user=dog port=1543 dbname=dogdata')
    {'user':'dog', 'port':'1543', 'dbname':'dogdata'}
    """
    dsn_dict = dict()
    try:
        # Provides a default implementation for parsing DSN strings.
        # The following is an example of a valid DSN string that fails to be parsed:
        # "db=moon user=ears options='-c statement_timeout=1000 -c lock_timeout=250'"
        dsn_dict = dict(_.split("=", 1) for _ in dsn.split())
    except Exception:
        log.debug("Failed to parse postgres dsn connection", exc_info=True)
    return dsn_dict


# Do not import from psycopg directly! This reference will be updated at runtime to use
# a better implementation that is provided by the psycopg library.
# This is done to avoid circular imports.
parse_pg_dsn = _dd_parse_pg_dsn


@ModuleWatchdog.after_module_imported("psycopg2")
def use_psycopg2_parse_dsn(psycopg_module):
    """Replaces parse_pg_dsn with the helper function defined in psycopg2"""
    global parse_pg_dsn

    try:
        from psycopg2.extensions import parse_dsn

        parse_pg_dsn = parse_dsn
    except ImportError:
        # Best effort, we'll use our own parser: _dd_parse_pg_dsn
        pass


@ModuleWatchdog.after_module_imported("psycopg")
def use_psycopg3_parse_dsn(psycopg_module):
    """Replaces parse_pg_dsn with the helper function defined in psycopg3"""
    global parse_pg_dsn

    try:
        from psycopg.conninfo import conninfo_to_dict

        parse_pg_dsn = conninfo_to_dict
    except ImportError:
        # Best effort, we'll use our own parser: _dd_parse_pg_dsn
        pass
