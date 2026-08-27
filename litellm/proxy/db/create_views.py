from typing import Any, Final, Protocol

from litellm import verbose_logger

_db = Any


class SupportsExecuteRaw(Protocol):
    """The one database operation create_view_tolerating_race needs.

    Narrower than the `_db = Any` the rest of this module still uses, so the
    helper's contract is checkable at its call sites without retyping every
    function here.
    """

    async def execute_raw(self, query: str, *args: object) -> int: ...


# Markers that indicate a view/relation does not yet exist in the database.
# Keeping these in one place avoids repeating the check across all view blocks
# and prevents overly broad matches (e.g. bare 'undefined' would also match
# 'undefined function' or 'column undefined_col referenced in query').
_VIEW_NOT_FOUND_MARKERS: Final = ("does not exist", "no such table", "undefined table")

# Markers for the inverse condition: another replica created the view between
# our existence probe and our CREATE.
_VIEW_ALREADY_EXISTS_MARKERS: Final = ("already exists", "duplicate object", "duplicate table")


async def create_view_tolerating_race(db: SupportsExecuteRaw, view_name: str, ddl: str) -> None:
    """
    Create a view, treating "a concurrent creator won" as success.

    Every replica booting against the same fresh database observes the view as
    absent and issues the CREATE; Postgres fails all but one with a
    duplicate-object error. The desired end state is still reached, so losing
    that race is success. Without this, the loser's exception propagates out of
    a detached startup task and the remaining views are never created.
    """
    try:
        await db.execute_raw(ddl)
        verbose_logger.debug("%s Created!", view_name)
    except Exception as e:
        if not any(marker in str(e).lower() for marker in _VIEW_ALREADY_EXISTS_MARKERS):
            raise
        verbose_logger.debug("%s already created by a concurrent replica", view_name)


async def create_missing_views(db: _db):
    """
    --------------------------------------------------
    NOTE: Copy of `litellm/db_scripts/create_views.py`.
    --------------------------------------------------
    Checks if the LiteLLM_VerificationTokenView and MonthlyGlobalSpend exists in the user's db.

    LiteLLM_VerificationTokenView: This view is used for getting the token + team data in user_api_key_auth

    MonthlyGlobalSpend: This view is used for the admin view to see global spend for this month

    If the view doesn't exist, one will be created.
    """

    try:
        # Try to select one row from the view
        await db.query_raw("""SELECT 1 FROM "LiteLLM_VerificationTokenView" LIMIT 1""")
        verbose_logger.debug("LiteLLM_VerificationTokenView Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        # If an error occurs, the view does not exist, so create it
        await create_view_tolerating_race(
            db,
            "LiteLLM_VerificationTokenView",
            """
                CREATE VIEW "LiteLLM_VerificationTokenView" AS
                SELECT
                v.*,
                t.spend AS team_spend,
                t.max_budget AS team_max_budget,
                t.tpm_limit AS team_tpm_limit,
                t.rpm_limit AS team_rpm_limit,
                p.project_alias AS project_alias
                FROM "LiteLLM_VerificationToken" v
                LEFT JOIN "LiteLLM_TeamTable" t ON v.team_id = t.team_id
                LEFT JOIN "LiteLLM_ProjectTable" p ON v.project_id = p.project_id;
            """,
        )

    try:
        await db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpend" LIMIT 1""")
        verbose_logger.debug("MonthlyGlobalSpend Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
        CREATE OR REPLACE VIEW "MonthlyGlobalSpend" AS 
        SELECT
        DATE("startTime") AS date, 
        SUM("spend") AS spend 
        FROM 
        "LiteLLM_SpendLogs" 
        WHERE 
        "startTime" >= (CURRENT_DATE - INTERVAL '30 days')
        GROUP BY 
        DATE("startTime");
        """
        await create_view_tolerating_race(db, "MonthlyGlobalSpend", sql_query)

    try:
        await db.query_raw("""SELECT 1 FROM "Last30dKeysBySpend" LIMIT 1""")
        verbose_logger.debug("Last30dKeysBySpend Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
        CREATE OR REPLACE VIEW "Last30dKeysBySpend" AS
        SELECT 
        L."api_key", 
        V."key_alias",
        V."key_name",
        SUM(L."spend") AS total_spend
        FROM
        "LiteLLM_SpendLogs" L
        LEFT JOIN 
        "LiteLLM_VerificationToken" V
        ON
        L."api_key" = V."token"
        WHERE
        L."startTime" >= (CURRENT_DATE - INTERVAL '30 days')
        GROUP BY
        L."api_key", V."key_alias", V."key_name"
        ORDER BY
        total_spend DESC;
        """
        await create_view_tolerating_race(db, "Last30dKeysBySpend", sql_query)

    try:
        await db.query_raw("""SELECT 1 FROM "Last30dModelsBySpend" LIMIT 1""")
        verbose_logger.debug("Last30dModelsBySpend Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
        CREATE OR REPLACE VIEW "Last30dModelsBySpend" AS
        SELECT
        "model",
        SUM("spend") AS total_spend
        FROM
        "LiteLLM_SpendLogs"
        WHERE
        "startTime" >= (CURRENT_DATE - INTERVAL '30 days')
        AND "model" != ''
        GROUP BY
        "model"
        ORDER BY
        total_spend DESC;
        """
        await create_view_tolerating_race(db, "Last30dModelsBySpend", sql_query)
    try:
        await db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpendPerKey" LIMIT 1""")
        verbose_logger.debug("MonthlyGlobalSpendPerKey Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
            CREATE OR REPLACE VIEW "MonthlyGlobalSpendPerKey" AS 
            SELECT
            DATE("startTime") AS date, 
            SUM("spend") AS spend,
            api_key as api_key
            FROM 
            "LiteLLM_SpendLogs" 
            WHERE 
            "startTime" >= (CURRENT_DATE - INTERVAL '30 days')
            GROUP BY 
            DATE("startTime"),
            api_key;
        """
        await create_view_tolerating_race(db, "MonthlyGlobalSpendPerKey", sql_query)
    try:
        await db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpendPerUserPerKey" LIMIT 1""")
        verbose_logger.debug("MonthlyGlobalSpendPerUserPerKey Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
            CREATE OR REPLACE VIEW "MonthlyGlobalSpendPerUserPerKey" AS 
            SELECT
            DATE("startTime") AS date, 
            SUM("spend") AS spend,
            api_key as api_key,
            "user" as "user"
            FROM 
            "LiteLLM_SpendLogs" 
            WHERE 
            "startTime" >= (CURRENT_DATE - INTERVAL '30 days')
            GROUP BY 
            DATE("startTime"),
            "user",
            api_key;
        """
        await create_view_tolerating_race(db, "MonthlyGlobalSpendPerUserPerKey", sql_query)

    try:
        await db.query_raw("""SELECT 1 FROM "DailyTagSpend" LIMIT 1""")
        verbose_logger.debug("DailyTagSpend Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
        CREATE OR REPLACE VIEW "DailyTagSpend" AS
        SELECT
            jsonb_array_elements_text(request_tags) AS individual_request_tag,
            DATE(s."startTime") AS spend_date,
            COUNT(*) AS log_count,
            SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs" s
        GROUP BY individual_request_tag, DATE(s."startTime");
        """
        await create_view_tolerating_race(db, "DailyTagSpend", sql_query)

    try:
        await db.query_raw("""SELECT 1 FROM "Last30dTopEndUsersSpend" LIMIT 1""")
        verbose_logger.debug("Last30dTopEndUsersSpend Exists!")
    except Exception as e:
        error_msg = str(e).lower()
        if not any(marker in error_msg for marker in _VIEW_NOT_FOUND_MARKERS):
            raise
        sql_query = """
        CREATE VIEW "Last30dTopEndUsersSpend" AS
        SELECT end_user, COUNT(*) AS total_events, SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        WHERE end_user <> '' AND end_user <> user
        AND "startTime" >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY end_user
        ORDER BY total_spend DESC
        LIMIT 100;
        """
        await create_view_tolerating_race(db, "Last30dTopEndUsersSpend", sql_query)


async def should_create_missing_views(db: _db) -> bool:
    """
    Run only on first time startup.

    If SpendLogs table already has values, then don't create views on startup.
    """

    sql_query: Final = """
    SELECT reltuples::BIGINT
    FROM pg_class
    WHERE oid = '"LiteLLM_SpendLogs"'::regclass;
    """

    result: Final = await db.query_raw(query=sql_query)

    verbose_logger.debug("Estimated Row count of LiteLLM_SpendLogs = %s", result)
    if (
        result
        and isinstance(result, list)
        and len(result) > 0
        and isinstance(result[0], dict)
        and "reltuples" in result[0]
        and result[0]["reltuples"] is not None
        and (result[0]["reltuples"] == 0 or result[0]["reltuples"] == -1)
    ):
        verbose_logger.debug("Should create views")
        return True

    return False
