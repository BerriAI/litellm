from typing import Any

from litellm import verbose_logger

_db = Any


async def create_missing_views(db: _db, dialect: str = "postgresql"):  # noqa: PLR0915
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
        print("LiteLLM_VerificationTokenView Exists!")  # noqa
    except Exception:
        # If an error occurs, the view does not exist, so create it
        await db.execute_raw(
            """
                CREATE VIEW "LiteLLM_VerificationTokenView" AS
                SELECT 
                v.*, 
                t.spend AS team_spend, 
                t.max_budget AS team_max_budget, 
                t.tpm_limit AS team_tpm_limit, 
                t.rpm_limit AS team_rpm_limit
                FROM "LiteLLM_VerificationToken" v
                LEFT JOIN "LiteLLM_TeamTable" t ON v.team_id = t.team_id;
            """
        )

        print("LiteLLM_VerificationTokenView Created!")  # noqa

    # Helper for creating/replacing views
    async def create_view(name: str, sql: str):
        try:
            await db.query_raw(f"""SELECT 1 FROM "{name}" LIMIT 1""")
            print(f"{name} Exists!")  # noqa
        except Exception:
            if dialect == "postgresql":
                await db.execute_raw(query=f"CREATE OR REPLACE VIEW \"{name}\" AS {sql}")
            else:
                await db.execute_raw(query=f"DROP VIEW IF EXISTS \"{name}\"")
                await db.execute_raw(query=f"CREATE VIEW \"{name}\" AS {sql}")
            print(f"{name} Created!")  # noqa

    # Common snippets
    if dialect == "postgresql":
        date_snippet = 'DATE("startTime")'
        interval_snippet = "CURRENT_DATE - INTERVAL '30 days'"
    else:
        date_snippet = "date(\"startTime\")"
        interval_snippet = "date('now', '-30 days')"

    # MonthlyGlobalSpend
    await create_view(
        "MonthlyGlobalSpend",
        f"""
        SELECT
        {date_snippet} AS date, 
        SUM("spend") AS spend 
        FROM 
        "LiteLLM_SpendLogs" 
        WHERE 
        "startTime" >= ({interval_snippet})
        GROUP BY 
        {date_snippet};
        """,
    )

    # Last30dKeysBySpend
    await create_view(
        "Last30dKeysBySpend",
        f"""
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
        L."startTime" >= ({interval_snippet})
        GROUP BY
        L."api_key", V."key_alias", V."key_name"
        ORDER BY
        total_spend DESC;
        """,
    )

    # Last30dModelsBySpend
    await create_view(
        "Last30dModelsBySpend",
        f"""
        SELECT
        "model",
        SUM("spend") AS total_spend
        FROM
        "LiteLLM_SpendLogs"
        WHERE
        "startTime" >= ({interval_snippet})
        AND "model" != ''
        GROUP BY
        "model"
        ORDER BY
        total_spend DESC;
        """,
    )

    # MonthlyGlobalSpendPerKey
    await create_view(
        "MonthlyGlobalSpendPerKey",
        f"""
            SELECT
            {date_snippet} AS date, 
            SUM("spend") AS spend,
            api_key as api_key
            FROM 
            "LiteLLM_SpendLogs" 
            WHERE 
            "startTime" >= ({interval_snippet})
            GROUP BY 
            {date_snippet},
            api_key;
        """,
    )

    # MonthlyGlobalSpendPerUserPerKey
    await create_view(
        "MonthlyGlobalSpendPerUserPerKey",
        f"""
            SELECT
            {date_snippet} AS date, 
            SUM("spend") AS spend,
            api_key as api_key,
            "user" as "user"
            FROM 
            "LiteLLM_SpendLogs" 
            WHERE 
            "startTime" >= ({interval_snippet})
            GROUP BY 
            {date_snippet},
            "user",
            api_key;
        """,
    )

    # DailyTagSpend
    if dialect == "postgresql":
        tag_sql = """
        SELECT
            jsonb_array_elements_text(request_tags) AS individual_request_tag,
            DATE(s."startTime") AS spend_date,
            COUNT(*) AS log_count,
            SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs" s
        GROUP BY individual_request_tag, DATE(s."startTime");
        """
    else:
        # SQLite: request_tags is expected to be a JSON string. json_each can iterate over it.
        tag_sql = """
        SELECT
            json_each.value AS individual_request_tag,
            date(s."startTime") AS spend_date,
            COUNT(*) AS log_count,
            SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs" s, json_each(s.request_tags)
        GROUP BY individual_request_tag, date(s."startTime");
        """
    await create_view("DailyTagSpend", tag_sql)

    # Last30dTopEndUsersSpend
    await create_view(
        "Last30dTopEndUsersSpend",
        f"""
        SELECT end_user, COUNT(*) AS total_events, SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        WHERE end_user <> '' AND end_user <> user
        AND "startTime" >= {interval_snippet}
        GROUP BY end_user
        ORDER BY total_spend DESC
        LIMIT 100;
        """,
    )

    return


async def should_create_missing_views(db: _db, dialect: str = "postgresql") -> bool:
    """
    Run only on first time startup.

    If SpendLogs table already has values, then don't create views on startup.
    """

    if dialect == "postgresql":
        sql_query = """
        SELECT reltuples::BIGINT
        FROM pg_class
        WHERE oid = '"LiteLLM_SpendLogs"'::regclass;
        """
    else:
        # SQLite
        sql_query = """
        SELECT count(*) as reltuples FROM "LiteLLM_SpendLogs" LIMIT 1
        """

    result = await db.query_raw(query=sql_query)

    verbose_logger.debug("Estimated Row count of LiteLLM_SpendLogs = {}".format(result))
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
