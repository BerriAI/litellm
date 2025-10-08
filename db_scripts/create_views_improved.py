"""
Python script to pre-create all views required by LiteLLM Proxy Server
This version is designed for container startup with better error handling.
"""

import asyncio
import os
import sys

# Only show warnings for actual errors, not expected conditions
SILENT_MODE = os.getenv("CREATE_VIEWS_SILENT", "false").lower() == "true"

def log_info(msg):
    """Print info messages unless in silent mode"""
    if not SILENT_MODE:
        print(f"INFO: {msg}")

def log_error(msg):
    """Always print error messages"""
    print(f"ERROR: {msg}", file=sys.stderr)

async def check_view_exists():  # noqa: PLR0915
    """
    Checks if the LiteLLM views exist in the user's db and creates them if not.
    """
    # Check if DATABASE_URL is configured
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        log_info("No DATABASE_URL configured, skipping view creation")
        return
    
    # Check if it's a PostgreSQL database
    if not DATABASE_URL.startswith(("postgresql://", "postgres://")):
        log_info("DATABASE_URL is not PostgreSQL, skipping view creation")
        return
    
    # Try to import prisma
    try:
        from prisma import Prisma
    except ImportError as e:
        log_error(f"Failed to import Prisma: {e}")
        sys.exit(1)
    
    # Try to connect to database
    try:
        db = Prisma(
            datasource={"url": DATABASE_URL},
            http={"timeout": 60000},
        )
        await db.connect()
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a connection issue (expected if DB isn't ready yet)
        if any(word in error_str for word in ["connection", "connect", "refused", "timeout", "host"]):
            log_info("Database not ready yet, skipping view creation")
            return
        else:
            log_error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    log_info("Creating LITELLM views...")
    
    # Create or verify each view
    views_status = []
    
    try:
        await db.query_raw("""SELECT 1 FROM "LiteLLM_VerificationTokenView" LIMIT 1""")
        views_status.append("LiteLLM_VerificationTokenView Exists!")
    except Exception:
        await db.execute_raw(
            """
                CREATE VIEW "LiteLLM_VerificationTokenView" AS
                SELECT 
                    k.token, 
                    k.key_alias, 
                    k.key_name,
                    k.budget_id,
                    k.team_id,
                    k.user_id,
                    t.max_parallel_requests,
                    t.team_alias, 
                    t.metadata as team_metadata,
                    t.tpm_limit,
                    t.rpm_limit,
                    t.budget_duration,
                    t.budget_reset_at,
                    t."blocked"
                FROM "LiteLLM_VerificationToken" AS k
                LEFT JOIN "LiteLLM_TeamTable" AS t ON k.team_id = t.team_id;
                """
        )
        views_status.append("LiteLLM_VerificationTokenView Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpend" LIMIT 1""")
        views_status.append("MonthlyGlobalSpend Exists!")
    except Exception:
        sql_query = """
        CREATE VIEW "MonthlyGlobalSpend" AS 
        SELECT 
            SUM(spend) AS spend, 
            DATE_TRUNC('month', "startTime")::DATE AS month,
            COUNT(*) AS total_events
        FROM 
            "LiteLLM_SpendLogs"
        WHERE 
            "startTime" >= DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 
            month;
        """
        await db.execute_raw(query=sql_query)
        views_status.append("MonthlyGlobalSpend Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "Last30dKeysBySpend" LIMIT 1""")
        views_status.append("Last30dKeysBySpend Exists!")
    except Exception:
        sql_query = """
        CREATE VIEW "Last30dKeysBySpend" AS 
        SELECT 
            l.api_key,
            SUM(l.spend) AS total_spend,
            COALESCE(MAX(k.team_id), '') AS team_id,
            COALESCE(MAX(k.key_alias), '') AS key_alias,
            COALESCE(MAX(k.key_name), l.api_key) AS key_name,
            MAX(k.last_refreshed_at) AS last_refreshed_at
        FROM 
            "LiteLLM_SpendLogs" l
        LEFT JOIN 
            "LiteLLM_VerificationToken" k ON l.api_key = k.token
        WHERE 
            l."startTime" >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY 
            l.api_key
        ORDER BY 
            total_spend DESC;
        """
        await db.execute_raw(query=sql_query)
        views_status.append("Last30dKeysBySpend Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "Last30dModelsBySpend" LIMIT 1""")
        views_status.append("Last30dModelsBySpend Exists!")
    except Exception:
        sql_query = """
        CREATE VIEW "Last30dModelsBySpend" AS 
        SELECT 
            model,
            SUM(spend) AS total_spend
        FROM 
            "LiteLLM_SpendLogs"
        WHERE 
            "startTime" >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY 
            model
        ORDER BY 
            total_spend DESC;
        """
        await db.execute_raw(query=sql_query)
        views_status.append("Last30dModelsBySpend Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpendPerKey" LIMIT 1""")
        views_status.append("MonthlyGlobalSpendPerKey Exists!")
    except Exception:
        sql_query = """
        CREATE VIEW "MonthlyGlobalSpendPerKey" AS 
        SELECT 
            api_key,
            DATE_TRUNC('month', "startTime")::DATE AS month,
            SUM(spend) AS total_spend,
            COUNT(*) AS total_events
        FROM 
            "LiteLLM_SpendLogs"
        GROUP BY 
            api_key, month
        ORDER BY 
            api_key, month DESC;
        """
        await db.execute_raw(query=sql_query)
        views_status.append("MonthlyGlobalSpendPerKey Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpendPerUserPerKey" LIMIT 1""")
        views_status.append("MonthlyGlobalSpendPerUserPerKey Exists!")
    except Exception:
        sql_query = """
        CREATE VIEW "MonthlyGlobalSpendPerUserPerKey" AS 
        SELECT 
            api_key,
            "user",
            DATE_TRUNC('month', "startTime")::DATE AS month,
            SUM(spend) AS total_spend,
            COUNT(*) AS total_events
        FROM 
            "LiteLLM_SpendLogs"
        GROUP BY 
            api_key, "user", month
        ORDER BY 
            api_key, "user", month DESC;
        """
        await db.execute_raw(query=sql_query)
        views_status.append("MonthlyGlobalSpendPerUserPerKey Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "DailyTagSpend" LIMIT 1""")
        views_status.append("DailyTagSpend Exists!")
    except Exception:
        sql_query = """
        CREATE VIEW "DailyTagSpend" AS
        SELECT 
            individual_request_tag,
            DATE(s."startTime") AS spend_date,
            SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs" s
        GROUP BY individual_request_tag, DATE(s."startTime");
        """
        await db.execute_raw(query=sql_query)
        views_status.append("DailyTagSpend Created!")
    
    try:
        await db.query_raw("""SELECT 1 FROM "Last30dTopEndUsersSpend" LIMIT 1""")
        views_status.append("Last30dTopEndUsersSpend Exists!")
    except Exception:
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
        await db.execute_raw(query=sql_query)
        views_status.append("Last30dTopEndUsersSpend Created!")
    
    # Print all statuses at once
    for status in views_status:
        print(status)
    
    # Disconnect from database
    await db.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(check_view_exists())
    except KeyboardInterrupt:
        log_info("View creation interrupted")
        sys.exit(0)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        sys.exit(1)
