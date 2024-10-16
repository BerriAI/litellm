from prisma import Prisma


async def apply_db_fixes(db: Prisma):
    try:
        sql_query = """
            UPDATE "LiteLLM_SpendLogs"
            SET team_id = (
                SELECT vt.team_id
                FROM "LiteLLM_VerificationToken" vt
                WHERE vt.token = "LiteLLM_SpendLogs".api_key
            )
            WHERE team_id IS NULL
            AND EXISTS (
                SELECT 1
                FROM "LiteLLM_VerificationToken" vt
                WHERE vt.token = "LiteLLM_SpendLogs".api_key
            );
        """
        response = await db.query_raw(sql_query)
        print(
            "Updated unassigned teams, Response=%s",
            response,
        )
    except Exception as e:
        raise Exception(f"Error apply_db_fixes: {str(e)}")
    return
