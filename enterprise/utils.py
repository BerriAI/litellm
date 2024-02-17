# Enterprise Proxy Util Endpoints


async def get_spend_by_tags(start_date=None, end_date=None, prisma_client=None):
    response = await prisma_client.db.query_raw(
        """
        SELECT
        jsonb_array_elements_text(request_tags) AS individual_request_tag,
        COUNT(*) AS log_count,
        SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        GROUP BY individual_request_tag;
        """
    )

    return response
