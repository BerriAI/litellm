# Enterprise Proxy Util Endpoints
from litellm._logging import verbose_logger


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


async def view_spend_logs_from_clickhouse(
    api_key=None, user_id=None, request_id=None, start_date=None, end_date=None
):
    verbose_logger.debug("Reading logs from Clickhouse")
    import os

    # if user has setup clickhouse
    # TODO: Move this to be a helper function
    # querying clickhouse for this data
    import clickhouse_connect
    from datetime import datetime

    port = os.getenv("CLICKHOUSE_PORT")
    if port is not None and isinstance(port, str):
        port = int(port)

    client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=port,
        username=os.getenv("CLICKHOUSE_USERNAME", ""),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )
    if (
        start_date is not None
        and isinstance(start_date, str)
        and end_date is not None
        and isinstance(end_date, str)
    ):
        # Convert the date strings to datetime objects
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        # get top spend per day
        response = client.query(
            f"""
                SELECT
                    toDate(startTime) AS day,
                    sum(spend) AS total_spend
                FROM
                    spend_logs
                WHERE
                    toDate(startTime) BETWEEN toDate('2024-02-01') AND toDate('2024-02-29')
                GROUP BY
                    day
                ORDER BY
                    total_spend
                """
        )

        results = []
        result_rows = list(response.result_rows)
        for response in result_rows:
            current_row = {}
            current_row["users"] = {"example": 0.0}
            current_row["models"] = {}

            current_row["spend"] = float(response[1])
            current_row["startTime"] = str(response[0])

            # stubbed api_key
            current_row[""] = 0.0  # type: ignore
            results.append(current_row)

        return results
    else:
        # check if spend logs exist, if it does then return last 10 logs, sorted in descending order of startTime
        response = client.query(
            """
                SELECT
                    *
                FROM
                    default.spend_logs
                ORDER BY
                    startTime DESC
                LIMIT
                    10
            """
        )

        # get size of spend logs
        num_rows = client.query("SELECT count(*) FROM default.spend_logs")
        num_rows = num_rows.result_rows[0][0]

        # safely access num_rows.result_rows[0][0]
        if num_rows is None:
            num_rows = 0

        raw_rows = list(response.result_rows)
        response_data = {
            "logs": raw_rows,
            "log_count": num_rows,
        }
        return response_data
