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


def _create_clickhouse_material_views(client=None, table_names=[]):
    # Create Materialized Views if they don't exist
    # Materialized Views send new inserted rows to the aggregate tables

    verbose_logger.debug("Clickhouse: Creating Materialized Views")
    if "daily_aggregated_spend_per_model_mv" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend_per_model_mv")
        client.command(
            """
            CREATE MATERIALIZED VIEW daily_aggregated_spend_per_model_mv
            TO daily_aggregated_spend_per_model
            AS
            SELECT
                toDate(startTime) as day,
                sumState(spend) AS DailySpend,
                model as model
            FROM spend_logs
            GROUP BY
                day, model
            """
        )
    if "daily_aggregated_spend_per_api_key_mv" not in table_names:
        verbose_logger.debug(
            "Clickhouse: Creating daily_aggregated_spend_per_api_key_mv"
        )
        client.command(
            """
            CREATE MATERIALIZED VIEW daily_aggregated_spend_per_api_key_mv
            TO daily_aggregated_spend_per_api_key
            AS
            SELECT
                toDate(startTime) as day,
                sumState(spend) AS DailySpend,
                api_key as api_key
            FROM spend_logs
            GROUP BY
                day, api_key
            """
        )
    if "daily_aggregated_spend_per_user_mv" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend_per_user_mv")
        client.command(
            """
            CREATE MATERIALIZED VIEW daily_aggregated_spend_per_user_mv
            TO daily_aggregated_spend_per_user
            AS
            SELECT
                toDate(startTime) as day,
                sumState(spend) AS DailySpend,
                user as user
            FROM spend_logs
            GROUP BY
                day, user
            """
        )
    if "daily_aggregated_spend_mv" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend_mv")
        client.command(
            """
            CREATE MATERIALIZED VIEW daily_aggregated_spend_mv
            TO daily_aggregated_spend
            AS
            SELECT
                toDate(startTime) as day,
                sumState(spend) AS DailySpend
            FROM spend_logs
            GROUP BY
                day
            """
        )


def _create_clickhouse_aggregate_tables(client=None, table_names=[]):
    # Basic Logging works without this - this is only used for low latency reporting apis
    verbose_logger.debug("Clickhouse: Creating Aggregate Tables")

    # Create Aggregeate Tables if they don't exist
    if "daily_aggregated_spend_per_model" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend_per_model")
        client.command(
            """
            CREATE TABLE daily_aggregated_spend_per_model
            (
                `day` Date,
                `DailySpend` AggregateFunction(sum, Float64),
                `model` String
            )
            ENGINE = SummingMergeTree()
            ORDER BY (day, model);
            """
        )
    if "daily_aggregated_spend_per_api_key" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend_per_api_key")
        client.command(
            """
            CREATE TABLE daily_aggregated_spend_per_api_key
            (
                `day` Date,
                `DailySpend` AggregateFunction(sum, Float64),
                `api_key` String
            )
            ENGINE = SummingMergeTree()
            ORDER BY (day, api_key);
            """
        )
    if "daily_aggregated_spend_per_user" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend_per_user")
        client.command(
            """
            CREATE TABLE daily_aggregated_spend_per_user
            (
                `day` Date,
                `DailySpend` AggregateFunction(sum, Float64),
                `user` String
            )
            ENGINE = SummingMergeTree()
            ORDER BY (day, user);
            """
        )
    if "daily_aggregated_spend" not in table_names:
        verbose_logger.debug("Clickhouse: Creating daily_aggregated_spend")
        client.command(
            """
            CREATE TABLE daily_aggregated_spend
            (
                `day` Date,
                `DailySpend` AggregateFunction(sum, Float64),
            )
            ENGINE = SummingMergeTree()
            ORDER BY (day);
            """
        )
    return


def _forecast_daily_cost(data: list):
    import requests
    from datetime import datetime, timedelta

    first_entry = data[0]
    last_entry = data[-1]

    # get the date today
    today_date = datetime.today().date()

    today_day_month = today_date.month

    # Parse the date from the first entry
    first_entry_date = datetime.strptime(first_entry["date"], "%Y-%m-%d").date()
    last_entry_date = datetime.strptime(last_entry["date"], "%Y-%m-%d")

    print("last entry date", last_entry_date)

    # Assuming today_date is a datetime object
    today_date = datetime.now()

    # Calculate the last day of the month
    last_day_of_todays_month = datetime(
        today_date.year, today_date.month % 12 + 1, 1
    ) - timedelta(days=1)

    # Calculate the remaining days in the month
    remaining_days = (last_day_of_todays_month - last_entry_date).days

    current_spend_this_month = 0
    series = {}
    for entry in data:
        date = entry["date"]
        spend = entry["spend"]
        series[date] = spend

        # check if the date is in this month
        if datetime.strptime(date, "%Y-%m-%d").month == today_day_month:
            current_spend_this_month += spend

    if len(series) < 10:
        num_items_to_fill = 11 - len(series)

        # avg spend for all days in series
        avg_spend = sum(series.values()) / len(series)
        for i in range(num_items_to_fill):
            # go backwards from the first entry
            date = first_entry_date - timedelta(days=i)
            series[date.strftime("%Y-%m-%d")] = avg_spend
            series[date.strftime("%Y-%m-%d")] = avg_spend

    payload = {"series": series, "count": remaining_days}
    print("Prediction Data:", payload)

    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        url="https://trend-api-production.up.railway.app/forecast",
        json=payload,
        headers=headers,
    )
    # check the status code
    response.raise_for_status()

    json_response = response.json()
    forecast_data = json_response["forecast"]

    # print("Forecast Data:", forecast_data)

    response_data = []
    total_predicted_spend = current_spend_this_month
    for date in forecast_data:
        spend = forecast_data[date]
        entry = {
            "date": date,
            "predicted_spend": spend,
        }
        total_predicted_spend += spend
        response_data.append(entry)

    # get month as a string, Jan, Feb, etc.
    today_month = today_date.strftime("%B")
    predicted_spend = (
        f"Predicted Spend for { today_month } 2024, ${total_predicted_spend}"
    )
    return {"response": response_data, "predicted_spend": predicted_spend}

    # print(f"Date: {entry['date']}, Spend: {entry['spend']}, Response: {response.text}")


# _forecast_daily_cost(
#     [
#         {"date": "2022-01-01", "spend": 100},

#     ]
# )
