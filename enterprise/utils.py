# Enterprise Proxy Util Endpoints
from typing import Optional, List
from litellm._logging import verbose_logger
from litellm.proxy.proxy_server import PrismaClient, HTTPException
import collections
from datetime import datetime


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


async def ui_get_spend_by_tags(
    start_date: str,
    end_date: str,
    prisma_client: Optional[PrismaClient] = None,
    tags_str: Optional[str] = None,
):
    """
    Should cover 2 cases:
    1. When user is getting spend for all_tags. "all_tags" in tags_list
    2. When user is getting spend for specific tags.
    """

    # tags_str is a list of strings csv of tags
    # tags_str = tag1,tag2,tag3
    # convert to list if it's not None
    tags_list: Optional[List[str]] = None
    if tags_str is not None and len(tags_str) > 0:
        tags_list = tags_str.split(",")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    response = None
    if tags_list is None or (isinstance(tags_list, list) and "all-tags" in tags_list):
        # Get spend for all tags
        sql_query = """
            SELECT
            jsonb_array_elements_text(request_tags) AS individual_request_tag,
            DATE(s."startTime") AS spend_date,
            COUNT(*) AS log_count,
            SUM(spend) AS total_spend
            FROM "LiteLLM_SpendLogs" s
            WHERE
                DATE(s."startTime") >= $1::date
                AND DATE(s."startTime") <= $2::date
            GROUP BY individual_request_tag, spend_date
            ORDER BY total_spend DESC;
        """
        response = await prisma_client.db.query_raw(
            sql_query,
            start_date,
            end_date,
        )
    else:
        # filter by tags list
        sql_query = """
            SELECT
                individual_request_tag,
                COUNT(*) AS log_count,
                SUM(spend) AS total_spend
            FROM (
                SELECT
                    jsonb_array_elements_text(request_tags) AS individual_request_tag,
                    DATE(s."startTime") AS spend_date,
                    spend
                FROM "LiteLLM_SpendLogs" s
                WHERE
                    DATE(s."startTime") >= $1::date
                    AND DATE(s."startTime") <= $2::date
            ) AS subquery
            WHERE individual_request_tag = ANY($3::text[])
            GROUP BY individual_request_tag
            ORDER BY total_spend DESC;
        """
        response = await prisma_client.db.query_raw(
            sql_query,
            start_date,
            end_date,
            tags_list,
        )

    # print("tags - spend")
    # print(response)
    # Bar Chart 1 - Spend per tag - Top 10 tags by spend
    total_spend_per_tag: collections.defaultdict = collections.defaultdict(float)
    total_requests_per_tag: collections.defaultdict = collections.defaultdict(int)
    for row in response:
        tag_name = row["individual_request_tag"]
        tag_spend = row["total_spend"]

        total_spend_per_tag[tag_name] += tag_spend
        total_requests_per_tag[tag_name] += row["log_count"]

    sorted_tags = sorted(total_spend_per_tag.items(), key=lambda x: x[1], reverse=True)
    # convert to ui format
    ui_tags = []
    for tag in sorted_tags:
        current_spend = tag[1]
        if current_spend is not None and isinstance(current_spend, float):
            current_spend = round(current_spend, 4)
        ui_tags.append(
            {
                "name": tag[0],
                "spend": current_spend,
                "log_count": total_requests_per_tag[tag[0]],
            }
        )

    return {"spend_per_tag": ui_tags}


def _forecast_daily_cost(data: list):
    import requests  # type: ignore
    from datetime import datetime, timedelta

    if len(data) == 0:
        return {
            "response": [],
            "predicted_spend": "Current Spend = $0, Predicted = $0",
        }
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
