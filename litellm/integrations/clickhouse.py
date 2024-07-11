# callback to make a request to an API endpoint

#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os

from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache

from typing import Literal, Union
import traceback


#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger


def create_client():
    try:
        import clickhouse_connect

        port = os.getenv("CLICKHOUSE_PORT")
        clickhouse_host = os.getenv("CLICKHOUSE_HOST")
        if clickhouse_host is not None:
            verbose_logger.debug("setting up clickhouse")
            if port is not None and isinstance(port, str):
                port = int(port)

            client = clickhouse_connect.get_client(
                host=os.getenv("CLICKHOUSE_HOST"),
                port=port,
                username=os.getenv("CLICKHOUSE_USERNAME"),
                password=os.getenv("CLICKHOUSE_PASSWORD"),
            )
            return client
        else:
            raise Exception("Clickhouse: Clickhouse host not set")
    except Exception as e:
        raise ValueError(f"Clickhouse: {e}")


def build_daily_metrics():
    click_house_client = create_client()

    # get daily spend
    daily_spend = click_house_client.query_df(
        """
        SELECT sumMerge(DailySpend) as daily_spend, day FROM daily_aggregated_spend GROUP BY day
        """
    )

    # get daily spend per model
    daily_spend_per_model = click_house_client.query_df(
        """
        SELECT sumMerge(DailySpend) as daily_spend, day, model FROM daily_aggregated_spend_per_model GROUP BY day, model
        """
    )
    new_df = daily_spend_per_model.to_dict(orient="records")
    import pandas as pd

    df = pd.DataFrame(new_df)
    # Group by 'day' and create a dictionary for each group
    result_dict = {}
    for day, group in df.groupby("day"):
        models = group["model"].tolist()
        spend = group["daily_spend"].tolist()
        spend_per_model = {model: spend for model, spend in zip(models, spend)}
        result_dict[day] = spend_per_model

    # Display the resulting dictionary

    # get daily spend per API key
    daily_spend_per_api_key = click_house_client.query_df(
        """
            SELECT
                daily_spend,
                day,
                api_key
            FROM (
                SELECT
                    sumMerge(DailySpend) as daily_spend,
                    day,
                    api_key,
                    RANK() OVER (PARTITION BY day ORDER BY sumMerge(DailySpend) DESC) as spend_rank
                FROM
                    daily_aggregated_spend_per_api_key
                GROUP BY
                    day,
                    api_key
            ) AS ranked_api_keys
            WHERE
                spend_rank <= 5
                AND day IS NOT NULL
            ORDER BY
                day,
                daily_spend DESC
        """
    )
    new_df = daily_spend_per_api_key.to_dict(orient="records")
    import pandas as pd

    df = pd.DataFrame(new_df)
    # Group by 'day' and create a dictionary for each group
    api_key_result_dict = {}
    for day, group in df.groupby("day"):
        api_keys = group["api_key"].tolist()
        spend = group["daily_spend"].tolist()
        spend_per_api_key = {api_key: spend for api_key, spend in zip(api_keys, spend)}
        api_key_result_dict[day] = spend_per_api_key

    # Display the resulting dictionary

    # Calculate total spend across all days
    total_spend = daily_spend["daily_spend"].sum()

    # Identify top models and top API keys with the highest spend across all days
    top_models = {}
    top_api_keys = {}

    for day, spend_per_model in result_dict.items():
        for model, model_spend in spend_per_model.items():
            if model not in top_models or model_spend > top_models[model]:
                top_models[model] = model_spend

    for day, spend_per_api_key in api_key_result_dict.items():
        for api_key, api_key_spend in spend_per_api_key.items():
            if api_key not in top_api_keys or api_key_spend > top_api_keys[api_key]:
                top_api_keys[api_key] = api_key_spend

    # for each day in daily spend, look up the day in result_dict and api_key_result_dict
    # Assuming daily_spend DataFrame has 'day' column
    result = []
    for index, row in daily_spend.iterrows():
        day = row["day"]
        data_day = row.to_dict()

        # Look up in result_dict
        if day in result_dict:
            spend_per_model = result_dict[day]
            # Assuming there is a column named 'model' in daily_spend
            data_day["spend_per_model"] = spend_per_model  # Assign 0 if model not found

        # Look up in api_key_result_dict
        if day in api_key_result_dict:
            spend_per_api_key = api_key_result_dict[day]
            # Assuming there is a column named 'api_key' in daily_spend
            data_day["spend_per_api_key"] = spend_per_api_key

        result.append(data_day)

    data_to_return = {}
    data_to_return["daily_spend"] = result

    data_to_return["total_spend"] = total_spend
    data_to_return["top_models"] = top_models
    data_to_return["top_api_keys"] = top_api_keys
    return data_to_return


# build_daily_metrics()


def _start_clickhouse():
    import clickhouse_connect

    port = os.getenv("CLICKHOUSE_PORT")
    clickhouse_host = os.getenv("CLICKHOUSE_HOST")
    if clickhouse_host is not None:
        verbose_logger.debug("setting up clickhouse")
        if port is not None and isinstance(port, str):
            port = int(port)

        client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST"),
            port=port,
            username=os.getenv("CLICKHOUSE_USERNAME"),
            password=os.getenv("CLICKHOUSE_PASSWORD"),
        )
        # view all tables in DB
        response = client.query("SHOW TABLES")
        verbose_logger.debug(
            f"checking if litellm spend logs exists, all tables={response.result_rows}"
        )
        # all tables is returned like this: all tables = [('new_table',), ('spend_logs',)]
        # check if spend_logs in all tables
        table_names = [all_tables[0] for all_tables in response.result_rows]

        if "spend_logs" not in table_names:
            verbose_logger.debug(
                "Clickhouse: spend logs table does not exist... creating it"
            )

            response = client.command(
                """
                CREATE TABLE default.spend_logs
                (
                    `request_id` String,
                    `call_type` String,
                    `api_key` String,
                    `spend` Float64,
                    `total_tokens` Int256,
                    `prompt_tokens` Int256,
                    `completion_tokens` Int256,
                    `startTime` DateTime,
                    `endTime` DateTime,
                    `model` String,
                    `user` String,
                    `metadata` String,
                    `cache_hit` String,
                    `cache_key` String,
                    `request_tags` String
                )
                ENGINE = MergeTree
                ORDER BY tuple();
                """
            )
        else:
            # check if spend logs exist, if it does then return the schema
            response = client.query("DESCRIBE default.spend_logs")
            verbose_logger.debug(f"spend logs schema ={response.result_rows}")


class ClickhouseLogger:
    # Class variables or attributes
    def __init__(self, endpoint=None, headers=None):
        import clickhouse_connect

        _start_clickhouse()

        verbose_logger.debug(
            f"ClickhouseLogger init, host {os.getenv('CLICKHOUSE_HOST')}, port {os.getenv('CLICKHOUSE_PORT')}, username {os.getenv('CLICKHOUSE_USERNAME')}"
        )

        port = os.getenv("CLICKHOUSE_PORT")
        if port is not None and isinstance(port, str):
            port = int(port)

        client = clickhouse_connect.get_client(
            host=os.getenv("CLICKHOUSE_HOST"),
            port=port,
            username=os.getenv("CLICKHOUSE_USERNAME"),
            password=os.getenv("CLICKHOUSE_PASSWORD"),
        )
        self.client = client

    # This is sync, because we run this in a separate thread. Running in a sepearate thread ensures it will never block an LLM API call
    # Experience with s3, Langfuse shows that async logging events are complicated and can block LLM calls
    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        try:
            verbose_logger.debug(
                f"ClickhouseLogger Logging - Enters logging function for model {kwargs}"
            )
            # follows the same params as langfuse.py
            from litellm.proxy.utils import get_logging_payload

            payload = get_logging_payload(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            metadata = payload.get("metadata", "") or ""
            request_tags = payload.get("request_tags", "") or ""
            payload["metadata"] = str(metadata)
            payload["request_tags"] = str(request_tags)
            # Build the initial payload

            verbose_logger.debug(f"\nClickhouse Logger - Logging payload = {payload}")

            # just get the payload items in one array and payload keys in 2nd array
            values = []
            keys = []
            for key, value in payload.items():
                keys.append(key)
                values.append(value)
            data = [values]

            response = self.client.insert("default.spend_logs", data, column_names=keys)

            # make request to endpoint with payload
            verbose_logger.debug(f"Clickhouse Logger - final response = {response}")
        except Exception as e:
            verbose_logger.debug(f"Clickhouse - {str(e)}\n{traceback.format_exc()}")
            pass
