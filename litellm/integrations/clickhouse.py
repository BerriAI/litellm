# callback to make a request to an API endpoint

#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os
import requests

from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache

from typing import Literal, Union

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback


#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger


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

        # RUN Enterprise Clickhouse Setup
        # TLDR: For Enterprise - we create views / aggregate tables for low latency reporting APIs
        _create_clickhouse_aggregate_tables(client=client, table_names=table_names)
        _create_clickhouse_material_views(client=client, table_names=table_names)


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
            traceback.print_exc()
            verbose_logger.debug(f"Clickhouse - {str(e)}\n{traceback.format_exc()}")
            pass
