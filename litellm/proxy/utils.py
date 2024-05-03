from typing import Optional, List, Any, Literal, Union
import os, subprocess, hashlib, importlib, asyncio, copy, json, aiohttp, httpx, time
import litellm, backoff, traceback
from litellm.proxy._types import (
    UserAPIKeyAuth,
    DynamoDBArgs,
    LiteLLM_VerificationToken,
    LiteLLM_VerificationTokenView,
    LiteLLM_SpendLogs,
    LiteLLM_UserTable,
    LiteLLM_EndUserTable,
    LiteLLM_TeamTable,
    Member,
)
from litellm.caching import DualCache, RedisCache
from litellm.router import Deployment, ModelInfo, LiteLLM_Params
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm._service_logger import ServiceLogging, ServiceTypes
from litellm import ModelResponse, EmbeddingResponse, ImageResponse
from litellm.proxy.hooks.max_budget_limiter import _PROXY_MaxBudgetLimiter
from litellm.proxy.hooks.tpm_rpm_limiter import _PROXY_MaxTPMRPMLimiter
from litellm.proxy.hooks.cache_control_check import _PROXY_CacheControlCheck
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.db.base_client import CustomDB
from litellm._logging import verbose_proxy_logger
from fastapi import HTTPException, status
import smtplib, re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from litellm.integrations.slack_alerting import SlackAlerting


def print_verbose(print_statement):
    verbose_proxy_logger.debug(print_statement)
    if litellm.set_verbose:
        print(f"LiteLLM Proxy: {print_statement}")  # noqa


### LOGGING ###
class ProxyLogging:
    """
    Logging/Custom Handlers for proxy.

    Implemented mainly to:
    - log successful/failed db read/writes
    - support the max parallel request integration
    """

    def __init__(
        self,
        user_api_key_cache: DualCache,
    ):
        ## INITIALIZE  LITELLM CALLBACKS ##
        self.call_details: dict = {}
        self.call_details["user_api_key_cache"] = user_api_key_cache
        self.internal_usage_cache = DualCache()
        self.max_parallel_request_limiter = _PROXY_MaxParallelRequestsHandler()
        self.max_tpm_rpm_limiter = _PROXY_MaxTPMRPMLimiter(
            internal_cache=self.internal_usage_cache
        )
        self.max_budget_limiter = _PROXY_MaxBudgetLimiter()
        self.cache_control_check = _PROXY_CacheControlCheck()
        self.alerting: Optional[List] = None
        self.alerting_threshold: float = 300  # default to 5 min. threshold
        self.alert_types: List[
            Literal[
                "llm_exceptions",
                "llm_too_slow",
                "llm_requests_hanging",
                "budget_alerts",
                "db_exceptions",
            ]
        ] = [
            "llm_exceptions",
            "llm_too_slow",
            "llm_requests_hanging",
            "budget_alerts",
            "db_exceptions",
        ]
        self.slack_alerting_instance = SlackAlerting(
            alerting_threshold=self.alerting_threshold,
            alerting=self.alerting,
            alert_types=self.alert_types,
        )

    def update_values(
        self,
        alerting: Optional[List],
        alerting_threshold: Optional[float],
        redis_cache: Optional[RedisCache],
        alert_types: Optional[
            List[
                Literal[
                    "llm_exceptions",
                    "llm_too_slow",
                    "llm_requests_hanging",
                    "budget_alerts",
                    "db_exceptions",
                ]
            ]
        ] = None,
    ):
        self.alerting = alerting
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold
        if alert_types is not None:
            self.alert_types = alert_types

        self.slack_alerting_instance.update_values(
            alerting=self.alerting,
            alerting_threshold=self.alerting_threshold,
            alert_types=self.alert_types,
        )

        if redis_cache is not None:
            self.internal_usage_cache.redis_cache = redis_cache

    def _init_litellm_callbacks(self):
        print_verbose(f"INITIALIZING LITELLM CALLBACKS!")
        self.service_logging_obj = ServiceLogging()
        litellm.callbacks.append(self.max_parallel_request_limiter)
        litellm.callbacks.append(self.max_tpm_rpm_limiter)
        litellm.callbacks.append(self.max_budget_limiter)
        litellm.callbacks.append(self.cache_control_check)
        litellm.callbacks.append(self.service_logging_obj)
        litellm.success_callback.append(
            self.slack_alerting_instance.response_taking_too_long_callback
        )
        for callback in litellm.callbacks:
            if callback not in litellm.input_callback:
                litellm.input_callback.append(callback)
            if callback not in litellm.success_callback:
                litellm.success_callback.append(callback)
            if callback not in litellm.failure_callback:
                litellm.failure_callback.append(callback)
            if callback not in litellm._async_success_callback:
                litellm._async_success_callback.append(callback)
            if callback not in litellm._async_failure_callback:
                litellm._async_failure_callback.append(callback)

        if (
            len(litellm.input_callback) > 0
            or len(litellm.success_callback) > 0
            or len(litellm.failure_callback) > 0
        ):
            callback_list = list(
                set(
                    litellm.input_callback
                    + litellm.success_callback
                    + litellm.failure_callback
                )
            )
            litellm.utils.set_callbacks(callback_list=callback_list)

    async def pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ):
        """
        Allows users to modify/reject the incoming request to the proxy, without having to deal with parsing Request body.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation
        """
        print_verbose(f"Inside Proxy Logging Pre-call hook!")
        ### ALERTING ###
        asyncio.create_task(
            self.slack_alerting_instance.response_taking_too_long(request_data=data)
        )

        try:
            for callback in litellm.callbacks:
                if isinstance(callback, CustomLogger) and "async_pre_call_hook" in vars(
                    callback.__class__
                ):
                    response = await callback.async_pre_call_hook(
                        user_api_key_dict=user_api_key_dict,
                        cache=self.call_details["user_api_key_cache"],
                        data=data,
                        call_type=call_type,
                    )
                    if response is not None:
                        data = response

            print_verbose(f"final data being sent to {call_type} call: {data}")
            return data
        except Exception as e:
            if "litellm_logging_obj" in data:
                logging_obj: litellm.utils.Logging = data["litellm_logging_obj"]

                ## ASYNC FAILURE HANDLER ##
                error_message = ""
                if isinstance(e, HTTPException):
                    if isinstance(e.detail, str):
                        error_message = e.detail
                    elif isinstance(e.detail, dict):
                        error_message = json.dumps(e.detail)
                    else:
                        error_message = str(e)
                else:
                    error_message = str(e)
                error_raised = Exception(f"{error_message}")
                await logging_obj.async_failure_handler(
                    exception=error_raised,
                    traceback_exception=traceback.format_exc(),
                )

                ## SYNC FAILURE HANDLER ##
                try:
                    logging_obj.failure_handler(
                        error_raised, traceback.format_exc()
                    )  # DO NOT MAKE THREADED - router retry fallback relies on this!
                except Exception as error_val:
                    pass
            raise e

    async def during_call_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ):
        """
        Runs the CustomLogger's async_moderation_hook()
        """
        for callback in litellm.callbacks:
            new_data = copy.deepcopy(data)
            try:
                if isinstance(callback, CustomLogger):
                    await callback.async_moderation_hook(
                        data=new_data,
                        user_api_key_dict=user_api_key_dict,
                        call_type=call_type,
                    )
            except Exception as e:
                raise e
        return data

    async def budget_alerts(
        self,
        type: Literal[
            "token_budget",
            "user_budget",
            "user_and_proxy_budget",
            "failed_budgets",
            "failed_tracking",
            "projected_limit_exceeded",
        ],
        user_max_budget: float,
        user_current_spend: float,
        user_info=None,
        error_message="",
    ):
        if self.alerting is None:
            # do nothing if alerting is not switched on
            return
        await self.slack_alerting_instance.budget_alerts(
            type=type,
            user_max_budget=user_max_budget,
            user_current_spend=user_current_spend,
            user_info=user_info,
            error_message=error_message,
        )

    async def alerting_handler(
        self,
        message: str,
        level: Literal["Low", "Medium", "High"],
        alert_type: Literal[
            "llm_exceptions",
            "llm_too_slow",
            "llm_requests_hanging",
            "budget_alerts",
            "db_exceptions",
        ],
    ):
        """
        Alerting based on thresholds: - https://github.com/BerriAI/litellm/issues/1298

        - Responses taking too long
        - Requests are hanging
        - Calls are failing
        - DB Read/Writes are failing
        - Proxy Close to max budget
        - Key Close to max budget

        Parameters:
            level: str - Low|Medium|High - if calls might fail (Medium) or are failing (High); Currently, no alerts would be 'Low'.
            message: str - what is the alert about
        """
        if self.alerting is None:
            return

        from datetime import datetime

        # Get the current timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
        formatted_message = (
            f"Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
        )
        if _proxy_base_url is not None:
            formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

        for client in self.alerting:
            if client == "slack":
                await self.slack_alerting_instance.send_alert(
                    message=message, level=level, alert_type=alert_type
                )
            elif client == "sentry":
                if litellm.utils.sentry_sdk_instance is not None:
                    litellm.utils.sentry_sdk_instance.capture_message(formatted_message)
                else:
                    raise Exception("Missing SENTRY_DSN from environment")

    async def failure_handler(
        self, original_exception, duration: float, call_type: str, traceback_str=""
    ):
        """
        Log failed db read/writes

        Currently only logs exceptions to sentry
        """
        ### ALERTING ###
        if "db_exceptions" not in self.alert_types:
            return
        if isinstance(original_exception, HTTPException):
            if isinstance(original_exception.detail, str):
                error_message = original_exception.detail
            elif isinstance(original_exception.detail, dict):
                error_message = json.dumps(original_exception.detail)
            else:
                error_message = str(original_exception)
        else:
            error_message = str(original_exception)
        if isinstance(traceback_str, str):
            error_message += traceback_str[:1000]
        asyncio.create_task(
            self.alerting_handler(
                message=f"DB read/write call failed: {error_message}",
                level="High",
                alert_type="db_exceptions",
            )
        )

        if hasattr(self, "service_logging_obj"):
            self.service_logging_obj.async_service_failure_hook(
                service=ServiceTypes.DB,
                duration=duration,
                error=error_message,
                call_type=call_type,
            )

        if litellm.utils.capture_exception:
            litellm.utils.capture_exception(error=original_exception)

    async def post_call_failure_hook(
        self, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth
    ):
        """
        Allows users to raise custom exceptions/log when a call fails, without having to deal with parsing Request body.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation
        """

        ### ALERTING ###
        if "llm_exceptions" not in self.alert_types:
            return
        asyncio.create_task(
            self.alerting_handler(
                message=f"LLM API call failed: {str(original_exception)}",
                level="High",
                alert_type="llm_exceptions",
            )
        )

        for callback in litellm.callbacks:
            try:
                if isinstance(callback, CustomLogger):
                    await callback.async_post_call_failure_hook(
                        user_api_key_dict=user_api_key_dict,
                        original_exception=original_exception,
                    )
            except Exception as e:
                raise e
        return

    async def post_call_success_hook(
        self,
        response: Union[ModelResponse, EmbeddingResponse, ImageResponse],
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        Allow user to modify outgoing data

        Covers:
        1. /chat/completions
        """
        for callback in litellm.callbacks:
            try:
                if isinstance(callback, CustomLogger):
                    await callback.async_post_call_success_hook(
                        user_api_key_dict=user_api_key_dict, response=response
                    )
            except Exception as e:
                raise e
        return response

    async def post_call_streaming_hook(
        self,
        response: str,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """
        - Check outgoing streaming response uptil that point
        - Run through moderation check
        - Reject request if it fails moderation check
        """
        new_response = copy.deepcopy(response)
        for callback in litellm.callbacks:
            try:
                if isinstance(callback, CustomLogger):
                    await callback.async_post_call_streaming_hook(
                        user_api_key_dict=user_api_key_dict, response=new_response
                    )
            except Exception as e:
                raise e
        return new_response


### DB CONNECTOR ###
# Define the retry decorator with backoff strategy
# Function to be called whenever a retry is about to happen
def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    print_verbose(f"Backing off... this was attempt #{details['tries']}")


class PrismaClient:
    user_list_transactons: dict = {}
    end_user_list_transactons: dict = {}
    key_list_transactons: dict = {}
    team_list_transactons: dict = {}
    org_list_transactons: dict = {}
    spend_log_transactions: List = []

    def __init__(self, database_url: str, proxy_logging_obj: ProxyLogging):
        print_verbose(
            "LiteLLM: DATABASE_URL Set in config, trying to 'pip install prisma'"
        )
        ## init logging object
        self.proxy_logging_obj = proxy_logging_obj
        try:
            from prisma import Prisma  # type: ignore
        except Exception as e:
            os.environ["DATABASE_URL"] = database_url
            # Save the current working directory
            original_dir = os.getcwd()
            # set the working directory to where this script is
            abspath = os.path.abspath(__file__)
            dname = os.path.dirname(abspath)
            os.chdir(dname)

            try:
                subprocess.run(["prisma", "generate"])
                subprocess.run(
                    ["prisma", "db", "push", "--accept-data-loss"]
                )  # this looks like a weird edge case when prisma just wont start on render. we need to have the --accept-data-loss
            except:
                raise Exception(
                    f"Unable to run prisma commands. Run `pip install prisma`"
                )
            finally:
                os.chdir(original_dir)
            # Now you can import the Prisma Client
            from prisma import Prisma  # type: ignore

        self.db = Prisma()  # Client to connect to Prisma db

    def hash_token(self, token: str):
        # Hash the string using SHA-256
        hashed_token = hashlib.sha256(token.encode()).hexdigest()

        return hashed_token

    def jsonify_object(self, data: dict) -> dict:
        db_data = copy.deepcopy(data)

        for k, v in db_data.items():
            if isinstance(v, dict):
                try:
                    db_data[k] = json.dumps(v)
                except:
                    # This avoids Prisma retrying this 5 times, and making 5 clients
                    db_data[k] = "failed-to-serialize-json"
        return db_data

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def check_view_exists(self):
        """
        Checks if the LiteLLM_VerificationTokenView and MonthlyGlobalSpend exists in the user's db.

        LiteLLM_VerificationTokenView: This view is used for getting the token + team data in user_api_key_auth

        MonthlyGlobalSpend: This view is used for the admin view to see global spend for this month

        If the view doesn't exist, one will be created.
        """
        try:
            # Try to select one row from the view
            await self.db.query_raw(
                """SELECT 1 FROM "LiteLLM_VerificationTokenView" LIMIT 1"""
            )
            print("LiteLLM_VerificationTokenView Exists!")  # noqa
        except Exception as e:
            # If an error occurs, the view does not exist, so create it
            value = await self.health_check()
            await self.db.execute_raw(
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

        try:
            await self.db.query_raw("""SELECT 1 FROM "MonthlyGlobalSpend" LIMIT 1""")
            print("MonthlyGlobalSpend Exists!")  # noqa
        except Exception as e:
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
            await self.db.execute_raw(query=sql_query)

            print("MonthlyGlobalSpend Created!")  # noqa

        try:
            await self.db.query_raw("""SELECT 1 FROM "Last30dKeysBySpend" LIMIT 1""")
            print("Last30dKeysBySpend Exists!")  # noqa
        except Exception as e:
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
            await self.db.execute_raw(query=sql_query)

            print("Last30dKeysBySpend Created!")  # noqa

        try:
            await self.db.query_raw("""SELECT 1 FROM "Last30dModelsBySpend" LIMIT 1""")
            print("Last30dModelsBySpend Exists!")  # noqa
        except Exception as e:
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
            await self.db.execute_raw(query=sql_query)

            print("Last30dModelsBySpend Created!")  # noqa
        try:
            await self.db.query_raw(
                """SELECT 1 FROM "MonthlyGlobalSpendPerKey" LIMIT 1"""
            )
            print("MonthlyGlobalSpendPerKey Exists!")  # noqa
        except Exception as e:
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
            await self.db.execute_raw(query=sql_query)

            print("MonthlyGlobalSpendPerKey Created!")  # noqa

        try:
            await self.db.query_raw(
                """SELECT 1 FROM "Last30dTopEndUsersSpend" LIMIT 1"""
            )
            print("Last30dTopEndUsersSpend Exists!")  # noqa
        except Exception as e:
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
            await self.db.execute_raw(query=sql_query)

            print("Last30dTopEndUsersSpend Created!")  # noqa

        return

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=1,  # maximum number of retries
        max_time=2,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def get_generic_data(
        self,
        key: str,
        value: Any,
        table_name: Literal["users", "keys", "config", "spend"],
    ):
        """
        Generic implementation of get data
        """
        verbose_proxy_logger.debug(
            f"PrismaClient: get_generic_data: {key}, table_name: {table_name}"
        )
        start_time = time.time()
        try:
            if table_name == "users":
                response = await self.db.litellm_usertable.find_first(
                    where={key: value}  # type: ignore
                )
            elif table_name == "keys":
                response = await self.db.litellm_verificationtoken.find_first(  # type: ignore
                    where={key: value}  # type: ignore
                )
            elif table_name == "config":
                response = await self.db.litellm_config.find_first(  # type: ignore
                    where={key: value}  # type: ignore
                )
            elif table_name == "spend":
                response = await self.db.l.find_first(  # type: ignore
                    where={key: value}  # type: ignore
                )
            return response
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception get_generic_data: {str(e)}"
            verbose_proxy_logger.error(error_msg)
            error_msg = error_msg + "\nException Type: {}".format(type(e))
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    traceback_str=error_traceback,
                    call_type="get_generic_data",
                )
            )

            raise e

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def get_data(
        self,
        token: Optional[Union[str, list]] = None,
        user_id: Optional[str] = None,
        user_id_list: Optional[list] = None,
        team_id: Optional[str] = None,
        team_id_list: Optional[list] = None,
        key_val: Optional[dict] = None,
        table_name: Optional[
            Literal[
                "user",
                "key",
                "config",
                "spend",
                "team",
                "user_notification",
                "combined_view",
            ]
        ] = None,
        query_type: Literal["find_unique", "find_all"] = "find_unique",
        expires: Optional[datetime] = None,
        reset_at: Optional[datetime] = None,
        offset: Optional[int] = None,  # pagination, what row number to start from
        limit: Optional[
            int
        ] = None,  # pagination, number of rows to getch when find_all==True
    ):
        args_passed_in = locals()
        start_time = time.time()
        verbose_proxy_logger.debug(
            f"PrismaClient: get_data - args_passed_in: {args_passed_in}"
        )
        try:
            response: Any = None
            if (token is not None and table_name is None) or (
                table_name is not None and table_name == "key"
            ):
                # check if plain text or hash
                if token is not None:
                    if isinstance(token, str):
                        hashed_token = token
                        if token.startswith("sk-"):
                            hashed_token = self.hash_token(token=token)
                        verbose_proxy_logger.debug(
                            f"PrismaClient: find_unique for token: {hashed_token}"
                        )
                if query_type == "find_unique":
                    if token is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"No token passed in. Token={token}"},
                        )
                    response = await self.db.litellm_verificationtoken.find_unique(
                        where={"token": hashed_token},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None:
                        # for prisma we need to cast the expires time to str
                        if response.expires is not None and isinstance(
                            response.expires, datetime
                        ):
                            response.expires = response.expires.isoformat()
                    else:
                        # Token does not exist.
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Authentication Error: invalid user key - user key does not exist in db. User Key={token}",
                        )
                elif query_type == "find_all" and user_id is not None:
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={"user_id": user_id},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all" and team_id is not None:
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={"team_id": team_id},
                        include={"litellm_budget_table": True},
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif (
                    query_type == "find_all"
                    and expires is not None
                    and reset_at is not None
                ):
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={  # type:ignore
                            "OR": [
                                {"expires": None},
                                {"expires": {"gt": expires}},
                            ],
                            "budget_reset_at": {"lt": reset_at},
                        }
                    )
                    if response is not None and len(response) > 0:
                        for r in response:
                            if isinstance(r.expires, datetime):
                                r.expires = r.expires.isoformat()
                elif query_type == "find_all":
                    where_filter: dict = {}
                    if token is not None:
                        where_filter["token"] = {}
                        if isinstance(token, str):
                            if token.startswith("sk-"):
                                token = self.hash_token(token=token)
                            where_filter["token"]["in"] = [token]
                        elif isinstance(token, list):
                            hashed_tokens = []
                            for t in token:
                                assert isinstance(t, str)
                                if t.startswith("sk-"):
                                    new_token = self.hash_token(token=t)
                                    hashed_tokens.append(new_token)
                                else:
                                    hashed_tokens.append(t)
                            where_filter["token"]["in"] = hashed_tokens
                    response = await self.db.litellm_verificationtoken.find_many(
                        order={"spend": "desc"},
                        where=where_filter,  # type: ignore
                        include={"litellm_budget_table": True},
                    )
                if response is not None:
                    return response
                else:
                    # Token does not exist.
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication Error: invalid user key - token does not exist",
                    )
            elif (user_id is not None and table_name is None) or (
                table_name is not None and table_name == "user"
            ):
                if query_type == "find_unique":
                    if key_val is None:
                        key_val = {"user_id": user_id}
                    response = await self.db.litellm_usertable.find_unique(  # type: ignore
                        where=key_val  # type: ignore
                    )
                elif query_type == "find_all" and key_val is not None:
                    response = await self.db.litellm_usertable.find_many(
                        where=key_val  # type: ignore
                    )  # type: ignore
                elif query_type == "find_all" and reset_at is not None:
                    response = await self.db.litellm_usertable.find_many(
                        where={  # type:ignore
                            "budget_reset_at": {"lt": reset_at},
                        }
                    )
                elif query_type == "find_all" and user_id_list is not None:
                    user_id_values = ", ".join(f"'{item}'" for item in user_id_list)
                    sql_query = f"""
                    SELECT *
                    FROM "LiteLLM_UserTable"
                    WHERE "user_id" IN ({user_id_values})
                    """
                    # Execute the raw query
                    # The asterisk before `user_id_list` unpacks the list into separate arguments
                    response = await self.db.query_raw(sql_query)
                elif query_type == "find_all":
                    if expires is not None:
                        response = await self.db.litellm_usertable.find_many(  # type: ignore
                            order={"spend": "desc"},
                            where={  # type:ignore
                                "OR": [
                                    {"expires": None},  # type:ignore
                                    {"expires": {"gt": expires}},  # type:ignore
                                ],
                            },
                        )
                    else:
                        # return all users in the table, get their key aliases ordered by spend
                        sql_query = """
                        SELECT
                            u.*,
                            json_agg(v.key_alias) AS key_aliases
                        FROM
                            "LiteLLM_UserTable" u
                        LEFT JOIN "LiteLLM_VerificationToken" v ON u.user_id = v.user_id
                        GROUP BY
                            u.user_id
                        ORDER BY u.spend DESC
                        LIMIT $1
                        OFFSET $2
                        """
                        response = await self.db.query_raw(sql_query, limit, offset)
                return response
            elif table_name == "spend":
                verbose_proxy_logger.debug(
                    f"PrismaClient: get_data: table_name == 'spend'"
                )
                if key_val is not None:
                    if query_type == "find_unique":
                        response = await self.db.litellm_spendlogs.find_unique(  # type: ignore
                            where={  # type: ignore
                                key_val["key"]: key_val["value"],  # type: ignore
                            }
                        )
                    elif query_type == "find_all":
                        response = await self.db.litellm_spendlogs.find_many(  # type: ignore
                            where={
                                key_val["key"]: key_val["value"],  # type: ignore
                            }
                        )
                    return response
                else:
                    response = await self.db.litellm_spendlogs.find_many(  # type: ignore
                        order={"startTime": "desc"},
                    )
                    return response
            elif table_name == "team":
                if query_type == "find_unique":
                    response = await self.db.litellm_teamtable.find_unique(
                        where={"team_id": team_id}  # type: ignore
                    )
                elif query_type == "find_all" and user_id is not None:
                    response = await self.db.litellm_teamtable.find_many(
                        where={
                            "members": {"has": user_id},
                        },
                    )
                elif query_type == "find_all" and team_id_list is not None:
                    response = await self.db.litellm_teamtable.find_many(
                        where={"team_id": {"in": team_id_list}}
                    )
                elif query_type == "find_all" and team_id_list is None:
                    response = await self.db.litellm_teamtable.find_many(take=20)
                return response
            elif table_name == "user_notification":
                if query_type == "find_unique":
                    response = await self.db.litellm_usernotifications.find_unique(  # type: ignore
                        where={"user_id": user_id}  # type: ignore
                    )
                elif query_type == "find_all":
                    response = await self.db.litellm_usernotifications.find_many()  # type: ignore
                return response
            elif table_name == "combined_view":
                # check if plain text or hash
                if token is not None:
                    if isinstance(token, str):
                        hashed_token = token
                        if token.startswith("sk-"):
                            hashed_token = self.hash_token(token=token)
                        verbose_proxy_logger.debug(
                            f"PrismaClient: find_unique for token: {hashed_token}"
                        )
                if query_type == "find_unique":
                    if token is None:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"No token passed in. Token={token}"},
                        )

                    sql_query = f"""
                    SELECT 
                    v.*,
                    t.spend AS team_spend, 
                    t.max_budget AS team_max_budget, 
                    t.tpm_limit AS team_tpm_limit,
                    t.rpm_limit AS team_rpm_limit,
                    t.models AS team_models,
                    t.blocked AS team_blocked,
                    t.team_alias AS team_alias,
                    m.aliases as team_model_aliases
                    FROM "LiteLLM_VerificationToken" AS v
                    LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
                    LEFT JOIN "LiteLLM_ModelTable" m ON t.model_id = m.id
                    WHERE v.token = '{token}'
                    """

                    response = await self.db.query_first(query=sql_query)

                    if response is not None:
                        if response["team_models"] is None:
                            response["team_models"] = []
                        if response["team_blocked"] is None:
                            response["team_blocked"] = False
                        response = LiteLLM_VerificationTokenView(**response)
                        # for prisma we need to cast the expires time to str
                        if response.expires is not None and isinstance(
                            response.expires, datetime
                        ):
                            response.expires = response.expires.isoformat()
                    return response
        except Exception as e:
            import traceback

            prisma_query_info = f"LiteLLM Prisma Client Exception: Error with `get_data`. Args passed in: {args_passed_in}"
            error_msg = prisma_query_info + str(e)
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            verbose_proxy_logger.debug(error_traceback)
            end_time = time.time()
            _duration = end_time - start_time

            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="get_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def insert_data(
        self,
        data: dict,
        table_name: Literal[
            "user", "key", "config", "spend", "team", "user_notification"
        ],
    ):
        """
        Add a key to the database. If it already exists, do nothing.
        """
        start_time = time.time()
        try:
            verbose_proxy_logger.debug("PrismaClient: insert_data: %s", data)
            if table_name == "key":
                token = data["token"]
                hashed_token = self.hash_token(token=token)
                db_data = self.jsonify_object(data=data)
                db_data["token"] = hashed_token
                print_verbose(
                    "PrismaClient: Before upsert into litellm_verificationtoken"
                )
                new_verification_token = await self.db.litellm_verificationtoken.upsert(  # type: ignore
                    where={
                        "token": hashed_token,
                    },
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Keys Table")
                return new_verification_token
            elif table_name == "user":
                db_data = self.jsonify_object(data=data)
                try:
                    new_user_row = await self.db.litellm_usertable.upsert(
                        where={"user_id": data["user_id"]},
                        data={
                            "create": {**db_data},  # type: ignore
                            "update": {},  # don't do anything if it already exists
                        },
                    )
                except Exception as e:
                    if (
                        "Foreign key constraint failed on the field: `LiteLLM_UserTable_organization_id_fkey (index)`"
                        in str(e)
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": f"Foreign Key Constraint failed. Organization ID={db_data['organization_id']} does not exist in LiteLLM_OrganizationTable. Create via `/organization/new`."
                            },
                        )
                    raise e
                verbose_proxy_logger.info("Data Inserted into User Table")
                return new_user_row
            elif table_name == "team":
                db_data = self.jsonify_object(data=data)
                if db_data.get("members_with_roles", None) is not None and isinstance(
                    db_data["members_with_roles"], list
                ):
                    db_data["members_with_roles"] = json.dumps(
                        db_data["members_with_roles"]
                    )
                new_team_row = await self.db.litellm_teamtable.upsert(
                    where={"team_id": data["team_id"]},
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Team Table")
                return new_team_row
            elif table_name == "config":
                """
                For each param,
                get the existing table values

                Add the new values

                Update DB
                """
                tasks = []
                for k, v in data.items():
                    updated_data = v
                    updated_data = json.dumps(updated_data)
                    updated_table_row = self.db.litellm_config.upsert(
                        where={"param_name": k},
                        data={
                            "create": {"param_name": k, "param_value": updated_data},
                            "update": {"param_value": updated_data},
                        },
                    )

                    tasks.append(updated_table_row)
                await asyncio.gather(*tasks)
                verbose_proxy_logger.info("Data Inserted into Config Table")
            elif table_name == "spend":
                db_data = self.jsonify_object(data=data)
                new_spend_row = await self.db.litellm_spendlogs.upsert(
                    where={"request_id": data["request_id"]},
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info("Data Inserted into Spend Table")
                return new_spend_row
            elif table_name == "user_notification":
                db_data = self.jsonify_object(data=data)
                new_user_notification_row = (
                    await self.db.litellm_usernotifications.upsert(  # type: ignore
                        where={"request_id": data["request_id"]},
                        data={
                            "create": {**db_data},  # type: ignore
                            "update": {},  # don't do anything if it already exists
                        },
                    )
                )
                verbose_proxy_logger.info("Data Inserted into Model Request Table")
                return new_user_notification_row

        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception in insert_data: {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="insert_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def update_data(
        self,
        token: Optional[str] = None,
        data: dict = {},
        data_list: Optional[List] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        query_type: Literal["update", "update_many"] = "update",
        table_name: Optional[Literal["user", "key", "config", "spend", "team"]] = None,
        update_key_values: Optional[dict] = None,
        update_key_values_custom_query: Optional[dict] = None,
    ):
        """
        Update existing data
        """
        verbose_proxy_logger.debug(
            f"PrismaClient: update_data, table_name: {table_name}"
        )
        start_time = time.time()
        try:
            db_data = self.jsonify_object(data=data)
            if update_key_values is not None:
                update_key_values = self.jsonify_object(data=update_key_values)
            if token is not None:
                print_verbose(f"token: {token}")
                # check if plain text or hash
                if token.startswith("sk-"):
                    token = self.hash_token(token=token)
                db_data["token"] = token
                response = await self.db.litellm_verificationtoken.update(
                    where={"token": token},  # type: ignore
                    data={**db_data},  # type: ignore
                )
                verbose_proxy_logger.debug(
                    "\033[91m"
                    + f"DB Token Table update succeeded {response}"
                    + "\033[0m"
                )
                _data: dict = {}
                if response is not None:
                    try:
                        _data = response.model_dump()  # type: ignore
                    except Exception as e:
                        _data = response.dict()
                return {"token": token, "data": _data}
            elif (
                user_id is not None
                or (table_name is not None and table_name == "user")
                and query_type == "update"
            ):
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                if user_id is None:
                    user_id = db_data["user_id"]
                if update_key_values is None:
                    if update_key_values_custom_query is not None:
                        update_key_values = update_key_values_custom_query
                    else:
                        update_key_values = db_data
                update_user_row = await self.db.litellm_usertable.upsert(
                    where={"user_id": user_id},  # type: ignore
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {
                            **update_key_values  # type: ignore
                        },  # just update user-specified values, if it already exists
                    },
                )
                verbose_proxy_logger.info(
                    "\033[91m"
                    + f"DB User Table - update succeeded {update_user_row}"
                    + "\033[0m"
                )
                return {"user_id": user_id, "data": update_user_row}
            elif (
                team_id is not None
                or (table_name is not None and table_name == "team")
                and query_type == "update"
            ):
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                if team_id is None:
                    team_id = db_data["team_id"]
                if update_key_values is None:
                    update_key_values = db_data
                if "team_id" not in db_data and team_id is not None:
                    db_data["team_id"] = team_id
                if "members_with_roles" in db_data and isinstance(
                    db_data["members_with_roles"], list
                ):
                    db_data["members_with_roles"] = json.dumps(
                        db_data["members_with_roles"]
                    )
                if "members_with_roles" in update_key_values and isinstance(
                    update_key_values["members_with_roles"], list
                ):
                    update_key_values["members_with_roles"] = json.dumps(
                        update_key_values["members_with_roles"]
                    )
                update_team_row = await self.db.litellm_teamtable.upsert(
                    where={"team_id": team_id},  # type: ignore
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {
                            **update_key_values  # type: ignore
                        },  # just update user-specified values, if it already exists
                    },
                )
                verbose_proxy_logger.info(
                    "\033[91m"
                    + f"DB Team Table - update succeeded {update_team_row}"
                    + "\033[0m"
                )
                return {"team_id": team_id, "data": update_team_row}
            elif (
                table_name is not None
                and table_name == "key"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for idx, t in enumerate(data_list):
                    # check if plain text or hash
                    if t.token.startswith("sk-"):  # type: ignore
                        t.token = self.hash_token(token=t.token)  # type: ignore
                    try:
                        data_json = self.jsonify_object(
                            data=t.model_dump(exclude_none=True)
                        )
                    except:
                        data_json = self.jsonify_object(data=t.dict(exclude_none=True))
                    batcher.litellm_verificationtoken.update(
                        where={"token": t.token},  # type: ignore
                        data={**data_json},  # type: ignore
                    )
                await batcher.commit()
                print_verbose(
                    "\033[91m" + f"DB Token Table update succeeded" + "\033[0m"
                )
            elif (
                table_name is not None
                and table_name == "user"
                and query_type == "update_many"
                and data_list is not None
                and isinstance(data_list, list)
            ):
                """
                Batch write update queries
                """
                batcher = self.db.batch_()
                for idx, user in enumerate(data_list):
                    try:
                        data_json = self.jsonify_object(data=user.model_dump())
                    except:
                        data_json = self.jsonify_object(data=user.dict())
                    batcher.litellm_usertable.upsert(
                        where={"user_id": user.user_id},  # type: ignore
                        data={
                            "create": {**data_json},  # type: ignore
                            "update": {
                                **data_json  # type: ignore
                            },  # just update user-specified values, if it already exists
                        },
                    )
                await batcher.commit()
                verbose_proxy_logger.info(
                    "\033[91m" + f"DB User Table Batch update succeeded" + "\033[0m"
                )
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception - update_data: {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="update_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def delete_data(
        self,
        tokens: Optional[List] = None,
        team_id_list: Optional[List] = None,
        table_name: Optional[Literal["user", "key", "config", "spend", "team"]] = None,
        user_id: Optional[str] = None,
    ):
        """
        Allow user to delete a key(s)

        Ensure user owns that key, unless admin.
        """
        start_time = time.time()
        try:
            if tokens is not None and isinstance(tokens, List):
                hashed_tokens = []
                for token in tokens:
                    if isinstance(token, str) and token.startswith("sk-"):
                        hashed_token = self.hash_token(token=token)
                    else:
                        hashed_token = token
                    hashed_tokens.append(hashed_token)
                filter_query: dict = {}
                if user_id is not None:
                    filter_query = {
                        "AND": [{"token": {"in": hashed_tokens}}, {"user_id": user_id}]
                    }
                else:
                    filter_query = {"token": {"in": hashed_tokens}}

                deleted_tokens = await self.db.litellm_verificationtoken.delete_many(
                    where=filter_query  # type: ignore
                )
                verbose_proxy_logger.debug("deleted_tokens: %s", deleted_tokens)
                return {"deleted_keys": deleted_tokens}
            elif (
                table_name == "team"
                and team_id_list is not None
                and isinstance(team_id_list, List)
            ):
                # admin only endpoint -> `/team/delete`
                await self.db.litellm_teamtable.delete_many(
                    where={"team_id": {"in": team_id_list}}
                )
                return {"deleted_teams": team_id_list}
            elif (
                table_name == "key"
                and team_id_list is not None
                and isinstance(team_id_list, List)
            ):
                # admin only endpoint -> `/team/delete`
                await self.db.litellm_verificationtoken.delete_many(
                    where={"team_id": {"in": team_id_list}}
                )
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception - delete_data: {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="delete_data",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def connect(self):
        start_time = time.time()
        try:
            verbose_proxy_logger.debug(
                "PrismaClient: connect() called Attempting to Connect to DB"
            )
            if self.db.is_connected() == False:
                verbose_proxy_logger.debug(
                    "PrismaClient: DB not connected, Attempting to Connect to DB"
                )
                await self.db.connect()
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception connect(): {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="connect",
                    traceback_str=error_traceback,
                )
            )
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def disconnect(self):
        start_time = time.time()
        try:
            await self.db.disconnect()
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception disconnect(): {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="disconnect",
                    traceback_str=error_traceback,
                )
            )
            raise e

    async def health_check(self):
        """
        Health check endpoint for the prisma client
        """
        start_time = time.time()
        try:
            sql_query = """
                SELECT 1
                FROM "LiteLLM_VerificationToken"
                LIMIT 1
                """

            # Execute the raw query
            # The asterisk before `user_id_list` unpacks the list into separate arguments
            response = await self.db.query_raw(sql_query)
            return response
        except Exception as e:
            import traceback

            error_msg = f"LiteLLM Prisma Client Exception disconnect(): {str(e)}"
            print_verbose(error_msg)
            error_traceback = error_msg + "\n" + traceback.format_exc()
            end_time = time.time()
            _duration = end_time - start_time
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(
                    original_exception=e,
                    duration=_duration,
                    call_type="health_check",
                    traceback_str=error_traceback,
                )
            )
            raise e


class DBClient:
    """
    Routes requests for CustomAuth

    [TODO] route b/w customauth and prisma
    """

    def __init__(
        self, custom_db_type: Literal["dynamo_db"], custom_db_args: dict
    ) -> None:
        if custom_db_type == "dynamo_db":
            from litellm.proxy.db.dynamo_db import DynamoDBWrapper

            self.db = DynamoDBWrapper(database_arguments=DynamoDBArgs(**custom_db_args))

    async def get_data(self, key: str, table_name: Literal["user", "key", "config"]):
        """
        Check if key valid
        """
        return await self.db.get_data(key=key, table_name=table_name)

    async def insert_data(
        self, value: Any, table_name: Literal["user", "key", "config"]
    ):
        """
        For new key / user logic
        """
        return await self.db.insert_data(value=value, table_name=table_name)

    async def update_data(
        self, key: str, value: Any, table_name: Literal["user", "key", "config"]
    ):
        """
        For cost tracking logic

        key - hash_key value \n
        value - dict with updated values
        """
        return await self.db.update_data(key=key, value=value, table_name=table_name)

    async def delete_data(
        self, keys: List[str], table_name: Literal["user", "key", "config"]
    ):
        """
        For /key/delete endpoints
        """
        return await self.db.delete_data(keys=keys, table_name=table_name)

    async def connect(self):
        """
        For connecting to db and creating / updating any tables
        """
        return await self.db.connect()

    async def disconnect(self):
        """
        For closing connection on server shutdown
        """
        return await self.db.disconnect()


### CUSTOM FILE ###
def get_instance_fn(value: str, config_file_path: Optional[str] = None) -> Any:
    try:
        print_verbose(f"value: {value}")
        # Split the path by dots to separate module from instance
        parts = value.split(".")

        # The module path is all but the last part, and the instance_name is the last part
        module_name = ".".join(parts[:-1])
        instance_name = parts[-1]

        # If config_file_path is provided, use it to determine the module spec and load the module
        if config_file_path is not None:
            directory = os.path.dirname(config_file_path)
            module_file_path = os.path.join(directory, *module_name.split("."))
            module_file_path += ".py"

            spec = importlib.util.spec_from_file_location(module_name, module_file_path)
            if spec is None:
                raise ImportError(
                    f"Could not find a module specification for {module_file_path}"
                )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
        else:
            # Dynamically import the module
            module = importlib.import_module(module_name)

        # Get the instance from the module
        instance = getattr(module, instance_name)

        return instance
    except ImportError as e:
        # Re-raise the exception with a user-friendly message
        raise ImportError(f"Could not import {instance_name} from {module_name}") from e
    except Exception as e:
        raise e


### HELPER FUNCTIONS ###
async def _cache_user_row(
    user_id: str, cache: DualCache, db: Union[PrismaClient, DBClient]
):
    """
    Check if a user_id exists in cache,
    if not retrieve it.
    """
    cache_key = f"{user_id}_user_api_key_user_id"
    response = cache.get_cache(key=cache_key)
    if response is None:  # Cache miss
        if isinstance(db, PrismaClient):
            user_row = await db.get_data(user_id=user_id)
        elif isinstance(db, DBClient):
            user_row = await db.get_data(key=user_id, table_name="user")
        if user_row is not None:
            print_verbose(f"User Row: {user_row}, type = {type(user_row)}")
            if hasattr(user_row, "model_dump_json") and callable(
                getattr(user_row, "model_dump_json")
            ):
                cache_value = user_row.model_dump_json()
                cache.set_cache(
                    key=cache_key, value=cache_value, ttl=600
                )  # store for 10 minutes
    return


async def send_email(sender_name, sender_email, receiver_email, subject, html):
    """
    smtp_host,
    smtp_port,
    smtp_username,
    smtp_password,
    sender_name,
    sender_email,
    """
    ## SERVER SETUP ##
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT", 587)  # default to port 587
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    ## EMAIL SETUP ##
    email_message = MIMEMultipart()
    email_message["From"] = f"{sender_name} <{sender_email}>"
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    # Attach the body to the email
    email_message.attach(MIMEText(html, "html"))

    try:
        print_verbose(f"SMTP Connection Init")
        # Establish a secure connection with the SMTP server
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if os.getenv("SMTP_TLS", "True") != "False":
                server.starttls()

            # Login to your email account
            server.login(smtp_username, smtp_password)

            # Send the email
            server.send_message(email_message)

    except Exception as e:
        print_verbose("An error occurred while sending the email:" + str(e))


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


def get_logging_payload(kwargs, response_obj, start_time, end_time):
    from litellm.proxy._types import LiteLLM_SpendLogs
    from pydantic import Json
    import uuid

    verbose_proxy_logger.debug(
        f"SpendTable: get_logging_payload - kwargs: {kwargs}\n\n"
    )

    if kwargs == None:
        kwargs = {}
    # standardize this function to be used across, s3, dynamoDB, langfuse logging
    litellm_params = kwargs.get("litellm_params", {})
    metadata = (
        litellm_params.get("metadata", {}) or {}
    )  # if litellm_params['metadata'] == None
    call_type = kwargs.get("call_type")
    cache_hit = kwargs.get("cache_hit", False)
    usage = response_obj["usage"]
    if type(usage) == litellm.Usage:
        usage = dict(usage)
    id = response_obj.get("id", kwargs.get("litellm_call_id"))
    api_key = metadata.get("user_api_key", "")
    if api_key is not None and isinstance(api_key, str) and api_key.startswith("sk-"):
        # hash the api_key
        api_key = hash_token(api_key)

    # clean up litellm metadata
    if isinstance(metadata, dict):
        clean_metadata = {}
        verbose_proxy_logger.debug(
            f"getting payload for SpendLogs, available keys in metadata: "
            + str(list(metadata.keys()))
        )
        for key in metadata:
            if key in [
                "headers",
                "endpoint",
                "model_group",
                "deployment",
                "model_info",
                "caching_groups",
                "previous_models",
            ]:
                continue
            else:
                clean_metadata[key] = metadata[key]

    if litellm.cache is not None:
        cache_key = litellm.cache.get_cache_key(**kwargs)
    else:
        cache_key = "Cache OFF"
    if cache_hit == True:
        import time

        id = f"{id}_cache_hit{time.time()}"  # SpendLogs does not allow duplicate request_id

    payload = {
        "request_id": id,
        "call_type": call_type,
        "api_key": api_key,
        "cache_hit": cache_hit,
        "startTime": start_time,
        "endTime": end_time,
        "model": kwargs.get("model", ""),
        "user": kwargs.get("litellm_params", {})
        .get("metadata", {})
        .get("user_api_key_user_id", ""),
        "team_id": kwargs.get("litellm_params", {})
        .get("metadata", {})
        .get("user_api_key_team_id", ""),
        "metadata": clean_metadata,
        "cache_key": cache_key,
        "spend": kwargs.get("response_cost", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "request_tags": metadata.get("tags", []),
        "end_user": kwargs.get("user", ""),
        "api_base": litellm_params.get("api_base", ""),
    }

    verbose_proxy_logger.debug("SpendTable: created payload - payload: %s\n\n", payload)
    json_fields = [
        field
        for field, field_type in LiteLLM_SpendLogs.__annotations__.items()
        if field_type == Json or field_type == Optional[Json]
    ]
    str_fields = [
        field
        for field, field_type in LiteLLM_SpendLogs.__annotations__.items()
        if field_type == str or field_type == Optional[str]
    ]
    datetime_fields = [
        field
        for field, field_type in LiteLLM_SpendLogs.__annotations__.items()
        if field_type == datetime
    ]

    for param in json_fields:
        if param in payload and type(payload[param]) != Json:
            if type(payload[param]) == litellm.ModelResponse:
                payload[param] = payload[param].model_dump_json()
            if type(payload[param]) == litellm.EmbeddingResponse:
                payload[param] = payload[param].model_dump_json()
            else:
                payload[param] = json.dumps(payload[param])

    for param in str_fields:
        if param in payload and type(payload[param]) != str:
            payload[param] = str(payload[param])

    return payload


def _duration_in_seconds(duration: str):
    match = re.match(r"(\d+)([smhd]?)", duration)
    if not match:
        raise ValueError("Invalid duration format")

    value, unit = match.groups()
    value = int(value)

    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    else:
        raise ValueError("Unsupported duration unit")


async def reset_budget(prisma_client: PrismaClient):
    """
    Gets all the non-expired keys for a db, which need spend to be reset

    Resets their spend

    Updates db
    """
    if prisma_client is not None:
        ### RESET KEY BUDGET ###
        now = datetime.utcnow()
        keys_to_reset = await prisma_client.get_data(
            table_name="key", query_type="find_all", expires=now, reset_at=now
        )

        if keys_to_reset is not None and len(keys_to_reset) > 0:
            for key in keys_to_reset:
                key.spend = 0.0
                duration_s = _duration_in_seconds(duration=key.budget_duration)
                key.budget_reset_at = now + timedelta(seconds=duration_s)

            await prisma_client.update_data(
                query_type="update_many", data_list=keys_to_reset, table_name="key"
            )

        ### RESET USER BUDGET ###
        now = datetime.utcnow()
        users_to_reset = await prisma_client.get_data(
            table_name="user", query_type="find_all", reset_at=now
        )

        if users_to_reset is not None and len(users_to_reset) > 0:
            for user in users_to_reset:
                user.spend = 0.0
                duration_s = _duration_in_seconds(duration=user.budget_duration)
                user.budget_reset_at = now + timedelta(seconds=duration_s)

            await prisma_client.update_data(
                query_type="update_many", data_list=users_to_reset, table_name="user"
            )


async def update_spend(
    prisma_client: PrismaClient,
    db_writer_client: Optional[HTTPHandler],
    proxy_logging_obj: ProxyLogging,
):
    """
    Batch write updates to db.

    Triggered every minute.

    Requires:
    user_id_list: dict,
    keys_list: list,
    team_list: list,
    spend_logs: list,
    """
    n_retry_times = 3
    ### UPDATE USER TABLE ###
    if len(prisma_client.user_list_transactons.keys()) > 0:
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            user_id,
                            response_cost,
                        ) in prisma_client.user_list_transactons.items():
                            batcher.litellm_usertable.update_many(
                                where={"user_id": user_id},
                                data={"spend": {"increment": response_cost}},
                            )
                prisma_client.user_list_transactons = (
                    {}
                )  # Clear the remaining transactions after processing all batches in the loop.
                break
            except httpx.ReadTimeout:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    raise  # Re-raise the last exception
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                import traceback

                error_msg = (
                    f"LiteLLM Prisma Client Exception - update user spend: {str(e)}"
                )
                print_verbose(error_msg)
                error_traceback = error_msg + "\n" + traceback.format_exc()
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    proxy_logging_obj.failure_handler(
                        original_exception=e,
                        duration=_duration,
                        call_type="update_spend",
                        traceback_str=error_traceback,
                    )
                )
                raise e

    ### UPDATE END-USER TABLE ###
    if len(prisma_client.end_user_list_transactons.keys()) > 0:
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            end_user_id,
                            response_cost,
                        ) in prisma_client.end_user_list_transactons.items():
                            max_end_user_budget = None
                            if litellm.max_end_user_budget is not None:
                                max_end_user_budget = litellm.max_end_user_budget
                            new_user_obj = LiteLLM_EndUserTable(
                                user_id=end_user_id, spend=response_cost, blocked=False
                            )
                            batcher.litellm_endusertable.update_many(
                                where={"user_id": end_user_id},
                                data={"spend": {"increment": response_cost}},
                            )
                prisma_client.end_user_list_transactons = (
                    {}
                )  # Clear the remaining transactions after processing all batches in the loop.
                break
            except httpx.ReadTimeout:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    raise  # Re-raise the last exception
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                import traceback

                error_msg = (
                    f"LiteLLM Prisma Client Exception - update end-user spend: {str(e)}"
                )
                print_verbose(error_msg)
                error_traceback = error_msg + "\n" + traceback.format_exc()
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    proxy_logging_obj.failure_handler(
                        original_exception=e,
                        duration=_duration,
                        call_type="update_spend",
                        traceback_str=error_traceback,
                    )
                )
                raise e

    ### UPDATE KEY TABLE ###
    verbose_proxy_logger.debug(
        "KEY Spend transactions: {}".format(
            len(prisma_client.key_list_transactons.keys())
        )
    )
    if len(prisma_client.key_list_transactons.keys()) > 0:
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            token,
                            response_cost,
                        ) in prisma_client.key_list_transactons.items():
                            batcher.litellm_verificationtoken.update_many(  # 'update_many' prevents error from being raised if no row exists
                                where={"token": token},
                                data={"spend": {"increment": response_cost}},
                            )
                prisma_client.key_list_transactons = (
                    {}
                )  # Clear the remaining transactions after processing all batches in the loop.
                break
            except httpx.ReadTimeout:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    raise  # Re-raise the last exception
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                import traceback

                error_msg = (
                    f"LiteLLM Prisma Client Exception - update key spend: {str(e)}"
                )
                print_verbose(error_msg)
                error_traceback = error_msg + "\n" + traceback.format_exc()
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    proxy_logging_obj.failure_handler(
                        original_exception=e,
                        duration=_duration,
                        call_type="update_spend",
                        traceback_str=error_traceback,
                    )
                )
                raise e

    ### UPDATE TEAM TABLE ###
    verbose_proxy_logger.debug(
        "Team Spend transactions: {}".format(
            len(prisma_client.team_list_transactons.keys())
        )
    )
    if len(prisma_client.team_list_transactons.keys()) > 0:
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            team_id,
                            response_cost,
                        ) in prisma_client.team_list_transactons.items():
                            verbose_proxy_logger.debug(
                                "Updating spend for team id={} by {}".format(
                                    team_id, response_cost
                                )
                            )
                            batcher.litellm_teamtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                where={"team_id": team_id},
                                data={"spend": {"increment": response_cost}},
                            )
                prisma_client.team_list_transactons = (
                    {}
                )  # Clear the remaining transactions after processing all batches in the loop.
                break
            except httpx.ReadTimeout:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    raise  # Re-raise the last exception
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                import traceback

                error_msg = (
                    f"LiteLLM Prisma Client Exception - update team spend: {str(e)}"
                )
                print_verbose(error_msg)
                error_traceback = error_msg + "\n" + traceback.format_exc()
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    proxy_logging_obj.failure_handler(
                        original_exception=e,
                        duration=_duration,
                        call_type="update_spend",
                        traceback_str=error_traceback,
                    )
                )
                raise e

    ### UPDATE ORG TABLE ###
    if len(prisma_client.org_list_transactons.keys()) > 0:
        for i in range(n_retry_times + 1):
            start_time = time.time()
            try:
                async with prisma_client.db.tx(
                    timeout=timedelta(seconds=60)
                ) as transaction:
                    async with transaction.batch_() as batcher:
                        for (
                            org_id,
                            response_cost,
                        ) in prisma_client.org_list_transactons.items():
                            batcher.litellm_organizationtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                where={"organization_id": org_id},
                                data={"spend": {"increment": response_cost}},
                            )
                prisma_client.org_list_transactons = (
                    {}
                )  # Clear the remaining transactions after processing all batches in the loop.
                break
            except httpx.ReadTimeout:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    raise  # Re-raise the last exception
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                import traceback

                error_msg = (
                    f"LiteLLM Prisma Client Exception - update org spend: {str(e)}"
                )
                print_verbose(error_msg)
                error_traceback = error_msg + "\n" + traceback.format_exc()
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    proxy_logging_obj.failure_handler(
                        original_exception=e,
                        duration=_duration,
                        call_type="update_spend",
                        traceback_str=error_traceback,
                    )
                )
                raise e

    ### UPDATE SPEND LOGS ###
    verbose_proxy_logger.debug(
        "Spend Logs transactions: {}".format(len(prisma_client.spend_log_transactions))
    )

    BATCH_SIZE = 100  # Preferred size of each batch to write to the database
    MAX_LOGS_PER_INTERVAL = 1000  # Maximum number of logs to flush in a single interval

    if len(prisma_client.spend_log_transactions) > 0:
        for _ in range(n_retry_times + 1):
            start_time = time.time()
            try:
                base_url = os.getenv("SPEND_LOGS_URL", None)
                ## WRITE TO SEPARATE SERVER ##
                if (
                    len(prisma_client.spend_log_transactions) > 0
                    and base_url is not None
                    and db_writer_client is not None
                ):
                    if not base_url.endswith("/"):
                        base_url += "/"
                    verbose_proxy_logger.debug("base_url: {}".format(base_url))
                    response = await db_writer_client.post(
                        url=base_url + "spend/update",
                        data=json.dumps(prisma_client.spend_log_transactions),  # type: ignore
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code == 200:
                        prisma_client.spend_log_transactions = []
                else:  ## (default) WRITE TO DB ##
                    logs_to_process = prisma_client.spend_log_transactions[
                        :MAX_LOGS_PER_INTERVAL
                    ]
                    for i in range(0, len(logs_to_process), BATCH_SIZE):
                        # Create sublist for current batch, ensuring it doesn't exceed the BATCH_SIZE
                        batch = logs_to_process[i : i + BATCH_SIZE]

                        # Convert datetime strings to Date objects
                        batch_with_dates = [
                            prisma_client.jsonify_object(
                                {
                                    **entry,
                                }
                            )
                            for entry in batch
                        ]

                        await prisma_client.db.litellm_spendlogs.create_many(
                            data=batch_with_dates, skip_duplicates=True  # type: ignore
                        )

                        verbose_proxy_logger.debug(
                            f"Flushed {len(batch)} logs to the DB."
                        )
                    # Remove the processed logs from spend_logs
                    prisma_client.spend_log_transactions = (
                        prisma_client.spend_log_transactions[len(logs_to_process) :]
                    )

                    verbose_proxy_logger.debug(
                        f"{len(logs_to_process)} logs processed. Remaining in queue: {len(prisma_client.spend_log_transactions)}"
                    )
                break
            except httpx.ReadTimeout:
                if i >= n_retry_times:  # If we've reached the maximum number of retries
                    raise  # Re-raise the last exception
                # Optionally, sleep for a bit before retrying
                await asyncio.sleep(2**i)  # Exponential backoff
            except Exception as e:
                import traceback

                error_msg = (
                    f"LiteLLM Prisma Client Exception - update spend logs: {str(e)}"
                )
                print_verbose(error_msg)
                error_traceback = error_msg + "\n" + traceback.format_exc()
                end_time = time.time()
                _duration = end_time - start_time
                asyncio.create_task(
                    proxy_logging_obj.failure_handler(
                        original_exception=e,
                        duration=_duration,
                        call_type="update_spend",
                        traceback_str=error_traceback,
                    )
                )
                raise e


async def _read_request_body(request):
    """
    Asynchronous function to read the request body and parse it as JSON or literal data.

    Parameters:
    - request: The request object to read the body from

    Returns:
    - dict: Parsed request data as a dictionary
    """
    import ast, json

    try:
        request_data = {}
        if request is None:
            return request_data
        body = await request.body()

        if body == b"" or body is None:
            return request_data
        body_str = body.decode()
        try:
            request_data = ast.literal_eval(body_str)
        except:
            request_data = json.loads(body_str)
        return request_data
    except:
        return {}


def _is_projected_spend_over_limit(
    current_spend: float, soft_budget_limit: Optional[float]
):
    from datetime import date

    if soft_budget_limit is None:
        # If there's no limit, we can't exceed it.
        return False

    today = date.today()

    # Finding the first day of the next month, then subtracting one day to get the end of the current month.
    if today.month == 12:  # December edge case
        end_month = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        end_month = date(today.year, today.month + 1, 1) - timedelta(days=1)

    remaining_days = (end_month - today).days

    # Check for the start of the month to avoid division by zero
    if today.day == 1:
        daily_spend_estimate = current_spend
    else:
        daily_spend_estimate = current_spend / (today.day - 1)

    # Total projected spend for the month
    projected_spend = current_spend + (daily_spend_estimate * remaining_days)

    if projected_spend > soft_budget_limit:
        print_verbose("Projected spend exceeds soft budget limit!")
        return True
    return False


def _get_projected_spend_over_limit(
    current_spend: float, soft_budget_limit: Optional[float]
) -> Optional[tuple]:
    import datetime

    if soft_budget_limit is None:
        return None

    today = datetime.date.today()
    end_month = datetime.date(today.year, today.month + 1, 1) - datetime.timedelta(
        days=1
    )
    remaining_days = (end_month - today).days

    daily_spend = current_spend / (
        today.day - 1
    )  # assuming the current spend till today (not including today)
    projected_spend = daily_spend * remaining_days

    if projected_spend > soft_budget_limit:
        approx_days = soft_budget_limit / daily_spend
        limit_exceed_date = today + datetime.timedelta(days=approx_days)

        # return the projected spend and the date it will exceeded
        return projected_spend, limit_exceed_date

    return None


def _is_valid_team_configs(team_id=None, team_config=None, request_data=None):
    if team_id is None or team_config is None or request_data is None:
        return
    # check if valid model called for team
    if "models" in team_config:
        valid_models = team_config.pop("models")
        model_in_request = request_data["model"]
        if model_in_request not in valid_models:
            raise Exception(
                f"Invalid model for team {team_id}: {model_in_request}.  Valid models for team are: {valid_models}\n"
            )
    return


def _is_user_proxy_admin(user_id_information=None):
    if (
        user_id_information == None
        or len(user_id_information) == 0
        or user_id_information[0] == None
    ):
        return False
    _user = user_id_information[0]
    if (
        _user.get("user_role", None) is not None
        and _user.get("user_role") == "proxy_admin"
    ):
        return True

    # if user_id_information contains litellm-proxy-budget
    # get first user_id that is not litellm-proxy-budget
    for user in user_id_information:
        if user.get("user_id") != "litellm-proxy-budget":
            _user = user
            break

    if (
        _user.get("user_role", None) is not None
        and _user.get("user_role") == "proxy_admin"
    ):
        return True

    return False


def encrypt_value(value: str, master_key: str):
    import hashlib
    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(master_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # encode message #
    value_bytes = value.encode("utf-8")

    encrypted = box.encrypt(value_bytes)

    return encrypted


def decrypt_value(value: bytes, master_key: str) -> str:
    import hashlib
    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(master_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # Convert the bytes object to a string
    plaintext = box.decrypt(value)

    plaintext = plaintext.decode("utf-8")  # type: ignore
    return plaintext  # type: ignore


# LiteLLM Admin UI - Non SSO Login
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>LiteLLM Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        label {
            display: block;
            margin-bottom: 8px;
        }

        input {
            width: 100%;
            padding: 8px;
            margin-bottom: 16px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        input[type="submit"] {
            background-color: #4caf50;
            color: #fff;
            cursor: pointer;
        }

        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <form action="/login" method="post">
        <h2>LiteLLM Login</h2>

        <p>By default Username is "admin" and Password is your set LiteLLM Proxy `MASTER_KEY`</p>
        <p>If you need to set UI credentials / SSO docs here: <a href="https://docs.litellm.ai/docs/proxy/ui" target="_blank">https://docs.litellm.ai/docs/proxy/ui</a></p>
        <br>
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required>
        <input type="submit" value="Submit">
    </form>
</body>
</html>
"""
