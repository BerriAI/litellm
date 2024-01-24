from typing import Optional, List, Any, Literal, Union
import os, subprocess, hashlib, importlib, asyncio, copy, json, aiohttp, httpx
import litellm, backoff
from litellm.proxy._types import (
    UserAPIKeyAuth,
    DynamoDBArgs,
    LiteLLM_VerificationToken,
    LiteLLM_SpendLogs,
)
from litellm.caching import DualCache
from litellm.proxy.hooks.parallel_request_limiter import MaxParallelRequestsHandler
from litellm.proxy.hooks.max_budget_limiter import MaxBudgetLimiter
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.db.base_client import CustomDB
from litellm._logging import verbose_proxy_logger
from fastapi import HTTPException, status
import smtplib, re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta


def print_verbose(print_statement):
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

    def __init__(self, user_api_key_cache: DualCache):
        ## INITIALIZE  LITELLM CALLBACKS ##
        self.call_details: dict = {}
        self.call_details["user_api_key_cache"] = user_api_key_cache
        self.max_parallel_request_limiter = MaxParallelRequestsHandler()
        self.max_budget_limiter = MaxBudgetLimiter()
        self.alerting: Optional[List] = None
        self.alerting_threshold: float = 300  # default to 5 min. threshold
        pass

    def update_values(
        self, alerting: Optional[List], alerting_threshold: Optional[float]
    ):
        self.alerting = alerting
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold

    def _init_litellm_callbacks(self):
        print_verbose(f"INITIALIZING LITELLM CALLBACKS!")
        litellm.callbacks.append(self.max_parallel_request_limiter)
        litellm.callbacks.append(self.max_budget_limiter)
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
        call_type: Literal["completion", "embeddings"],
    ):
        """
        Allows users to modify/reject the incoming request to the proxy, without having to deal with parsing Request body.

        Covers:
        1. /chat/completions
        2. /embeddings
        3. /image/generation
        """
        ### ALERTING ###
        asyncio.create_task(self.response_taking_too_long())

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
            raise e

    async def success_handler(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        call_type: Literal["completion", "embeddings"],
        start_time,
        end_time,
    ):
        """
        Log successful API calls / db read/writes
        """

        pass

    async def response_taking_too_long(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        type: Literal["hanging_request", "slow_response"] = "hanging_request",
    ):
        if type == "hanging_request":
            # Simulate a long-running operation that could take more than 5 minutes
            await asyncio.sleep(
                self.alerting_threshold
            )  # Set it to 5 minutes - i'd imagine this might be different for streaming, non-streaming, non-completion (embedding + img) requests

            await self.alerting_handler(
                message=f"Requests are hanging - {self.alerting_threshold}s+ request time",
                level="Medium",
            )

        elif (
            type == "slow_response" and start_time is not None and end_time is not None
        ):
            if end_time - start_time > self.alerting_threshold:
                await self.alerting_handler(
                    message=f"Responses are slow - {round(end_time-start_time,2)}s response time",
                    level="Low",
                )

    async def alerting_handler(
        self, message: str, level: Literal["Low", "Medium", "High"]
    ):
        """
        Alerting based on thresholds: - https://github.com/BerriAI/litellm/issues/1298

        - Responses taking too long
        - Requests are hanging
        - Calls are failing
        - DB Read/Writes are failing

        Parameters:
            level: str - Low|Medium|High - if calls might fail (Medium) or are failing (High); Currently, no alerts would be 'Low'.
            message: str - what is the alert about
        """
        formatted_message = f"Level: {level}\n\nMessage: {message}"
        if self.alerting is None:
            return

        for client in self.alerting:
            if client == "slack":
                slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", None)
                if slack_webhook_url is None:
                    raise Exception("Missing SLACK_WEBHOOK_URL from environment")
                payload = {"text": formatted_message}
                headers = {"Content-type": "application/json"}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        slack_webhook_url, json=payload, headers=headers
                    ) as response:
                        if response.status == 200:
                            pass
            elif client == "sentry":
                if litellm.utils.sentry_sdk_instance is not None:
                    litellm.utils.sentry_sdk_instance.capture_message(formatted_message)
                else:
                    raise Exception("Missing SENTRY_DSN from environment")

    async def failure_handler(self, original_exception):
        """
        Log failed db read/writes

        Currently only logs exceptions to sentry
        """
        ### ALERTING ###
        if isinstance(original_exception, HTTPException):
            error_message = original_exception.detail
        else:
            error_message = str(original_exception)
        asyncio.create_task(
            self.alerting_handler(
                message=f"DB read/write call failed: {error_message}",
                level="High",
            )
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
        asyncio.create_task(
            self.alerting_handler(
                message=f"LLM API call failed: {str(original_exception)}", level="High"
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


### DB CONNECTOR ###
# Define the retry decorator with backoff strategy
# Function to be called whenever a retry is about to happen
def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    print_verbose(f"Backing off... this was attempt #{details['tries']}")


class PrismaClient:
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

        self.db = Prisma(
            http={
                "limits": httpx.Limits(
                    max_connections=1000, max_keepalive_connections=100
                )
            }
        )  # Client to connect to Prisma db

    def hash_token(self, token: str):
        # Hash the string using SHA-256
        hashed_token = hashlib.sha256(token.encode()).hexdigest()

        return hashed_token

    def jsonify_object(self, data: dict) -> dict:
        db_data = copy.deepcopy(data)

        for k, v in db_data.items():
            if isinstance(v, dict):
                db_data[k] = json.dumps(v)
        return db_data

    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
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
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
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
        token: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        table_name: Optional[Literal["user", "key", "config", "spend"]] = None,
        query_type: Literal["find_unique", "find_all"] = "find_unique",
        expires: Optional[datetime] = None,
        reset_at: Optional[datetime] = None,
    ):
        try:
            print_verbose("PrismaClient: get_data")

            response: Any = None
            if token is not None or (table_name is not None and table_name == "key"):
                # check if plain text or hash
                if token is not None:
                    hashed_token = token
                    if token.startswith("sk-"):
                        hashed_token = self.hash_token(token=token)
                print_verbose("PrismaClient: find_unique")
                if query_type == "find_unique":
                    response = await self.db.litellm_verificationtoken.find_unique(
                        where={"token": hashed_token}
                    )
                    if response is not None:
                        # for prisma we need to cast the expires time to str
                        if isinstance(response.expires, datetime):
                            response.expires = response.expires.isoformat()
                elif query_type == "find_all" and user_id is not None:
                    response = await self.db.litellm_verificationtoken.find_many(
                        where={"user_id": user_id}
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
                    response = await self.db.litellm_verificationtoken.find_many(
                        order={"spend": "desc"},
                    )
                print_verbose(f"PrismaClient: response={response}")
                if response is not None:
                    return response
                else:
                    # Token does not exist.
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication Error: invalid user key - token does not exist",
                    )
            elif user_id is not None:
                response = await self.db.litellm_usertable.find_unique(  # type: ignore
                    where={
                        "user_id": user_id,
                    }
                )
                return response
            elif table_name == "spend":
                verbose_proxy_logger.debug(
                    f"PrismaClient: get_data: table_name == 'spend'"
                )
                if request_id is not None:
                    response = await self.db.litellm_spendlogs.find_unique(  # type: ignore
                        where={
                            "request_id": request_id,
                        }
                    )
                    return response
                else:
                    response = await self.db.litellm_spendlogs.find_many(  # type: ignore
                        order={"startTime": "desc"},
                    )
                    return response

        except Exception as e:
            print_verbose(f"LiteLLM Prisma Client Exception: {e}")
            import traceback

            traceback.print_exc()
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
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
        self, data: dict, table_name: Literal["user", "key", "config", "spend"]
    ):
        """
        Add a key to the database. If it already exists, do nothing.
        """
        try:
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
                verbose_proxy_logger.info(f"Data Inserted into Keys Table")
                return new_verification_token
            elif table_name == "user":
                db_data = self.jsonify_object(data=data)
                new_user_row = await self.db.litellm_usertable.upsert(
                    where={"user_id": data["user_id"]},
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info(f"Data Inserted into User Table")
                return new_user_row
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
                verbose_proxy_logger.info(f"Data Inserted into Config Table")
            elif table_name == "spend":
                db_data = self.jsonify_object(data=data)
                new_spend_row = await self.db.litellm_spendlogs.upsert(
                    where={"request_id": data["request_id"]},
                    data={
                        "create": {**db_data},  # type: ignore
                        "update": {},  # don't do anything if it already exists
                    },
                )
                verbose_proxy_logger.info(f"Data Inserted into Spend Table")
                return new_spend_row

        except Exception as e:
            print_verbose(f"LiteLLM Prisma Client Exception: {e}")
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
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
        query_type: Literal["update", "update_many"] = "update",
        table_name: Optional[Literal["user", "key", "config", "spend"]] = None,
    ):
        """
        Update existing data
        """
        try:
            db_data = self.jsonify_object(data=data)
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
                return {"token": token, "data": db_data}
            elif user_id is not None:
                """
                If data['spend'] + data['user'], update the user table with spend info as well
                """
                update_user_row = await self.db.litellm_usertable.update(
                    where={"user_id": user_id},  # type: ignore
                    data={**db_data},  # type: ignore
                )
                if update_user_row is None:
                    # if the provided user does not exist, STILL Track this!
                    # make a new user with {"user_id": user_id, "spend": data['spend']}

                    db_data["user_id"] = user_id
                    update_user_row = await self.db.litellm_usertable.upsert(
                        where={"user_id": user_id},  # type: ignore
                        data={
                            "create": {**db_data},  # type: ignore
                            "update": {},  # don't do anything if it already exists
                        },
                    )
                print_verbose(
                    "\033[91m"
                    + f"DB User Table - update succeeded {update_user_row}"
                    + "\033[0m"
                )
                return {"user_id": user_id, "data": db_data}
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
                        data_json = self.jsonify_object(data=t.model_dump())
                    except:
                        data_json = self.jsonify_object(data=t.dict())
                    batcher.litellm_verificationtoken.update(
                        where={"token": t.token},  # type: ignore
                        data={**data_json},  # type: ignore
                    )
                await batcher.commit()
                print_verbose(
                    "\033[91m" + f"DB Token Table update succeeded" + "\033[0m"
                )
        except Exception as e:
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
            )
            print_verbose("\033[91m" + f"DB write failed: {e}" + "\033[0m")
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,  # base exception to catch for the backoff
        max_tries=3,  # maximum number of retries
        max_time=10,  # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def delete_data(self, tokens: List):
        """
        Allow user to delete a key(s)
        """
        try:
            hashed_tokens = [self.hash_token(token=token) for token in tokens]
            await self.db.litellm_verificationtoken.delete_many(
                where={"token": {"in": hashed_tokens}}
            )
            return {"deleted_keys": tokens}
        except Exception as e:
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
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
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
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
        try:
            await self.db.disconnect()
        except Exception as e:
            asyncio.create_task(
                self.proxy_logging_obj.failure_handler(original_exception=e)
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
    print_verbose(f"Prisma: _cache_user_row, user_id: {user_id}")
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
            server.starttls()

            # Login to your email account
            server.login(smtp_username, smtp_password)

            # Send the email
            server.send_message(email_message)

    except Exception as e:
        print_verbose("An error occurred while sending the email:", str(e))


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


def get_logging_payload(kwargs, response_obj, start_time, end_time):
    from litellm.proxy._types import LiteLLM_SpendLogs
    from pydantic import Json
    import uuid

    if kwargs == None:
        kwargs = {}
    # standardize this function to be used across, s3, dynamoDB, langfuse logging
    litellm_params = kwargs.get("litellm_params", {})
    metadata = (
        litellm_params.get("metadata", {}) or {}
    )  # if litellm_params['metadata'] == None
    messages = kwargs.get("messages")
    optional_params = kwargs.get("optional_params", {})
    call_type = kwargs.get("call_type", "litellm.completion")
    cache_hit = kwargs.get("cache_hit", False)
    usage = response_obj["usage"]
    id = response_obj.get("id", str(uuid.uuid4()))
    api_key = metadata.get("user_api_key", "")
    if api_key is not None and isinstance(api_key, str) and api_key.startswith("sk-"):
        # hash the api_key
        api_key = hash_token(api_key)

    if "headers" in metadata and "authorization" in metadata["headers"]:
        metadata["headers"].pop(
            "authorization"
        )  # do not store the original `sk-..` api key in the db

    payload = {
        "request_id": id,
        "call_type": call_type,
        "api_key": api_key,
        "cache_hit": cache_hit,
        "startTime": start_time,
        "endTime": end_time,
        "model": kwargs.get("model", ""),
        "user": kwargs.get("user", ""),
        "modelParameters": optional_params,
        "messages": messages,
        "response": response_obj,
        "usage": usage,
        "metadata": metadata,
    }

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
            elif type(payload[param]) == litellm.Usage:
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
        now = datetime.utcnow()
        keys_to_reset = await prisma_client.get_data(
            table_name="key", query_type="find_all", expires=now, reset_at=now
        )

        for key in keys_to_reset:
            key.spend = 0.0
            duration_s = _duration_in_seconds(duration=key.budget_duration)
            key.budget_reset_at = key.budget_reset_at + timedelta(seconds=duration_s)

        if len(keys_to_reset) > 0:
            await prisma_client.update_data(
                query_type="update_many", data_list=keys_to_reset, table_name="key"
            )
