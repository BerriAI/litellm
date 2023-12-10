from typing import Optional, List, Any, Literal
import os, subprocess, hashlib, importlib, asyncio
import litellm, backoff
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from litellm.proxy.hooks.parallel_request_limiter import MaxParallelRequestsHandler

def print_verbose(print_statement):
    if litellm.set_verbose:
        print(print_statement) # noqa
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
        pass

    def _init_litellm_callbacks(self):
        
        litellm.callbacks.append(self.max_parallel_request_limiter)
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
            litellm.utils.set_callbacks(
                    callback_list=callback_list
                )

    async def pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, data: dict, call_type: Literal["completion", "embeddings"]): 
        """
        Allows users to modify/reject the incoming request to the proxy, without having to deal with parsing Request body.

        Covers: 
        1. /chat/completions
        2. /embeddings 
        """
        try: 
            self.call_details["data"] = data
            self.call_details["call_type"] = call_type

            ## check if max parallel requests set   
            if user_api_key_dict.max_parallel_requests is not None: 
                ## if set, check if request allowed
                await self.max_parallel_request_limiter.max_parallel_request_allow_request(
                    max_parallel_requests=user_api_key_dict.max_parallel_requests,
                    api_key=user_api_key_dict.api_key,
                    user_api_key_cache=self.call_details["user_api_key_cache"])
                
            return data
        except Exception as e:
            raise e
        
    async def success_handler(self, *args, **kwargs): 
        """
        Log successful db read/writes
        """
        pass

    async def failure_handler(self, original_exception):
        """
        Log failed db read/writes

        Currently only logs exceptions to sentry
        """
        if litellm.utils.capture_exception: 
            litellm.utils.capture_exception(error=original_exception)

    async def post_call_failure_hook(self, original_exception: Exception, user_api_key_dict: UserAPIKeyAuth): 
        """
        Allows users to raise custom exceptions/log when a call fails, without having to deal with parsing Request body.

        Covers: 
        1. /chat/completions
        2. /embeddings 
        """
        # check if max parallel requests set
        if user_api_key_dict.max_parallel_requests is not None:
            ## decrement call count if call failed
            if (hasattr(original_exception, "status_code") 
                and original_exception.status_code == 429 
                and "Max parallel request limit reached" in str(original_exception)):
                pass # ignore failed calls due to max limit being reached
            else:  
                await self.max_parallel_request_limiter.async_log_failure_call(
                    api_key=user_api_key_dict.api_key,
                    user_api_key_cache=self.call_details["user_api_key_cache"])
        return
   

### DB CONNECTOR ###
# Define the retry decorator with backoff strategy
# Function to be called whenever a retry is about to happen
def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    print_verbose(f"Backing off... this was attempt #{details['tries']}")

class PrismaClient:
    def __init__(self, database_url: str, proxy_logging_obj: ProxyLogging):
        print_verbose("LiteLLM: DATABASE_URL Set in config, trying to 'pip install prisma'")
        ## init logging object
        self.proxy_logging_obj = proxy_logging_obj

        os.environ["DATABASE_URL"] = database_url
        # Save the current working directory
        original_dir = os.getcwd()
        # set the working directory to where this script is
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        try:
            subprocess.run(['prisma', 'generate'])
            subprocess.run(['prisma', 'db', 'push', '--accept-data-loss']) # this looks like a weird edge case when prisma just wont start on render. we need to have the --accept-data-loss
        finally:
            os.chdir(original_dir)
        # Now you can import the Prisma Client
        from prisma import Client # type: ignore
        self.db = Client()  #Client to connect to Prisma db

        

    def hash_token(self, token: str):
        # Hash the string using SHA-256
        hashed_token = hashlib.sha256(token.encode()).hexdigest()
        
        return hashed_token

    @backoff.on_exception(
        backoff.expo,
        Exception,        # base exception to catch for the backoff
        max_tries=3,      # maximum number of retries
        max_time=10,      # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def get_data(self, token: str, expires: Optional[Any]=None): 
        try: 
            # check if plain text or hash
            if token.startswith("sk-"): 
                token = self.hash_token(token=token)
            if expires: 
                response = await self.db.litellm_verificationtoken.find_first(
                        where={
                            "token": token,
                            "expires": {"gte": expires}  # Check if the token is not expired
                        }
                    )
            else: 
                response = await self.db.litellm_verificationtoken.find_unique(
                    where={
                        "token": token
                    }
                )
            return response
        except Exception as e: 
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,        # base exception to catch for the backoff
        max_tries=3,      # maximum number of retries
        max_time=10,      # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def insert_data(self, data: dict):
        """
        Add a key to the database. If it already exists, do nothing. 
        """
        try: 
            token = data["token"]
            hashed_token = self.hash_token(token=token)
            data["token"] = hashed_token

            new_verification_token = await self.db.litellm_verificationtoken.upsert( # type: ignore
                where={
                    'token': hashed_token,
                },
                data={
                    "create": {**data}, #type: ignore
                    "update": {} # don't do anything if it already exists
                }
            )

            return new_verification_token
        except Exception as e:
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,        # base exception to catch for the backoff
        max_tries=3,      # maximum number of retries
        max_time=10,      # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def update_data(self, token: str, data: dict):
        """
        Update existing data
        """
        try: 
            print_verbose(f"token: {token}")
            # check if plain text or hash
            if token.startswith("sk-"): 
                token = self.hash_token(token=token)

            data["token"] = token 
            response = await self.db.litellm_verificationtoken.update(
                where={
                    "token": token
                },
                data={**data} # type: ignore 
            )
            print_verbose("\033[91m" + f"DB write succeeded {response}" + "\033[0m")
            return {"token": token, "data": data}
        except Exception as e: 
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            print_verbose("\033[91m" + f"DB write failed: {e}" + "\033[0m")
            raise e


    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,        # base exception to catch for the backoff
        max_tries=3,      # maximum number of retries
        max_time=10,      # maximum total time to retry for
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
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            raise e
    
    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,        # base exception to catch for the backoff
        max_tries=3,      # maximum number of retries
        max_time=10,      # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def connect(self): 
        try:
            await self.db.connect()
        except Exception as e: 
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            raise e

    # Define a retrying strategy with exponential backoff
    @backoff.on_exception(
        backoff.expo,
        Exception,        # base exception to catch for the backoff
        max_tries=3,      # maximum number of retries
        max_time=10,      # maximum total time to retry for
        on_backoff=on_backoff,  # specifying the function to call on backoff
    )
    async def disconnect(self): 
        try:
            await self.db.disconnect()
        except Exception as e: 
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            raise e

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
            module_file_path = os.path.join(directory, *module_name.split('.'))
            module_file_path += '.py'

            spec = importlib.util.spec_from_file_location(module_name, module_file_path)
            if spec is None:
                raise ImportError(f"Could not find a module specification for {module_file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module) # type: ignore
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

    