from typing import Optional, List, Any, Literal
import os, subprocess, hashlib, importlib, asyncio
import litellm, backoff

### LOGGING ### 
class ProxyLogging: 
    """
    Logging for proxy.  

    Implemented mainly to log successful/failed db read/writes.

    Currently just logs this to a provided sentry integration.
    """

    def __init__(self,):
        ## INITIALIZE  LITELLM CALLBACKS ##
        self._init_litellm_callbacks()
        pass


    def _init_litellm_callbacks(self): 
        if len(litellm.callbacks) > 0: 
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
   

### DB CONNECTOR ###
# Define the retry decorator with backoff strategy
# Function to be called whenever a retry is about to happen
def on_backoff(details):
    # The 'tries' key in the details dictionary contains the number of completed tries
    print(f"Backing off... this was attempt #{details['tries']}")

class PrismaClient:
    def __init__(self, database_url: str, proxy_logging_obj: ProxyLogging):
        print("LiteLLM: DATABASE_URL Set in config, trying to 'pip install prisma'")
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
            hashed_token = self.hash_token(token=token)
            if expires: 
                response = await self.db.litellm_verificationtoken.find_first(
                        where={
                            "token": hashed_token,
                            "expires": {"gte": expires}  # Check if the token is not expired
                        }
                    )
            else: 
                response = await self.db.litellm_verificationtoken.find_unique(
                    where={
                        "token": hashed_token
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
            hashed_token = self.hash_token(token=token)
            data["token"] = hashed_token
            await self.db.litellm_verificationtoken.update(
                where={
                    "token": hashed_token
                },
                data={**data} # type: ignore 
            )
            print("\033[91m" + f"DB write succeeded" + "\033[0m")
            return {"token": token, "data": data}
        except Exception as e: 
            asyncio.create_task(self.proxy_logging_obj.failure_handler(original_exception=e))
            print()
            print()
            print()
            print("\033[91m" + f"DB write failed: {e}" + "\033[0m")
            print()
            print()
            print()
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
        print(f"value: {value}")
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

 

### CALL HOOKS ###
class CallHooks: 
    """
    Allows users to modify the incoming request / output to the proxy, without having to deal with parsing Request body.

    Covers: 
    1. /chat/completions
    2. /embeddings 
    """

    def __init__(self, *args, **kwargs):
        self.call_details = {}

    def update_router_config(self, litellm_settings: dict, general_settings: dict, model_list: list):
        self.call_details["litellm_settings"] = litellm_settings
        self.call_details["general_settings"] = general_settings
        self.call_details["model_list"] = model_list

    def pre_call(self, data: dict, call_type: Literal["completion", "embeddings"]): 
        self.call_details["data"] = data
        return data

    def post_call_success(self, response: Optional[Any]=None, call_type: Optional[Literal["completion", "embeddings"]]=None, chunk: Optional[Any]=None):
        return response 

    def post_call_failure(self, *args, **kwargs): 
        pass