from typing import Optional, List, Any 
import os, subprocess, hashlib, importlib

### DB CONNECTOR ###
class PrismaClient:
    def __init__(self, database_url: str):
        print("LiteLLM: DATABASE_URL Set in config, trying to 'pip install prisma'")
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

    async def get_data(self, token: str, expires: Optional[Any]=None):
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

    async def insert_data(self, data: dict):
        """
        Add a key to the database. If it already exists, do nothing. 
        """
        token = data["token"]
        hashed_token = self.hash_token(token=token)
        data["token"] = hashed_token
        print(f"passed in data: {data}; hashed_token: {hashed_token}")

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

    async def update_data(self, token: str, data: dict):
        """
        Update existing data
        """
        hashed_token = self.hash_token(token=token)
        data["token"] = hashed_token
        await self.db.litellm_verificationtoken.update(
            where={
                "token": hashed_token
            },
            data={**data} # type: ignore 
        )
        return {"token": token, "data": data}

    async def delete_data(self, tokens: List):
        """
        Allow user to delete a key(s)
        """
        hashed_tokens = [self.hash_token(token=token) for token in tokens]
        await self.db.litellm_verificationtoken.delete_many(
                where={"token": {"in": hashed_tokens}}
            )
        return {"deleted_keys": tokens}
    
    async def connect(self): 
        await self.db.connect()

    async def disconnect(self): 
        await self.db.disconnect()

# ### CUSTOM FILE ###
# def get_instance_fn(value: str, config_file_path: Optional[str]=None):
#     try:
#         # Split the path by dots to separate module from instance
#         parts = value.split(".")
#         # The module path is all but the last part, and the instance is the last part
#         module_path = ".".join(parts[:-1])
#         instance_name = parts[-1]
        
#         if config_file_path is not None: 
#             directory = os.path.dirname(config_file_path)
#             module_path = os.path.join(directory, module_path)
#         # Dynamically import the module
#         module = importlib.import_module(module_path)
        
#         # Get the instance from the module
#         instance = getattr(module, instance_name)
        
#         return instance
#     except ImportError as e:
#         print(e)
#         raise ImportError(f"Could not import file at {value}")

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