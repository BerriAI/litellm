from litellm.proxy.types import UserAPIKeyAuth
from fastapi import Request
from dotenv import load_dotenv
import os 

load_dotenv()
async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth: 
    try: 
        modified_master_key = f"{os.getenv('PROXY_MASTER_KEY')}-1234"
        if api_key == modified_master_key:
            return UserAPIKeyAuth(api_key=api_key)
        raise Exception
    except: 
        raise Exception