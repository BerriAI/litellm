from litellm.proxy._types import UserAPIKeyAuth
from fastapi import Request
from dotenv import load_dotenv
import os

load_dotenv()


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    try:
        print(f"api_key: {api_key}")
        if api_key == "":
            raise Exception(
                f"CustomAuth - Malformed API Key passed in. Ensure Key has `Bearer` prefix"
            )
        if api_key == f"{os.getenv('PROXY_MASTER_KEY')}-1234":
            return UserAPIKeyAuth(api_key=api_key)
        raise Exception
    except Exception as e:
        if len(str(e)) > 0:
            raise e
        raise Exception("Failed custom auth")
