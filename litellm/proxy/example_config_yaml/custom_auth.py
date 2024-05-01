from litellm.proxy._types import UserAPIKeyAuth, GenerateKeyRequest
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


async def generate_key_fn(data: GenerateKeyRequest):
    """
    Asynchronously decides if a key should be generated or not based on the provided data.

    Args:
        data (GenerateKeyRequest): The data to be used for decision making.

    Returns:
        bool: True if a key should be generated, False otherwise.
    """
    # decide if a key should be generated or not
    data_json = data.json()  # type: ignore

    # Unpacking variables
    team_id = data_json.get("team_id")
    duration = data_json.get("duration")
    models = data_json.get("models")
    aliases = data_json.get("aliases")
    config = data_json.get("config")
    spend = data_json.get("spend")
    user_id = data_json.get("user_id")
    max_parallel_requests = data_json.get("max_parallel_requests")
    metadata = data_json.get("metadata")
    tpm_limit = data_json.get("tpm_limit")
    rpm_limit = data_json.get("rpm_limit")

    if team_id is not None and len(team_id) > 0:
        return {
            "decision": True,
        }
    else:
        return {
            "decision": True,
            "message": "This violates LiteLLM Proxy Rules. No team id provided.",
        }
