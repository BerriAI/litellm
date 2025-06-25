from fastapi import APIRouter, Request

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter(
    tags=["google genai endpoints"],
)


@router.post("/v1beta/models/{model_name}:countTokens")
async def google_count_tokens(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai countTokens endpoint.
    """
    return {}


@router.post("/v1beta/models/{model_name}:generateContent")
async def google_generate_content(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai generateContent endpoint.
    """
    return {}


@router.post("/v1beta/models/{model_name}:streamGenerateContent")
async def google_stream_generate_content(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai streamGenerateContent endpoint.
    """
    return {}

