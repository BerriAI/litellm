from fastapi import APIRouter, Depends, Request

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter(
    tags=["google genai endpoints"],
)


@router.post("/v1beta/models/{model_name}:countTokens", dependencies=[Depends(user_api_key_auth)])
async def google_count_tokens(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai countTokens endpoint.
    """
    return {}


@router.post("/v1beta/models/{model_name}:generateContent", dependencies=[Depends(user_api_key_auth)])
async def google_generate_content(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai generateContent endpoint.
    """
    return {}


@router.post("/v1beta/models/{model_name}:streamGenerateContent", dependencies=[Depends(user_api_key_auth)])
async def google_stream_generate_content(request: Request, model_name: str):
    """
    Not Implemented, this is a placeholder for the google genai streamGenerateContent endpoint.
    """
    return {}

