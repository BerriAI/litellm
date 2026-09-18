"""
Mock server that implements the /beta/litellm_prompt_management endpoint
and acts as a wrapper for calling the Braintrust API.

This server transforms Braintrust's prompt API response into the format
expected by LiteLLM's generic prompt management client.

Usage:
    python braintrust_prompt_wrapper_server.py

    # Then test with:
    curl -H "Authorization: Bearer YOUR_BRAINTRUST_TOKEN" \
         "http://localhost:8080/beta/litellm_prompt_management?prompt_id=YOUR_PROMPT_ID"
"""

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI(
    title="Braintrust Prompt Wrapper",
    description="Wrapper server for Braintrust prompts to work with LiteLLM",
    version="1.0.0",
)


def transform_braintrust_message(message: Dict[str, Any]) -> Dict[str, str]:
    """
    Transform a Braintrust message to LiteLLM format.

    Braintrust message format:
    {
        "role": "system",
        "content": "...",
        "name": "..." (optional)
    }

    LiteLLM format:
    {
        "role": "system",
        "content": "..."
    }
    """
    result = {
        "role": message.get("role", "user"),
        "content": message.get("content", ""),
    }

    # Include name if present
    if "name" in message:
        result["name"] = message["name"]

    return result


def transform_braintrust_response(
    braintrust_response: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Transform Braintrust API response to LiteLLM prompt management format.

    Braintrust response format:
    {
        "objects": [{
            "id": "prompt_id",
            "prompt_data": {
                "prompt": {
                    "type": "chat",
                    "messages": [...],
                    "tools": "..."
                },
                "options": {
                    "model": "gpt-4",
                    "params": {
                        "temperature": 0.7,
                        "max_tokens": 100,
                        ...
                    }
                }
            }
        }]
    }

    LiteLLM format:
    {
        "prompt_id": "prompt_id",
        "prompt_template": [...],
        "prompt_template_model": "gpt-4",
        "prompt_template_optional_params": {...}
    }
    """
    # Extract the first object from the objects array if it exists
    if "objects" in braintrust_response and len(braintrust_response["objects"]) > 0:
        prompt_object = braintrust_response["objects"][0]
    else:
        prompt_object = braintrust_response

    prompt_data = prompt_object.get("prompt_data", {})
    prompt_info = prompt_data.get("prompt", {})
    options = prompt_data.get("options", {})

    # Extract messages
    messages = prompt_info.get("messages", [])
    transformed_messages = [transform_braintrust_message(msg) for msg in messages]

    # Extract model
    model = options.get("model")

    # Extract optional parameters
    params = options.get("params", {})
    optional_params: Dict[str, Any] = {}

    # Map common parameters
    param_mapping = {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "max_completion_tokens": "max_tokens",  # Alternative name
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "n": "n",
        "stop": "stop",
    }

    for braintrust_param, litellm_param in param_mapping.items():
        if braintrust_param in params:
            value = params[braintrust_param]
            if value is not None:
                optional_params[litellm_param] = value

    # Handle response_format
    if "response_format" in params:
        optional_params["response_format"] = params["response_format"]

    # Handle tool_choice
    if "tool_choice" in params:
        optional_params["tool_choice"] = params["tool_choice"]

    # Handle function_call
    if "function_call" in params:
        optional_params["function_call"] = params["function_call"]

    # Add tools if present
    if "tools" in prompt_info and prompt_info["tools"]:
        optional_params["tools"] = prompt_info["tools"]

    # Handle tool_functions from prompt_data
    if "tool_functions" in prompt_data and prompt_data["tool_functions"]:
        optional_params["tool_functions"] = prompt_data["tool_functions"]

    return {
        "prompt_id": prompt_object.get("id"),
        "prompt_template": transformed_messages,
        "prompt_template_model": model,
        "prompt_template_optional_params": optional_params if optional_params else None,
    }


@app.get("/beta/litellm_prompt_management")
async def get_prompt(
    prompt_id: str = Query(..., description="The Braintrust prompt ID to fetch"),
    authorization: Optional[str] = Header(
        None, description="Bearer token for Braintrust API"
    ),
) -> JSONResponse:
    """
    Fetch a prompt from Braintrust and transform it to LiteLLM format.

    Args:
        prompt_id: The Braintrust prompt ID
        authorization: Bearer token for Braintrust API (from header)

    Returns:
        JSONResponse with the transformed prompt data
    """
    # Extract token from Authorization header or environment
    braintrust_token = None
    if authorization and authorization.startswith("Bearer "):
        braintrust_token = authorization.replace("Bearer ", "")
    else:
        braintrust_token = os.getenv("BRAINTRUST_API_KEY")

    if not braintrust_token:
        raise HTTPException(
            status_code=401,
            detail="No Braintrust API token provided. Pass via Authorization header or set BRAINTRUST_API_KEY environment variable.",
        )

    # Call Braintrust API
    braintrust_url = f"https://api.braintrust.dev/v1/prompt/{prompt_id}"
    headers = {
        "Authorization": f"Bearer {braintrust_token}",
        "Accept": "application/json",
    }
    print(f"headers: {headers}")
    print(f"braintrust_url: {braintrust_url}")
    print(f"braintrust_token: {braintrust_token}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(braintrust_url, headers=headers)
            response.raise_for_status()
            braintrust_data = response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Braintrust API error: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Braintrust API: {str(e)}",
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse Braintrust API response: {str(e)}",
        )

    print(f"braintrust_data: {braintrust_data}")
    # Transform the response
    try:
        transformed_data = transform_braintrust_response(braintrust_data)
        print(f"transformed_data: {transformed_data}")
        return JSONResponse(content=transformed_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transform Braintrust response: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "braintrust-prompt-wrapper"}


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Braintrust Prompt Wrapper for LiteLLM",
        "version": "1.0.0",
        "endpoints": {
            "prompt_management": "/beta/litellm_prompt_management?prompt_id=<id>",
            "health": "/health",
        },
        "documentation": "/docs",
    }


def main():
    """Run the server."""
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"🚀 Starting Braintrust Prompt Wrapper Server on {host}:{port}")
    print(f"📚 API Documentation available at http://{host}:{port}/docs")
    print(
        f"🔑 Make sure to set BRAINTRUST_API_KEY environment variable or pass token in Authorization header"
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
