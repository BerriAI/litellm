import ast
import asyncio
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.main import FileObject
from litellm.fine_tuning.main import vertex_fine_tuning_apis_instance
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()
default_vertex_config = None


def set_default_vertex_config(config):
    global default_vertex_config
    if config is None:
        return

    if not isinstance(config, dict):
        raise ValueError("invalid config, vertex default config must be a dictionary")

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                config[key] = litellm.get_secret(value)

    default_vertex_config = config


def exception_handler(e: Exception):
    verbose_proxy_logger.error(
        "litellm.proxy.proxy_server.v1/projects/tuningJobs(): Exception occurred - {}".format(
            str(e)
        )
    )
    verbose_proxy_logger.debug(traceback.format_exc())
    if isinstance(e, HTTPException):
        return ProxyException(
            message=getattr(e, "message", str(e.detail)),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
        )
    else:
        error_msg = f"{str(e)}"
        return ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


async def execute_post_vertex_ai_request(
    request: Request,
    route: str,
):
    from litellm.fine_tuning.main import vertex_fine_tuning_apis_instance

    if default_vertex_config is None:
        raise ValueError(
            "Vertex credentials not added on litellm proxy, please add `default_vertex_config` on your config.yaml"
        )
    vertex_project = default_vertex_config.get("vertex_project", None)
    vertex_location = default_vertex_config.get("vertex_location", None)
    vertex_credentials = default_vertex_config.get("vertex_credentials", None)

    request_data_json = {}
    body = await request.body()
    body_str = body.decode()
    if len(body_str) > 0:
        try:
            request_data_json = ast.literal_eval(body_str)
        except:
            request_data_json = json.loads(body_str)

    verbose_proxy_logger.debug(
        "Request received by LiteLLM:\n{}".format(
            json.dumps(request_data_json, indent=4)
        ),
    )

    response = (
        await vertex_fine_tuning_apis_instance.pass_through_vertex_ai_POST_request(
            request_data=request_data_json,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            request_route=route,
        )
    )

    return response


@router.post(
    "/vertex-ai/publishers/google/models/{model_id:path}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_generate_content(
    request: Request,
    fastapi_response: Response,
    model_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /generateContent endpoint

    Example Curl:
    ```
    curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:generateContent \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-1234" \
      -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
    ```

    Vertex API Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#rest
    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route=f"/publishers/google/models/{model_id}:generateContent",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e


@router.post(
    "/vertex-ai/publishers/google/models/{model_id:path}:predict",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_predict_endpoint(
    request: Request,
    fastapi_response: Response,
    model_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /predict endpoint
    Use this for:
    - Embeddings API - Text Embedding, Multi Modal Embedding
    - Imagen API
    - Code Completion API

    Example Curl:
    ```
    curl http://localhost:4000/vertex-ai/publishers/google/models/textembedding-gecko@001:predict \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-1234" \
      -d '{"instances":[{"content": "gm"}]}'
    ```

    Vertex API Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#generative-ai-get-text-embedding-drest
    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route=f"/publishers/google/models/{model_id}:predict",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e


@router.post(
    "/vertex-ai/publishers/google/models/{model_id:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_countTokens_endpoint(
    request: Request,
    fastapi_response: Response,
    model_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /countTokens endpoint
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/count-tokens#curl


    Example Curl:
    ```
    curl http://localhost:4000/vertex-ai/publishers/google/models/gemini-1.5-flash-001:countTokens \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer sk-1234" \
      -d '{"contents":[{"role": "user", "parts":[{"text": "hi"}]}]}'
    ```

    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route=f"/publishers/google/models/{model_id}:countTokens",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e


@router.post(
    "/vertex-ai/batchPredictionJobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_create_batch_prediction_job(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /batchPredictionJobs endpoint

    Vertex API Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api#syntax

    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route="/batchPredictionJobs",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e


@router.post(
    "/vertex-ai/tuningJobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_create_fine_tuning_job(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /tuningJobs endpoint

    Vertex API Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning

    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route="/tuningJobs",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e


@router.post(
    "/vertex-ai/tuningJobs/{job_id:path}:cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_cancel_fine_tuning_job(
    request: Request,
    job_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. tuningJobs/{job_id:path}:cancel

    Vertex API Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning#cancel_a_tuning_job

    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:

        response = await execute_post_vertex_ai_request(
            request=request,
            route=f"/tuningJobs/{job_id}:cancel",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e


@router.post(
    "/vertex-ai/cachedContents",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_create_add_cached_content(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /cachedContents endpoint

    Vertex API Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-create#create-context-cache-sample-drest

    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route="/cachedContents",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e
