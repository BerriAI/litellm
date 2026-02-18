"""
OpenAI Evals API endpoints - /v1/evals
"""

from typing import Optional

import orjson
from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.llms.openai_evals import (
    CancelEvalResponse,
    CancelRunResponse,
    DeleteEvalResponse,
    Eval,
    ListEvalsResponse,
    ListRunsResponse,
    Run,
    RunDeleteResponse,
)

router = APIRouter()


@router.post(
    "/v1/evals",
    tags=["OpenAI Evals API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=Eval,
)
async def create_eval(
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new evaluation.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`

    Example usage:
    ```bash
    curl -X POST "http://localhost:4000/v1/evals" \
      -H "Authorization: Bearer your-key" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Test Eval",
        "data_source_config": {"type": "file", "file_id": "file-abc123"},
        "testing_criteria": {"graders": [{"type": "llm_as_judge"}]}
      }'
    ```

    Returns: Eval object with id, status, timestamps, etc.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Extract model for routing (header > query > body)
    # When using extra_body={"model": "..."}, the OpenAI SDK merges it into the body
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_eval",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/evals",
    tags=["OpenAI Evals API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListEvalsResponse,
)
async def list_evals(
    fastapi_response: Response,
    request: Request,
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    order: Optional[str] = None,
    order_by: Optional[str] = None,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List evaluations with pagination.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`

    Example usage:
    ```bash
    curl "http://localhost:4000/v1/evals?limit=10" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: ListEvalsResponse with list of evaluations
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body (optional for GET)
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Use query params if not in body
    if "limit" not in data and limit is not None:
        data["limit"] = limit
    if "after" not in data and after is not None:
        data["after"] = after
    if "before" not in data and before is not None:
        data["before"] = before
    if "order" not in data and order is not None:
        data["order"] = order
    if "order_by" not in data and order_by is not None:
        data["order_by"] = order_by

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_evals",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/evals/{eval_id}",
    tags=["OpenAI Evals API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=Eval,
)
async def get_eval(
    eval_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a specific evaluation by ID.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`

    Example usage:
    ```bash
    curl "http://localhost:4000/v1/evals/eval_123" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: Eval object
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body (optional for GET)
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id from path parameter
    data["eval_id"] = eval_id

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_eval",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/evals/{eval_id}",
    tags=["OpenAI Evals API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=Eval,
)
async def update_eval(
    eval_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an evaluation.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`

    Example usage:
    ```bash
    curl -X POST "http://localhost:4000/v1/evals/eval_123" \
      -H "Authorization: Bearer your-key" \
      -H "Content-Type: application/json" \
      -d '{"name": "Updated Name"}'
    ```

    Returns: Updated Eval object
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id from path parameter
    data["eval_id"] = eval_id

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aupdate_eval",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/evals/{eval_id}",
    tags=["OpenAI Evals API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=DeleteEvalResponse,
)
async def delete_eval(
    eval_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete an evaluation.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`

    Example usage:
    ```bash
    curl -X DELETE "http://localhost:4000/v1/evals/eval_123" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: DeleteEvalResponse with deletion confirmation
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body (optional for DELETE)
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id from path parameter
    data["eval_id"] = eval_id

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_eval",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/evals/{eval_id}/cancel",
    tags=["OpenAI Evals API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CancelEvalResponse,
)
async def cancel_eval(
    eval_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a running evaluation.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`

    Example usage:
    ```bash
    curl -X POST "http://localhost:4000/v1/evals/eval_123/cancel" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: CancelEvalResponse with cancellation confirmation
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body (optional for cancel)
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id from path parameter
    data["eval_id"] = eval_id

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_eval",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )

# ===================================
# Run API Endpoints
# ===================================


@router.post(
    "/v1/evals/{eval_id}/runs",
    tags=["OpenAI Evals API - Runs"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=Run,
)
async def create_run(
    eval_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new run for an evaluation.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`
    - Pass model via body: `{"model": "gpt-4-account-1"}`
    - Pass model via completion.model: `{"completion": {"model": "gpt-4-account-1"}}`

    Example usage:
    ```bash
    curl -X POST "http://localhost:4000/v1/evals/eval_123/runs" \
      -H "Authorization: Bearer your-key" \
      -H "Content-Type: application/json" \
      -d '{
        "data_source": {"type": "dataset", "dataset_id": "dataset_123"},
        "completion": {"model": "gpt-4", "temperature": 0.7}
      }'
    ```

    Returns: Run object with id, status, timestamps, etc.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id from path parameter
    data["eval_id"] = eval_id

    # Extract model for routing (header > query > body > completion.model)
    model = (
        request.headers.get("x-litellm-model")
        or request.query_params.get("model")
        or data.get("model")
        or (data.get("completion", {}).get("model") if isinstance(data.get("completion"), dict) else None)
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_run",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/evals/{eval_id}/runs",
    tags=["OpenAI Evals API - Runs"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListRunsResponse,
)
async def list_runs(
    eval_id: str,
    fastapi_response: Response,
    request: Request,
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    order: Optional[str] = None,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all runs for an evaluation with pagination.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`

    Example usage:
    ```bash
    curl "http://localhost:4000/v1/evals/eval_123/runs?limit=10" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: ListRunsResponse with list of runs
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Build request data
    data = {
        "eval_id": eval_id,
        "limit": limit,
        "after": after,
        "before": before,
        "order": order,
    }

    # Extract model for routing (header > query)
    model = request.headers.get("x-litellm-model") or request.query_params.get(
        "model"
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_runs",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=str(data.get("model")) if data.get("model") is not None else None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/evals/{eval_id}/runs/{run_id}",
    tags=["OpenAI Evals API - Runs"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=Run,
)
async def get_run(
    eval_id: str,
    run_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a specific run by ID.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`

    Example usage:
    ```bash
    curl "http://localhost:4000/v1/evals/eval_123/runs/run_456" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: Run object with full details
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Build request data
    data = {
        "eval_id": eval_id,
        "run_id": run_id,
    }

    # Extract model for routing (header > query)
    model = request.headers.get("x-litellm-model") or request.query_params.get(
        "model"
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data and custom_llm_provider is not None:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_run",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/evals/{eval_id}/runs/{run_id}",
    tags=["OpenAI Evals API - Runs"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CancelRunResponse,
)
async def cancel_run(
    eval_id: str,
    run_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a running run.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`

    Example usage:
    ```bash
    curl -X POST "http://localhost:4000/v1/evals/eval_123/runs/run_456/cancel" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: CancelRunResponse with cancellation confirmation
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body (optional for cancel)
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id and run_id from path parameters
    data["eval_id"] = eval_id
    data["run_id"] = run_id

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_run",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/evals/{eval_id}/runs/{run_id}",
    tags=["OpenAI Evals API - Runs"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RunDeleteResponse,
)
async def delete_run(
    eval_id: str,
    run_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a run.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: gpt-4-account-1`
    - Pass model via query: `?model=gpt-4-account-1`

    Example usage:
    ```bash
    curl -X DELETE "http://localhost:4000/v1/evals/eval_123/runs/run_456" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: RunDeleteResponse with deletion confirmation
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body (optional for delete)
    body = await request.body()
    data = orjson.loads(body) if body else {}

    # Set eval_id and run_id from path parameters
    data["eval_id"] = eval_id
    data["run_id"] = run_id

    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_run",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
