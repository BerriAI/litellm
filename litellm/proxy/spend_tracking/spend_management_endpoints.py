#### SPEND MANAGEMENT #####
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get(
    "/spend/keys",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def spend_key_fn():
    """
    View all keys created, ordered by spend

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/keys" \
-H "Authorization: Bearer sk-1234"
    ```
    """

    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        key_info = await prisma_client.get_data(table_name="key", query_type="find_all")
        return key_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/spend/users",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def spend_user_fn(
    user_id: Optional[str] = fastapi.Query(
        default=None,
        description="Get User Table row for user_id",
    ),
):
    """
    View all users created, ordered by spend

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/users" \
-H "Authorization: Bearer sk-1234"
    ```

    View User Table row for user_id
    ```
    curl -X GET "http://0.0.0.0:8000/spend/users?user_id=1234" \
-H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if user_id is not None:
            user_info = await prisma_client.get_data(
                table_name="user", query_type="find_unique", user_id=user_id
            )
            return [user_info]
        else:
            user_info = await prisma_client.get_data(
                table_name="user", query_type="find_all"
            )

        return user_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/spend/tags",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def view_spend_tags(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing key spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view key spend",
    ),
):
    """
    LiteLLM Enterprise - View Spend Per Request Tag

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:8000/spend/tags" \
-H "Authorization: Bearer sk-1234"
    ```

    Spend with Start Date and End Date
    ```
    curl -X GET "http://0.0.0.0:8000/spend/tags?start_date=2022-01-01&end_date=2022-02-01" \
-H "Authorization: Bearer sk-1234"
    ```
    """

    from enterprise.utils import get_spend_by_tags
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        # run the following SQL query on prisma
        """
        SELECT
        jsonb_array_elements_text(request_tags) AS individual_request_tag,
        COUNT(*) AS log_count,
        SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        GROUP BY individual_request_tag;
        """
        response = await get_spend_by_tags(
            start_date=start_date, end_date=end_date, prisma_client=prisma_client
        )

        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/tags Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/tags Error" + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def get_global_activity_internal_user(
    user_api_key_dict: UserAPIKeyAuth, start_date: datetime, end_date: datetime
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    user_id = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(status_code=500, detail={"error": "No user_id found"})

    sql_query = """
    SELECT
        date_trunc('day', "startTime") AS date,
        COUNT(*) AS api_requests,
        SUM(total_tokens) AS total_tokens
    FROM "LiteLLM_SpendLogs"
    WHERE "startTime" BETWEEN $1::date AND $2::date + interval '1 day'
    AND "user" = $3
    GROUP BY date_trunc('day', "startTime")
    """
    db_response = await prisma_client.db.query_raw(
        sql_query, start_date, end_date, user_id
    )

    return db_response


@router.get(
    "/global/activity",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
    include_in_schema=False,
)
async def get_global_activity(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get number of API Requests, total tokens through proxy

    {
        "daily_data": [
                const chartdata = [
                {
                date: 'Jan 22',
                api_requests: 10,
                total_tokens: 2000
                },
                {
                date: 'Jan 23',
                api_requests: 10,
                total_tokens: 12
                },
        ],
        "sum_api_requests": 20,
        "sum_total_tokens": 2012
    }
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import llm_router, prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if (
            user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
            or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        ):
            db_response = await get_global_activity_internal_user(
                user_api_key_dict, start_date_obj, end_date_obj
            )
        else:

            sql_query = """
            SELECT
                date_trunc('day', "startTime") AS date,
                COUNT(*) AS api_requests,
                SUM(total_tokens) AS total_tokens
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" BETWEEN $1::date AND $2::date + interval '1 day'
            GROUP BY date_trunc('day', "startTime")
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj
            )

        if db_response is None:
            return []

        sum_api_requests = 0
        sum_total_tokens = 0
        daily_data = []
        for row in db_response:
            # cast date to datetime
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            daily_data.append(row)
            sum_api_requests += row.get("api_requests", 0)
            sum_total_tokens += row.get("total_tokens", 0)

        # sort daily_data by date
        daily_data = sorted(daily_data, key=lambda x: x["date"])

        data_to_return = {
            "daily_data": daily_data,
            "sum_api_requests": sum_api_requests,
            "sum_total_tokens": sum_total_tokens,
        }

        return data_to_return

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


async def get_global_activity_model_internal_user(
    user_api_key_dict: UserAPIKeyAuth, start_date: datetime, end_date: datetime
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    user_id = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(status_code=500, detail={"error": "No user_id found"})

    sql_query = """
    SELECT
        model_group,
        date_trunc('day', "startTime") AS date,
        COUNT(*) AS api_requests,
        SUM(total_tokens) AS total_tokens
    FROM "LiteLLM_SpendLogs"
    WHERE "startTime" BETWEEN $1::date AND $2::date + interval '1 day'
    AND "user" = $3
    GROUP BY model_group, date_trunc('day', "startTime")
    """
    db_response = await prisma_client.db.query_raw(
        sql_query, start_date, end_date, user_id
    )

    return db_response


@router.get(
    "/global/activity/model",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
    include_in_schema=False,
)
async def get_global_activity_model(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get number of API Requests, total tokens through proxy - Grouped by MODEL

    [
        {
            "model": "gpt-4",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    api_requests: 10,
                    total_tokens: 2000
                    },
                    {
                    date: 'Jan 23',
                    api_requests: 10,
                    total_tokens: 12
                    },
            ],
            "sum_api_requests": 20,
            "sum_total_tokens": 2012

        },
        {
            "model": "azure/gpt-4-turbo",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    api_requests: 10,
                    total_tokens: 2000
                    },
                    {
                    date: 'Jan 23',
                    api_requests: 10,
                    total_tokens: 12
                    },
            ],
            "sum_api_requests": 20,
            "sum_total_tokens": 2012

        },
    ]
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import llm_router, premium_user, prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if (
            user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
            or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        ):
            db_response = await get_global_activity_model_internal_user(
                user_api_key_dict, start_date_obj, end_date_obj
            )
        else:

            sql_query = """
            SELECT
                model_group,
                date_trunc('day', "startTime") AS date,
                COUNT(*) AS api_requests,
                SUM(total_tokens) AS total_tokens
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" BETWEEN $1::date AND $2::date + interval '1 day'
            GROUP BY model_group, date_trunc('day', "startTime")
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj
            )
        if db_response is None:
            return []

        model_ui_data: dict = (
            {}
        )  # {"gpt-4": {"daily_data": [], "sum_api_requests": 0, "sum_total_tokens": 0}}

        for row in db_response:
            _model = row["model_group"]
            if _model not in model_ui_data:
                model_ui_data[_model] = {
                    "daily_data": [],
                    "sum_api_requests": 0,
                    "sum_total_tokens": 0,
                }
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            model_ui_data[_model]["daily_data"].append(row)
            model_ui_data[_model]["sum_api_requests"] += row.get("api_requests", 0)
            model_ui_data[_model]["sum_total_tokens"] += row.get("total_tokens", 0)

        # sort mode ui data by sum_api_requests -> get top 10 models
        model_ui_data = dict(
            sorted(
                model_ui_data.items(),
                key=lambda x: x[1]["sum_api_requests"],
                reverse=True,
            )[:10]
        )

        response = []
        for model, data in model_ui_data.items():
            _sort_daily_data = sorted(data["daily_data"], key=lambda x: x["date"])

            response.append(
                {
                    "model": model,
                    "daily_data": _sort_daily_data,
                    "sum_api_requests": data["sum_api_requests"],
                    "sum_total_tokens": data["sum_total_tokens"],
                }
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


@router.get(
    "/global/activity/exceptions/deployment",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
    include_in_schema=False,
)
async def get_global_activity_exceptions_per_deployment(
    model_group: str = fastapi.Query(
        description="Filter by model group",
    ),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of 429 errors - Grouped by deployment

    [
        {
            "deployment": "https://azure-us-east-1.openai.azure.com/",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    num_rate_limit_exceptions: 10
                    },
                    {
                    date: 'Jan 23',
                    num_rate_limit_exceptions: 12
                    },
            ],
            "sum_num_rate_limit_exceptions": 20,

        },
        {
            "deployment": "https://azure-us-east-1.openai.azure.com/",
            "daily_data": [
                    const chartdata = [
                    {
                    date: 'Jan 22',
                    num_rate_limit_exceptions: 10,
                    },
                    {
                    date: 'Jan 23',
                    num_rate_limit_exceptions: 12
                    },
            ],
            "sum_num_rate_limit_exceptions": 20,

        },
    ]
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import llm_router, premium_user, prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT
            api_base,
            date_trunc('day', "startTime")::date AS date,
            COUNT(*) AS num_rate_limit_exceptions
        FROM
            "LiteLLM_ErrorLogs"
        WHERE
            "startTime" >= $1::date
            AND "startTime" < ($2::date + INTERVAL '1 day')
            AND model_group = $3
            AND status_code = '429'
        GROUP BY
            api_base,
            date_trunc('day', "startTime")
        ORDER BY
            date;
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj, model_group
        )
        if db_response is None:
            return []

        model_ui_data: dict = (
            {}
        )  # {"gpt-4": {"daily_data": [], "sum_api_requests": 0, "sum_total_tokens": 0}}

        for row in db_response:
            _model = row["api_base"]
            if _model not in model_ui_data:
                model_ui_data[_model] = {
                    "daily_data": [],
                    "sum_num_rate_limit_exceptions": 0,
                }
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            model_ui_data[_model]["daily_data"].append(row)
            model_ui_data[_model]["sum_num_rate_limit_exceptions"] += row.get(
                "num_rate_limit_exceptions", 0
            )

        # sort mode ui data by sum_api_requests -> get top 10 models
        model_ui_data = dict(
            sorted(
                model_ui_data.items(),
                key=lambda x: x[1]["sum_num_rate_limit_exceptions"],
                reverse=True,
            )[:10]
        )

        response = []
        for model, data in model_ui_data.items():
            _sort_daily_data = sorted(data["daily_data"], key=lambda x: x["date"])

            response.append(
                {
                    "api_base": model,
                    "daily_data": _sort_daily_data,
                    "sum_num_rate_limit_exceptions": data[
                        "sum_num_rate_limit_exceptions"
                    ],
                }
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e)},
        )


@router.get(
    "/global/activity/exceptions",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
    include_in_schema=False,
)
async def get_global_activity_exceptions(
    model_group: str = fastapi.Query(
        description="Filter by model group",
    ),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
):
    """
    Get number of API Requests, total tokens through proxy

    {
        "daily_data": [
                const chartdata = [
                {
                date: 'Jan 22',
                num_rate_limit_exceptions: 10,
                },
                {
                date: 'Jan 23',
                num_rate_limit_exceptions: 10,
                },
        ],
        "sum_api_exceptions": 20,
    }
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import llm_router, prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT
            date_trunc('day', "startTime")::date AS date,
            COUNT(*) AS num_rate_limit_exceptions
        FROM
            "LiteLLM_ErrorLogs"
        WHERE
            "startTime" >= $1::date
            AND "startTime" < ($2::date + INTERVAL '1 day')
            AND model_group = $3
            AND status_code = '429'
        GROUP BY
            date_trunc('day', "startTime")
        ORDER BY
            date;
        """
        db_response = await prisma_client.db.query_raw(
            sql_query, start_date_obj, end_date_obj, model_group
        )

        if db_response is None:
            return []

        sum_num_rate_limit_exceptions = 0
        daily_data = []
        for row in db_response:
            # cast date to datetime
            _date_obj = datetime.fromisoformat(row["date"])
            row["date"] = _date_obj.strftime("%b %d")

            daily_data.append(row)
            sum_num_rate_limit_exceptions += row.get("num_rate_limit_exceptions", 0)

        # sort daily_data by date
        daily_data = sorted(daily_data, key=lambda x: x["date"])

        data_to_return = {
            "daily_data": daily_data,
            "sum_num_rate_limit_exceptions": sum_num_rate_limit_exceptions,
        }

        return data_to_return

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/spend/provider",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_spend_provider(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get breakdown of spend per provider
    [
        {
            "provider": "Azure OpenAI",
            "spend": 20
        },
        {
            "provider": "OpenAI",
            "spend": 10
        },
        {
            "provider": "VertexAI",
            "spend": 30
        }
    ]
    """
    from collections import defaultdict

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import llm_router, prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if (
            user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
            or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        ):
            user_id = user_api_key_dict.user_id
            if user_id is None:
                raise HTTPException(
                    status_code=400, detail={"error": "No user_id found"}
                )

            sql_query = """
            SELECT
            model_id,
            SUM(spend) AS spend
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" BETWEEN $1::date AND $2::date 
            AND length(model_id) > 0
            AND "user" = $3
            GROUP BY model_id
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj, user_id
            )
        else:
            sql_query = """
            SELECT
            model_id,
            SUM(spend) AS spend
            FROM "LiteLLM_SpendLogs"
            WHERE "startTime" BETWEEN $1::date AND $2::date AND length(model_id) > 0
            GROUP BY model_id
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj
            )

        if db_response is None:
            return []

        ###################################
        # Convert model_id -> to Provider #
        ###################################

        # we use the in memory router for this
        ui_response = []
        provider_spend_mapping: defaultdict = defaultdict(int)
        for row in db_response:
            _model_id = row["model_id"]
            _provider = "Unknown"
            if llm_router is not None:
                _deployment = llm_router.get_deployment(model_id=_model_id)
                if _deployment is not None:
                    try:
                        _, _provider, _, _ = litellm.get_llm_provider(
                            model=_deployment.litellm_params.model,
                            custom_llm_provider=_deployment.litellm_params.custom_llm_provider,
                            api_base=_deployment.litellm_params.api_base,
                            litellm_params=_deployment.litellm_params,
                        )
                        provider_spend_mapping[_provider] += row["spend"]
                    except Exception:
                        pass

        for provider, spend in provider_spend_mapping.items():
            ui_response.append({"provider": provider, "spend": spend})

        return ui_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/spend/report",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def get_global_spend_report(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view spend",
    ),
    group_by: Optional[Literal["team", "customer", "api_key"]] = fastapi.Query(
        default="team",
        description="Group spend by internal team or customer or api_key",
    ),
    api_key: Optional[str] = fastapi.Query(
        default=None,
        description="View spend for a specific api_key. Example api_key='sk-1234",
    ),
    internal_user_id: Optional[str] = fastapi.Query(
        default=None,
        description="View spend for a specific internal_user_id. Example internal_user_id='1234",
    ),
):
    """
    Get Daily Spend per Team, based on specific startTime and endTime. Per team, view usage by each key, model
    [
        {
            "group-by-day": "2024-05-10",
            "teams": [
                {
                    "team_name": "team-1"
                    "spend": 10,
                    "keys": [
                        "key": "1213",
                        "usage": {
                            "model-1": {
                                    "cost": 12.50,
                                    "input_tokens": 1000,
                                    "output_tokens": 5000,
                                    "requests": 100
                                },
                                "audio-modelname1": {
                                "cost": 25.50,
                                "seconds": 25,
                                "requests": 50
                        },
                        }
                    }
            ]
        ]
    }
    """
    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide start_date and end_date"},
        )

    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    from litellm.proxy.proxy_server import premium_user, prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if premium_user is not True:
            verbose_proxy_logger.debug("accessing /spend/report but not a premium user")
            raise ValueError(
                "/spend/report endpoint " + CommonProxyErrors.not_premium_user.value
            )
        if api_key is not None:
            verbose_proxy_logger.debug("Getting /spend for api_key: %s", api_key)
            if api_key.startswith("sk-"):
                api_key = hash_token(token=api_key)
            sql_query = """
                WITH SpendByModelApiKey AS (
                    SELECT
                        sl.api_key,
                        sl.model,
                        SUM(sl.spend) AS model_cost,
                        SUM(sl.prompt_tokens) AS model_input_tokens,
                        SUM(sl.completion_tokens) AS model_output_tokens
                    FROM
                        "LiteLLM_SpendLogs" sl
                    WHERE
                        sl."startTime" BETWEEN $1::date AND $2::date AND sl.api_key = $3
                    GROUP BY
                        sl.api_key,
                        sl.model
                )
                SELECT
                    api_key,
                    SUM(model_cost) AS total_cost,
                    SUM(model_input_tokens) AS total_input_tokens,
                    SUM(model_output_tokens) AS total_output_tokens,
                    jsonb_agg(jsonb_build_object(
                        'model', model,
                        'total_cost', model_cost,
                        'total_input_tokens', model_input_tokens,
                        'total_output_tokens', model_output_tokens
                    )) AS model_details
                FROM
                    SpendByModelApiKey
                GROUP BY
                    api_key
                ORDER BY
                    total_cost DESC;
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj, api_key
            )
            if db_response is None:
                return []

            return db_response
        elif internal_user_id is not None:
            verbose_proxy_logger.debug(
                "Getting /spend for internal_user_id: %s", internal_user_id
            )
            sql_query = """
                WITH SpendByModelApiKey AS (
                    SELECT
                        sl.api_key,
                        sl.model,
                        SUM(sl.spend) AS model_cost,
                        SUM(sl.prompt_tokens) AS model_input_tokens,
                        SUM(sl.completion_tokens) AS model_output_tokens
                    FROM
                        "LiteLLM_SpendLogs" sl
                    WHERE
                        sl."startTime" BETWEEN $1::date AND $2::date AND sl.user = $3
                    GROUP BY
                        sl.api_key,
                        sl.model
                )
                SELECT
                    api_key,
                    SUM(model_cost) AS total_cost,
                    SUM(model_input_tokens) AS total_input_tokens,
                    SUM(model_output_tokens) AS total_output_tokens,
                    jsonb_agg(jsonb_build_object(
                        'model', model,
                        'total_cost', model_cost,
                        'total_input_tokens', model_input_tokens,
                        'total_output_tokens', model_output_tokens
                    )) AS model_details
                FROM
                    SpendByModelApiKey
                GROUP BY
                    api_key
                ORDER BY
                    total_cost DESC;
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj, internal_user_id
            )
            if db_response is None:
                return []

            return db_response

        if group_by == "team":
            # first get data from spend logs -> SpendByModelApiKey
            # then read data from "SpendByModelApiKey" to format the response obj
            sql_query = """

            WITH SpendByModelApiKey AS (
                SELECT
                    date_trunc('day', sl."startTime") AS group_by_day,
                    COALESCE(tt.team_alias, 'Unassigned Team') AS team_name,
                    sl.model,
                    sl.api_key,
                    SUM(sl.spend) AS model_api_spend,
                    SUM(sl.total_tokens) AS model_api_tokens
                FROM 
                    "LiteLLM_SpendLogs" sl
                LEFT JOIN 
                    "LiteLLM_TeamTable" tt 
                ON 
                    sl.team_id = tt.team_id
                WHERE
                    sl."startTime" BETWEEN $1::date AND $2::date
                GROUP BY
                    date_trunc('day', sl."startTime"),
                    tt.team_alias,
                    sl.model,
                    sl.api_key
            )
                SELECT
                    group_by_day,
                    jsonb_agg(jsonb_build_object(
                        'team_name', team_name,
                        'total_spend', total_spend,
                        'metadata', metadata
                    )) AS teams
                FROM (
                    SELECT
                        group_by_day,
                        team_name,
                        SUM(model_api_spend) AS total_spend,
                        jsonb_agg(jsonb_build_object(
                            'model', model,
                            'api_key', api_key,
                            'spend', model_api_spend,
                            'total_tokens', model_api_tokens
                        )) AS metadata
                    FROM 
                        SpendByModelApiKey
                    GROUP BY
                        group_by_day,
                        team_name
                ) AS aggregated
                GROUP BY
                    group_by_day
                ORDER BY
                    group_by_day;
                """

            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj
            )
            if db_response is None:
                return []

            return db_response

        elif group_by == "customer":
            sql_query = """

            WITH SpendByModelApiKey AS (
                SELECT
                    date_trunc('day', sl."startTime") AS group_by_day,
                    sl.end_user AS customer,
                    sl.model,
                    sl.api_key,
                    SUM(sl.spend) AS model_api_spend,
                    SUM(sl.total_tokens) AS model_api_tokens
                FROM
                    "LiteLLM_SpendLogs" sl
                WHERE
                    sl."startTime" BETWEEN $1::date AND $2::date
                GROUP BY
                    date_trunc('day', sl."startTime"),
                    customer,
                    sl.model,
                    sl.api_key
            )
            SELECT
                group_by_day,
                jsonb_agg(jsonb_build_object(
                    'customer', customer,
                    'total_spend', total_spend,
                    'metadata', metadata
                )) AS customers
            FROM
                (
                    SELECT
                        group_by_day,
                        customer,
                        SUM(model_api_spend) AS total_spend,
                        jsonb_agg(jsonb_build_object(
                            'model', model,
                            'api_key', api_key,
                            'spend', model_api_spend,
                            'total_tokens', model_api_tokens
                        )) AS metadata
                    FROM
                        SpendByModelApiKey
                    GROUP BY
                        group_by_day,
                        customer
                ) AS aggregated
            GROUP BY
                group_by_day
            ORDER BY
                group_by_day;
                """

            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj
            )
            if db_response is None:
                return []

            return db_response
        elif group_by == "api_key":
            sql_query = """
                WITH SpendByModelApiKey AS (
                    SELECT
                        sl.api_key,
                        sl.model,
                        SUM(sl.spend) AS model_cost,
                        SUM(sl.prompt_tokens) AS model_input_tokens,
                        SUM(sl.completion_tokens) AS model_output_tokens
                    FROM
                        "LiteLLM_SpendLogs" sl
                    WHERE
                        sl."startTime" BETWEEN $1::date AND $2::date
                    GROUP BY
                        sl.api_key,
                        sl.model
                )
                SELECT
                    api_key,
                    SUM(model_cost) AS total_cost,
                    SUM(model_input_tokens) AS total_input_tokens,
                    SUM(model_output_tokens) AS total_output_tokens,
                    jsonb_agg(jsonb_build_object(
                        'model', model,
                        'total_cost', model_cost,
                        'total_input_tokens', model_input_tokens,
                        'total_output_tokens', model_output_tokens
                    )) AS model_details
                FROM
                    SpendByModelApiKey
                GROUP BY
                    api_key
                ORDER BY
                    total_cost DESC;
            """
            db_response = await prisma_client.db.query_raw(
                sql_query, start_date_obj, end_date_obj
            )
            if db_response is None:
                return []

            return db_response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)},
        )


@router.get(
    "/global/spend/all_tag_names",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def global_get_all_tag_names():
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        sql_query = """
        SELECT DISTINCT
            jsonb_array_elements_text(request_tags) AS individual_request_tag
        FROM "LiteLLM_SpendLogs";
        """

        db_response = await prisma_client.db.query_raw(sql_query)
        if db_response is None:
            return []

        _tag_names = []
        for row in db_response:
            _tag_names.append(row.get("individual_request_tag"))

        return {"tag_names": _tag_names}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/all_tag_names Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/all_tag_names Error" + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/global/spend/tags",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def global_view_spend_tags(
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing key spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view key spend",
    ),
    tags: Optional[str] = fastapi.Query(
        default=None,
        description="comman separated tags to filter on",
    ),
):
    """
    LiteLLM Enterprise - View Spend Per Request Tag. Used by LiteLLM UI

    Example Request:
    ```
    curl -X GET "http://0.0.0.0:4000/spend/tags" \
-H "Authorization: Bearer sk-1234"
    ```

    Spend with Start Date and End Date
    ```
    curl -X GET "http://0.0.0.0:4000/spend/tags?start_date=2022-01-01&end_date=2022-02-01" \
-H "Authorization: Bearer sk-1234"
    ```
    """
    import traceback

    from enterprise.utils import ui_get_spend_by_tags
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )

        if end_date is None or start_date is None:
            raise ProxyException(
                message="Please provide start_date and end_date",
                type="bad_request",
                param=None,
                code=status.HTTP_400_BAD_REQUEST,
            )
        response = await ui_get_spend_by_tags(
            start_date=start_date,
            end_date=end_date,
            tags_str=tags,
            prisma_client=prisma_client,
        )

        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        error_str = str(e) + "\n" + error_trace
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/tags Error({error_str})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/tags Error" + error_str,
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def _get_spend_report_for_time_range(
    start_date: str,
    end_date: str,
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        verbose_proxy_logger.error(
            "Database not connected. Connect a database to your proxy for weekly, monthly spend reports"
        )
        return None

    try:
        sql_query = """
        SELECT
            t.team_alias,
            SUM(s.spend) AS total_spend
        FROM
            "LiteLLM_SpendLogs" s
        LEFT JOIN
            "LiteLLM_TeamTable" t ON s.team_id = t.team_id
        WHERE
            s."startTime"::DATE >= $1::date AND s."startTime"::DATE <= $2::date
        GROUP BY
            t.team_alias
        ORDER BY
            total_spend DESC;
        """
        response = await prisma_client.db.query_raw(sql_query, start_date, end_date)

        # get spend per tag for today
        sql_query = """
        SELECT 
        jsonb_array_elements_text(request_tags) AS individual_request_tag,
        SUM(spend) AS total_spend
        FROM "LiteLLM_SpendLogs"
        WHERE "startTime"::DATE >= $1::date AND "startTime"::DATE <= $2::date
        GROUP BY individual_request_tag
        ORDER BY total_spend DESC;
        """

        spend_per_tag = await prisma_client.db.query_raw(
            sql_query, start_date, end_date
        )

        return response, spend_per_tag
    except Exception as e:
        verbose_proxy_logger.error(
            "Exception in _get_daily_spend_reports {}".format(str(e))
        )


@router.post(
    "/spend/calculate",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {
            "cost": {
                "description": "The calculated cost",
                "example": 0.0,
                "type": "float",
            }
        }
    },
)
async def calculate_spend(request: SpendCalculateRequest):
    """
    Accepts all the params of completion_cost.

    Calculate spend **before** making call:

    Note: If you see a spend of $0.0 you need to set custom_pricing for your model: https://docs.litellm.ai/docs/proxy/custom_pricing

    ```
    curl --location 'http://localhost:4000/spend/calculate'
    --header 'Authorization: Bearer sk-1234'
    --header 'Content-Type: application/json'
    --data '{
        "model": "anthropic.claude-v2",
        "messages": [{"role": "user", "content": "Hey, how'''s it going?"}]
    }'
    ```

    Calculate spend **after** making call:

    ```
    curl --location 'http://localhost:4000/spend/calculate'
    --header 'Authorization: Bearer sk-1234'
    --header 'Content-Type: application/json'
    --data '{
        "completion_response": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there, how may I assist you today?"
                },
                "logprobs": null,
                "finish_reason": "stop"
            }]
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
    }'
    ```
    """
    try:
        from litellm import completion_cost
        from litellm.cost_calculator import CostPerToken
        from litellm.proxy.proxy_server import llm_router

        _cost = None
        if request.model is not None:
            if request.messages is None:
                raise HTTPException(
                    status_code=400,
                    detail="Bad Request - messages must be provided if 'model' is provided",
                )

            # check if model in llm_router
            _model_in_llm_router = None
            cost_per_token: Optional[CostPerToken] = None
            if llm_router is not None:
                if (
                    llm_router.model_group_alias is not None
                    and request.model in llm_router.model_group_alias
                ):
                    # lookup alias in llm_router
                    _model_group_name = llm_router.model_group_alias[request.model]
                    for model in llm_router.model_list:
                        if model.get("model_name") == _model_group_name:
                            _model_in_llm_router = model

                else:
                    # no model_group aliases set -> try finding model in llm_router
                    # find model in llm_router
                    for model in llm_router.model_list:
                        if model.get("model_name") == request.model:
                            _model_in_llm_router = model

            """
            3 cases for /spend/calculate

            1. user passes model, and model is defined on litellm config.yaml or in DB. use info on config or in DB in this case
            2. user passes model, and model is not defined on litellm config.yaml or in DB. Pass model as is to litellm.completion_cost
            3. user passes completion_response
            
            """
            if _model_in_llm_router is not None:
                _litellm_params = _model_in_llm_router.get("litellm_params")
                _litellm_model_name = _litellm_params.get("model")
                input_cost_per_token = _litellm_params.get("input_cost_per_token")
                output_cost_per_token = _litellm_params.get("output_cost_per_token")
                if (
                    input_cost_per_token is not None
                    or output_cost_per_token is not None
                ):
                    cost_per_token = CostPerToken(
                        input_cost_per_token=input_cost_per_token,
                        output_cost_per_token=output_cost_per_token,
                    )

                _cost = completion_cost(
                    model=_litellm_model_name,
                    messages=request.messages,
                    custom_cost_per_token=cost_per_token,
                )
            else:
                _cost = completion_cost(model=request.model, messages=request.messages)
        elif request.completion_response is not None:
            _completion_response = litellm.ModelResponse(**request.completion_response)
            _cost = completion_cost(completion_response=_completion_response)
        else:
            raise HTTPException(
                status_code=400,
                detail="Bad Request - Either 'model' or 'completion_response' must be provided",
            )
        return {"cost": _cost}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


@router.get(
    "/spend/logs",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    responses={
        200: {"model": List[LiteLLM_SpendLogs]},
    },
)
async def view_spend_logs(  # noqa: PLR0915
    api_key: Optional[str] = fastapi.Query(
        default=None,
        description="Get spend logs based on api key",
    ),
    user_id: Optional[str] = fastapi.Query(
        default=None,
        description="Get spend logs based on user_id",
    ),
    request_id: Optional[str] = fastapi.Query(
        default=None,
        description="request_id to get spend logs for specific request_id. If none passed then pass spend logs for all requests",
    ),
    start_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time from which to start viewing key spend",
    ),
    end_date: Optional[str] = fastapi.Query(
        default=None,
        description="Time till which to view key spend",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    View all spend logs, if request_id is provided, only logs for that request_id will be returned

    Example Request for all logs
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Request for specific request_id
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs?request_id=chatcmpl-6dcb2540-d3d7-4e49-bb27-291f863f112e" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Request for specific api_key
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs?api_key=sk-Fn8Ej39NkBQmUagFEoUWPQ" \
-H "Authorization: Bearer sk-1234"
    ```

    Example Request for specific user_id
    ```
    curl -X GET "http://0.0.0.0:8000/spend/logs?user_id=ishaan@berri.ai" \
-H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if (
        user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
        or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    ):
        user_id = user_api_key_dict.user_id

    try:
        verbose_proxy_logger.debug("inside view_spend_logs")
        if prisma_client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        spend_logs = []
        if (
            start_date is not None
            and isinstance(start_date, str)
            and end_date is not None
            and isinstance(end_date, str)
        ):
            # Convert the date strings to datetime objects
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            filter_query = {
                "startTime": {
                    "gte": start_date_obj,  # Greater than or equal to Start Date
                    "lte": end_date_obj,  # Less than or equal to End Date
                }
            }

            if api_key is not None and isinstance(api_key, str):
                filter_query["api_key"] = api_key  # type: ignore
            elif request_id is not None and isinstance(request_id, str):
                filter_query["request_id"] = request_id  # type: ignore
            elif user_id is not None and isinstance(user_id, str):
                filter_query["user"] = user_id  # type: ignore

            # SQL query
            response = await prisma_client.db.litellm_spendlogs.group_by(
                by=["api_key", "user", "model", "startTime"],
                where=filter_query,  # type: ignore
                sum={
                    "spend": True,
                },
            )

            if (
                isinstance(response, list)
                and len(response) > 0
                and isinstance(response[0], dict)
            ):
                result: dict = {}
                for record in response:
                    dt_object = datetime.strptime(
                        str(record["startTime"]), "%Y-%m-%dT%H:%M:%S.%fZ"  # type: ignore
                    )  # type: ignore
                    date = dt_object.date()
                    if date not in result:
                        result[date] = {"users": {}, "models": {}}
                    api_key = record["api_key"]  # type: ignore
                    user_id = record["user"]  # type: ignore
                    model = record["model"]  # type: ignore
                    result[date]["spend"] = result[date].get("spend", 0) + record.get(
                        "_sum", {}
                    ).get("spend", 0)
                    result[date][api_key] = result[date].get(api_key, 0) + record.get(
                        "_sum", {}
                    ).get("spend", 0)
                    result[date]["users"][user_id] = result[date]["users"].get(
                        user_id, 0
                    ) + record.get("_sum", {}).get("spend", 0)
                    result[date]["models"][model] = result[date]["models"].get(
                        model, 0
                    ) + record.get("_sum", {}).get("spend", 0)
                return_list = []
                final_date = None
                for k, v in sorted(result.items()):
                    return_list.append({**v, "startTime": k})
                    final_date = k

                end_date_date = end_date_obj.date()
                if final_date is not None and final_date < end_date_date:
                    current_date = final_date + timedelta(days=1)
                    while current_date <= end_date_date:
                        # Represent current_date as string because original response has it this way
                        return_list.append(
                            {
                                "startTime": current_date,
                                "spend": 0,
                                "users": {},
                                "models": {},
                            }
                        )  # If no data, will stay as zero
                        current_date += timedelta(days=1)  # Move on to the next day

                return return_list

            return response

        elif api_key is not None and isinstance(api_key, str):
            if api_key.startswith("sk-"):
                hashed_token = prisma_client.hash_token(token=api_key)
            else:
                hashed_token = api_key
            spend_log = await prisma_client.get_data(
                table_name="spend",
                query_type="find_all",
                key_val={"key": "api_key", "value": hashed_token},
            )
            if isinstance(spend_log, list):
                return spend_log
            else:
                return [spend_log]
        elif request_id is not None:
            spend_log = await prisma_client.get_data(
                table_name="spend",
                query_type="find_unique",
                key_val={"key": "request_id", "value": request_id},
            )
            return [spend_log]
        elif user_id is not None:
            spend_log = await prisma_client.get_data(
                table_name="spend",
                query_type="find_all",
                key_val={"key": "user", "value": user_id},
            )
            if isinstance(spend_log, list):
                return spend_log
            else:
                return [spend_log]
        else:
            spend_logs = await prisma_client.get_data(
                table_name="spend", query_type="find_all"
            )

            return spend_logs

        return None

    except Exception as e:
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/spend/logs Error({str(e)})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/spend/logs Error" + str(e),
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post(
    "/global/spend/reset",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def global_spend_reset():
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Globally reset spend for All API Keys and Teams, maintain LiteLLM_SpendLogs

    1. LiteLLM_SpendLogs will maintain the logs on spend, no data gets deleted from there
    2. LiteLLM_VerificationTokens spend will be set = 0
    3. LiteLLM_TeamTable spend will be set = 0

    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_401_UNAUTHORIZED,
        )

    await prisma_client.db.litellm_verificationtoken.update_many(
        data={"spend": 0.0}, where={}
    )
    await prisma_client.db.litellm_teamtable.update_many(data={"spend": 0.0}, where={})

    return {
        "message": "Spend for all API Keys and Teams reset successfully",
        "status": "success",
    }


@router.post(
    "/global/spend/refresh",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend_refresh():
    """
    ADMIN ONLY / MASTER KEY Only Endpoint

    Globally refresh spend MonthlyGlobalSpend view
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_401_UNAUTHORIZED,
        )

    ## RESET GLOBAL SPEND VIEW ###
    async def is_materialized_global_spend_view() -> bool:
        """
        Return True if materialized view exists

        Else False
        """
        sql_query = """
        SELECT relname, relkind
        FROM pg_class
        WHERE relname = 'MonthlyGlobalSpend';            
        """
        try:
            resp = await prisma_client.db.query_raw(sql_query)

            assert resp[0]["relkind"] == "m"
            return True
        except Exception:
            return False

    view_exists = await is_materialized_global_spend_view()

    if view_exists:
        # refresh materialized view
        sql_query = """
        REFRESH MATERIALIZED VIEW "MonthlyGlobalSpend";    
        """
        try:
            from litellm.proxy._types import CommonProxyErrors
            from litellm.proxy.proxy_server import proxy_logging_obj
            from litellm.proxy.utils import PrismaClient

            db_url = os.getenv("DATABASE_URL")
            if db_url is None:
                raise Exception(CommonProxyErrors.db_not_connected_error.value)
            new_client = PrismaClient(
                database_url=db_url,
                proxy_logging_obj=proxy_logging_obj,
                http_client={
                    "timeout": 6000,
                },
            )
            await new_client.db.connect()
            await new_client.db.query_raw(sql_query)
            verbose_proxy_logger.info("MonthlyGlobalSpend view refreshed")
            return {
                "message": "MonthlyGlobalSpend view refreshed",
                "status": "success",
            }

        except Exception as e:
            verbose_proxy_logger.exception(
                "Failed to refresh materialized view - {}".format(str(e))
            )
            return {
                "message": "Failed to refresh materialized view",
                "status": "failure",
            }


async def global_spend_for_internal_user(
    api_key: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    try:

        user_id = user_api_key_dict.user_id
        if user_id is None:
            raise ValueError("/global/spend/logs Error: User ID is None")
        if api_key is not None:
            sql_query = """
                SELECT * FROM "MonthlyGlobalSpendPerUserPerKey"
                WHERE "api_key" = $1 AND "user" = $2
                ORDER BY "date";
                """

            response = await prisma_client.db.query_raw(sql_query, api_key, user_id)

            return response

        sql_query = """SELECT * FROM "MonthlyGlobalSpendPerUserPerKey"  WHERE "user" = $1 ORDER BY "date";"""

        response = await prisma_client.db.query_raw(sql_query, user_id)

        return response
    except Exception as e:
        verbose_proxy_logger.error(f"/global/spend/logs Error: {str(e)}")
        raise e


@router.get(
    "/global/spend/logs",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend_logs(
    api_key: Optional[str] = fastapi.Query(
        default=None,
        description="API Key to get global spend (spend per day for last 30d). Admin-only endpoint",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get global spend (spend per day for last 30d). Admin-only endpoint

    More efficient implementation of /spend/logs, by creating a view over the spend logs table.
    """
    import traceback

    from litellm.integrations.prometheus_helpers.prometheus_api import (
        get_daily_spend_from_prometheus,
        is_prometheus_connected,
    )
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise ProxyException(
                message="Prisma Client is not initialized",
                type="internal_error",
                param="None",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if (
            user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
            or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        ):
            response = await global_spend_for_internal_user(
                api_key=api_key, user_api_key_dict=user_api_key_dict
            )

            return response

        prometheus_api_enabled = is_prometheus_connected()

        if prometheus_api_enabled:
            response = await get_daily_spend_from_prometheus(api_key=api_key)
            return response
        else:
            if api_key is None:
                sql_query = """SELECT * FROM "MonthlyGlobalSpend" ORDER BY "date";"""

                response = await prisma_client.db.query_raw(query=sql_query)

                return response
            else:
                sql_query = """
                    SELECT * FROM "MonthlyGlobalSpendPerKey"
                    WHERE "api_key" = $1
                    ORDER BY "date";
                    """

                response = await prisma_client.db.query_raw(sql_query, api_key)

                return response

    except Exception as e:
        error_trace = traceback.format_exc()
        error_str = str(e) + "\n" + error_trace
        verbose_proxy_logger.error(f"/global/spend/logs Error: {error_str}")
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/global/spend/logs Error({error_str})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/global/spend/logs Error" + error_str,
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get(
    "/global/spend",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend():
    """
    [BETA] This is a beta endpoint. It will change.

    View total spend across all proxy keys
    """
    import traceback

    from litellm.proxy.proxy_server import prisma_client

    try:

        total_spend = 0.0

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No db connected"})
        sql_query = """SELECT SUM(spend) as total_spend FROM "MonthlyGlobalSpend";"""
        response = await prisma_client.db.query_raw(query=sql_query)
        if response is not None:
            if isinstance(response, list) and len(response) > 0:
                total_spend = response[0].get("total_spend", 0.0)

        return {"spend": total_spend, "max_budget": litellm.max_budget}
    except Exception as e:
        error_trace = traceback.format_exc()
        error_str = str(e) + "\n" + error_trace
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"/global/spend Error({error_str})"),
                type="internal_error",
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="/global/spend Error" + error_str,
            type="internal_error",
            param=getattr(e, "param", "None"),
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def global_spend_key_internal_user(
    user_api_key_dict: UserAPIKeyAuth, limit: int = 10
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    user_id = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(status_code=500, detail={"error": "No user_id found"})

    sql_query = """
            WITH top_api_keys AS (
            SELECT 
                api_key,
                SUM(spend) as total_spend
            FROM 
                "LiteLLM_SpendLogs"
            WHERE 
                "user" = $1
            GROUP BY 
                api_key
            ORDER BY 
                total_spend DESC
            LIMIT $2  -- Adjust this number to get more or fewer top keys
        )
        SELECT 
            t.api_key,
            t.total_spend,
            v.key_alias,
            v.key_name
        FROM 
            top_api_keys t
        LEFT JOIN 
            "LiteLLM_VerificationToken" v ON t.api_key = v.token
        ORDER BY 
            t.total_spend DESC;
    
    """

    response = await prisma_client.db.query_raw(sql_query, user_id, limit)

    return response


@router.get(
    "/global/spend/keys",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend_keys(
    limit: int = fastapi.Query(
        default=None,
        description="Number of keys to get. Will return Top 'n' keys.",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get the top 'n' keys with the highest spend, ordered by spend.
    """
    from litellm.proxy.proxy_server import prisma_client

    if (
        user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
        or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    ):
        response = await global_spend_key_internal_user(
            user_api_key_dict=user_api_key_dict
        )

        return response
    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})
    sql_query = f"""SELECT * FROM "Last30dKeysBySpend" LIMIT {limit};"""

    response = await prisma_client.db.query_raw(query=sql_query)

    return response


@router.get(
    "/global/spend/teams",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend_per_team():
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get daily spend, grouped by `team_id` and `date`
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})
    sql_query = """
        SELECT
            t.team_alias as team_alias,
            DATE(s."startTime") AS spend_date,
            SUM(s.spend) AS total_spend
        FROM
            "LiteLLM_SpendLogs" s
        LEFT JOIN
            "LiteLLM_TeamTable" t ON s.team_id = t.team_id
        WHERE
            s."startTime" >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY
            t.team_alias,
            DATE(s."startTime")
        ORDER BY
            spend_date;
        """
    response = await prisma_client.db.query_raw(query=sql_query)

    # transform the response for the Admin UI
    spend_by_date = {}
    team_aliases = set()
    total_spend_per_team = {}
    for row in response:
        row_date = row["spend_date"]
        if row_date is None:
            continue
        team_alias = row["team_alias"]
        if team_alias is None:
            team_alias = "Unassigned"
        team_aliases.add(team_alias)
        if row_date in spend_by_date:
            # get the team_id for this entry
            # get the spend for this entry
            spend = row["total_spend"]
            spend = round(spend, 2)
            current_date_entries = spend_by_date[row_date]
            current_date_entries[team_alias] = spend
        else:
            spend = row["total_spend"]
            spend = round(spend, 2)
            spend_by_date[row_date] = {team_alias: spend}

        if team_alias in total_spend_per_team:
            total_spend_per_team[team_alias] += spend
        else:
            total_spend_per_team[team_alias] = spend

    total_spend_per_team_ui = []
    # order the elements in total_spend_per_team by spend
    total_spend_per_team = dict(
        sorted(total_spend_per_team.items(), key=lambda item: item[1], reverse=True)
    )
    for team_id in total_spend_per_team:
        # only add first 10 elements to total_spend_per_team_ui
        if len(total_spend_per_team_ui) >= 10:
            break
        if team_id is None:
            team_id = "Unassigned"
        total_spend_per_team_ui.append(
            {"team_id": team_id, "total_spend": total_spend_per_team[team_id]}
        )

    # sort spend_by_date by it's key (which is a date)

    response_data = []
    for key in spend_by_date:
        value = spend_by_date[key]
        response_data.append({"date": key, **value})

    return {
        "daily_spend": response_data,
        "teams": list(team_aliases),
        "total_spend_per_team": total_spend_per_team_ui,
    }


@router.get(
    "/global/all_end_users",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_view_all_end_users():
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to just get all the unique `end_users`
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    sql_query = """
    SELECT DISTINCT end_user FROM "LiteLLM_SpendLogs"
    """

    db_response = await prisma_client.db.query_raw(query=sql_query)
    if db_response is None:
        return []

    _end_users = []
    for row in db_response:
        _end_users.append(row["end_user"])

    return {"end_users": _end_users}


@router.post(
    "/global/spend/end_users",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend_end_users(data: Optional[GlobalEndUsersSpend] = None):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get the top 'n' keys with the highest spend, ordered by spend.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    """
    Gets the top 100 end-users for a given api key
    """
    startTime = None
    endTime = None
    selected_api_key = None
    if data is not None:
        startTime = data.startTime
        endTime = data.endTime
        selected_api_key = data.api_key

    startTime = startTime or datetime.now() - timedelta(days=30)
    endTime = endTime or datetime.now()

    sql_query = """
SELECT end_user, COUNT(*) AS total_count, SUM(spend) AS total_spend
FROM "LiteLLM_SpendLogs"
WHERE "startTime" >= $1::timestamp
  AND "startTime" < $2::timestamp
  AND (
    CASE
      WHEN $3::TEXT IS NULL THEN TRUE
      ELSE api_key = $3
    END
  )
GROUP BY end_user
ORDER BY total_spend DESC
LIMIT 100
    """
    response = await prisma_client.db.query_raw(
        sql_query, startTime, endTime, selected_api_key
    )

    return response


async def global_spend_models_internal_user(
    user_api_key_dict: UserAPIKeyAuth, limit: int = 10
):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    user_id = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(status_code=500, detail={"error": "No user_id found"})

    sql_query = """
        SELECT 
            model,
            SUM(spend) as total_spend,
            SUM(total_tokens) as total_tokens
        FROM 
            "LiteLLM_SpendLogs"
        WHERE 
            "user" = $1
        GROUP BY 
            model
        ORDER BY 
            total_spend DESC
        LIMIT $2;
    """

    response = await prisma_client.db.query_raw(sql_query, user_id, limit)

    return response


@router.get(
    "/global/spend/models",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_spend_models(
    limit: int = fastapi.Query(
        default=10,
        description="Number of models to get. Will return Top 'n' models.",
    ),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [BETA] This is a beta endpoint. It will change.

    Use this to get the top 'n' models with the highest spend, ordered by spend.
    """
    from litellm.proxy.proxy_server import prisma_client

    if (
        user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER
        or user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    ):
        response = await global_spend_models_internal_user(
            user_api_key_dict=user_api_key_dict, limit=limit
        )
        return response

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    sql_query = f"""SELECT * FROM "Last30dModelsBySpend" LIMIT {limit};"""

    response = await prisma_client.db.query_raw(query=sql_query)

    return response


@router.post(
    "/global/predict/spend/logs",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def global_predict_spend_logs(request: Request):
    from enterprise.utils import _forecast_daily_cost

    data = await request.json()
    data = data.get("data")
    return _forecast_daily_cost(data)
