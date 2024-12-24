import os
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.auth_utils import get_request_route, pre_db_read_auth_checks
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body


async def fetch_data(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"Content-Type": "application/json"},
            follow_redirects=True
        )
        return response


budserve_app_baseurl = os.getenv("BUDSERVE_APP_BASEURL", "http://localhost:9000")


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    """
    Custom Auth dependency for User API Key Authentication
    We receive budserve ap key and check if it is valid

    Steps:

    1. Check api-key in cache
    2. Get api-key details from db
    3. Check expiry
    4. Check budget
    5. Check model budget
    """
    try:
        from litellm.proxy.proxy_server import (
            prisma_client,
            user_api_key_cache,
        )

        if prisma_client is None:
            raise Exception("Prisma client not initialized")

        api_key = f"sk-{api_key}"

        route: str = get_request_route(request=request)
        # get the request body
        request_data = await _read_request_body(request=request)
        await pre_db_read_auth_checks(
            request_data=request_data,
            request=request,
            route=route,
        )

        # look for info is user_api_key_auth cache
        verbose_proxy_logger.debug(f"API key sent in request >>> {api_key}")
        hashed_token = hash_token(api_key)
        valid_token: Optional[UserAPIKeyAuth] = (
            await user_api_key_cache.async_get_cache(key=hashed_token)
        )

        verbose_proxy_logger.info(
            f"Valid token from cache for key : {hashed_token} >>> {valid_token}"
        )
        if valid_token is None:
            # getting token details from authentication service
            url = f"{budserve_app_baseurl}/credentials/details/{api_key.removeprefix('sk-')}"
            credential_details_response = await fetch_data(url)
            verbose_proxy_logger.debug(f"Credential details response >>> {credential_details_response}")
            if credential_details_response.status_code != 200:
                # No token was found when looking up in the DB
                raise Exception("Invalid api key passed")
            credential_dict = credential_details_response.json()["result"]
            # credential_dict = {
            #     "key": api_key.removeprefix("sk-"),
            #     "expiry": (datetime.now() + timedelta(days=1)).strftime(
            #         "%Y-%m-%d %H:%M:%S"
            #     ),
            #     "max_budget": 1,
            #     "model_budgets": {"gpt-4": 0.003, "gpt-3.5-turbo": 0.002},
            # }
            valid_token = UserAPIKeyAuth(
                api_key=f"sk-{credential_dict['key']}",
                expires=credential_dict["expiry"],
                max_budget=credential_dict["max_budget"],
                model_max_budget=credential_dict["model_budgets"] or {},
            )
            api_key_spend = await prisma_client.db.litellm_spendlogs.group_by(
                by=["api_key"],
                sum={"spend": True},
                where={
                    "AND": [
                        {"api_key": valid_token.token},
                    ]
                },  # type: ignore
            )
            if (
                len(api_key_spend) > 0
                and "_sum" in api_key_spend[0]
                and "spend" in api_key_spend[0]["_sum"]
                and api_key_spend[0]["_sum"]["spend"]
            ):
                valid_token.spend = api_key_spend[0]["_sum"]["spend"]
            # Add hashed token to cache
            verbose_proxy_logger.info(
                f"Valid token storing in cache for key : {valid_token.token}"
            )
            await user_api_key_cache.async_set_cache(
                key=valid_token.token,
                value=valid_token,
            )
            verbose_proxy_logger.info(f"Valid token from DB >>> {valid_token}")
        verbose_proxy_logger.info(f"Valid token spend >> {valid_token.spend}")
        if valid_token is not None:
            if valid_token.expires is not None:
                current_time = datetime.now(timezone.utc)
                expiry_time = datetime.fromisoformat(valid_token.expires)
                if (
                    expiry_time.tzinfo is None
                    or expiry_time.tzinfo.utcoffset(expiry_time) is None
                ):
                    expiry_time = expiry_time.replace(tzinfo=timezone.utc)
                verbose_proxy_logger.debug(
                    f"Checking if token expired, expiry time {expiry_time} and current time {current_time}"
                )
                if expiry_time < current_time:
                    # Token exists but is expired.
                    raise ProxyException(
                        message=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                        type=ProxyErrorTypes.expired_key,
                        code=400,
                        param=api_key,
                    )
            if valid_token.spend is not None and valid_token.max_budget is not None:
                if valid_token.spend >= valid_token.max_budget:
                    raise litellm.BudgetExceededError(
                        current_cost=valid_token.spend,
                        max_budget=valid_token.max_budget,
                    )
                max_budget_per_model = valid_token.model_max_budget
                current_model = request_data.get("model", None)
                if (
                    max_budget_per_model is not None
                    and isinstance(max_budget_per_model, dict)
                    and len(max_budget_per_model) > 0
                    and prisma_client is not None
                    and current_model is not None
                    and valid_token.token is not None
                ):
                    ## GET THE SPEND FOR THIS MODEL
                    twenty_eight_days_ago = datetime.now() - timedelta(days=28)
                    model_spend = await prisma_client.db.litellm_spendlogs.group_by(
                        by=["model"],
                        sum={"spend": True},
                        where={
                            "AND": [
                                {"api_key": valid_token.token},
                                {"startTime": {"gt": twenty_eight_days_ago}},
                                {"model": current_model},
                            ]
                        },  # type: ignore
                    )
                    verbose_proxy_logger.debug(f"model spends >> {model_spend}")
                    if (
                        len(model_spend) > 0
                        and max_budget_per_model.get(current_model, None) is not None
                    ):
                        if (
                            "model" in model_spend[0]
                            and model_spend[0].get("model") == current_model
                            and "_sum" in model_spend[0]
                            and "spend" in model_spend[0]["_sum"]
                            and model_spend[0]["_sum"]["spend"]
                            >= max_budget_per_model[current_model]
                        ):
                            current_model_spend = model_spend[0]["_sum"]["spend"]
                            current_model_budget = max_budget_per_model[current_model]
                            raise litellm.BudgetExceededError(
                                current_cost=current_model_spend,
                                max_budget=current_model_budget,
                            )
            return valid_token
        else:
            # No token was found when looking up in the DB
            raise Exception("Invalid api key passed")

    except Exception as e:
        if isinstance(e, litellm.BudgetExceededError):
            raise ProxyException(
                message=e.message,
                type=ProxyErrorTypes.budget_exceeded,
                param=None,
                code=400,
            )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_401_UNAUTHORIZED,
        )
