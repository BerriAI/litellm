import json
import httpx
import os

from starlette.middleware.base import BaseHTTPMiddleware
from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy._types import ProxyException


class BudServeMiddleware(BaseHTTPMiddleware):
    llm_request_list = [
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/images/generation",
        "/audio/speech",
        "/audio/transcriptions",
    ]

    async def get_api_key(self, request):
        authorization_header = request.headers.get("Authorization")
        api_key = authorization_header.split(" ")[1]
        return api_key
    
    async def fetch_user_config(self, api_key: str, endpoint_name: str):
        # redis key : router_config:{api_key}:{endpoint_name}
        budserve_app_baseurl = os.getenv('BUDSERVE_APP_BASEURL', 'http://localhost:9000')
        url = f"{budserve_app_baseurl}/credentials/router-config"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params={"api_key": api_key, "endpoint_name": endpoint_name},
                    headers={"Content-Type": "application/json"},
                    follow_redirects=True
                )
                verbose_proxy_logger.debug(f"Response: {response}")
                response_data = response.json()
                if response_data.get("success", False):
                    return response_data.get("result", None)
                else:
                    raise ProxyException(
                        message=response_data.get("message", "Error fetching user config"),
                        type="not_found",
                        param=endpoint_name,
                        code=404
                    )
        except Exception as e:
            verbose_proxy_logger.error(f"Error fetching user config from {url}: {e}")
            if isinstance(e, ProxyException):
                raise e
            else:
                raise ProxyException(
                    message=f"Error fetching user config from {url}: {e}",
                    type="internal_server_error",
                    param=endpoint_name,
                    code=500
                )

    async def dispatch(
        self,
        request,
        call_next,
    ):
        """
        Steps to prepare user_config

        1. api_key and model (endpoint_name) fetch all endpoint details : model_list
        2. Using models involved in endpoint details, fetch proprietary credentials
        3. Create user_config using model_configuration (endpoint model) and router_config (project model)
        4. Add validations for fallbacks
        """
        route: str = get_request_route(request=request)
        verbose_proxy_logger.info(f"Request: {route}")
        run_through_middleware = any(
            each_route in route for each_route in self.llm_request_list
        )
        verbose_proxy_logger.info(f"Run Through Middleware: {run_through_middleware}")
        if not run_through_middleware:
            return await call_next(request)

        # get the request body
        request_data = await _read_request_body(request=request)
        api_key = await self.get_api_key(request)
        endpoint_name = request_data.get("model")

        # get endpoint details to fill cache_params
        user_config = await self.fetch_user_config(api_key, endpoint_name)
        
        # redis connection params we will set as kubernetes env variables
        # can be fetched using os.getenv
        request_data["user_config"] = {
            "cache_responses": False if not user_config.get("cache_configuration") else True,
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": os.getenv("REDIS_PORT", 6379),
            "redis_password": os.getenv("REDIS_PASSWORD", ""),
            "endpoint_cache_settings": {
                "cache": False if not user_config.get("cache_configuration") else True,
                "type": "gpt_cache_redis",  # redis-semantic
                "cache_params": {
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": os.getenv("REDIS_PORT", 6379),
                    "password": os.getenv("REDIS_PASSWORD", ""),
                    "similarity_threshold": user_config  \
                        .get("cache_configuration", {})  \
                        .get("score_threshold") 
                        if user_config.get("cache_configuration") 
                        else os.getenv("CACHE_SCORE_THRESHOLD"),
                    "redis_semantic_cache_use_async": False,
                    "redis_semantic_cache_embedding_model": user_config  \
                        .get("cache_configuration", {})  \
                        .get("embedding_model") 
                        if user_config.get("cache_configuration") 
                        else os.getenv("CACHE_EMBEDDING_MODEL"),
                    "eviction_policy": {
                        "policy": user_config  \
                            .get("cache_configuration", {})  \
                            .get("eviction_policy")
                            if user_config.get("cache_configuration")
                            else os.getenv("CACHE_EVICTION_POLICY"),
                        "max_size": user_config  \
                            .get("cache_configuration", {})  \
                            .get("max_size")
                            if user_config.get("cache_configuration")
                            else os.getenv("CACHE_MAX_SIZE"),
                        "ttl": user_config  \
                            .get("cache_configuration", {})  \
                            .get("ttl")
                            if user_config.get("cache_configuration")
                            else os.getenv("CACHE_TTL")
                    },
                },
            },
            "model_list": [user_config.get("model_configuration", {})],
        }
        request._body = json.dumps(request_data).encode("utf-8")
        return await call_next(request)
