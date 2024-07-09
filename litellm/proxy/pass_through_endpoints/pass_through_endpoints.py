import ast
import traceback
from base64 import b64encode

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import StreamingResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

async_client = httpx.AsyncClient()


async def set_env_variables_in_header(custom_headers: dict):
    """
    checks if nay headers on config.yaml are defined as os.environ/COHERE_API_KEY etc

    only runs for headers defined on config.yaml

    example header can be

    {"Authorization": "bearer os.environ/COHERE_API_KEY"}
    """
    headers = {}
    for key, value in custom_headers.items():
        # langfuse Api requires base64 encoded headers - it's simpleer to just ask litellm users to set their langfuse public and secret keys
        # we can then get the b64 encoded keys here
        if key == "LANGFUSE_PUBLIC_KEY" or key == "LANGFUSE_SECRET_KEY":
            # langfuse requires b64 encoded headers - we construct that here
            _langfuse_public_key = custom_headers["LANGFUSE_PUBLIC_KEY"]
            _langfuse_secret_key = custom_headers["LANGFUSE_SECRET_KEY"]
            if isinstance(
                _langfuse_public_key, str
            ) and _langfuse_public_key.startswith("os.environ/"):
                _langfuse_public_key = litellm.get_secret(_langfuse_public_key)
            if isinstance(
                _langfuse_secret_key, str
            ) and _langfuse_secret_key.startswith("os.environ/"):
                _langfuse_secret_key = litellm.get_secret(_langfuse_secret_key)
            headers["Authorization"] = "Basic " + b64encode(
                f"{_langfuse_public_key}:{_langfuse_secret_key}".encode("utf-8")
            ).decode("ascii")
        else:
            # for all other headers
            headers[key] = value
            if isinstance(value, str) and "os.environ/" in value:
                verbose_proxy_logger.debug(
                    "pass through endpoint - looking up 'os.environ/' variable"
                )
                # get string section that is os.environ/
                start_index = value.find("os.environ/")
                _variable_name = value[start_index:]

                verbose_proxy_logger.debug(
                    "pass through endpoint - getting secret for variable name: %s",
                    _variable_name,
                )
                _secret_value = litellm.get_secret(_variable_name)
                new_value = value.replace(_variable_name, _secret_value)
                headers[key] = new_value
    return headers


async def pass_through_request(request: Request, target: str, custom_headers: dict):
    try:

        url = httpx.URL(target)
        headers = custom_headers

        request_body = await request.body()
        _parsed_body = ast.literal_eval(request_body.decode("utf-8"))

        verbose_proxy_logger.debug(
            "Pass through endpoint sending request to \nURL {}\nheaders: {}\nbody: {}\n".format(
                url, headers, _parsed_body
            )
        )

        response = await async_client.request(
            method=request.method,
            url=url,
            headers=headers,
            params=request.query_params,
            json=_parsed_body,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        content = await response.aread()
        return Response(
            content=content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except Exception as e:
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.pass through endpoint(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


def create_pass_through_route(endpoint, target, custom_headers=None):
    async def endpoint_func(request: Request):
        return await pass_through_request(request, target, custom_headers)

    return endpoint_func


async def initialize_pass_through_endpoints(pass_through_endpoints: list):

    verbose_proxy_logger.debug("initializing pass through endpoints")
    from litellm.proxy._types import CommonProxyErrors, LiteLLMRoutes
    from litellm.proxy.proxy_server import app, premium_user

    for endpoint in pass_through_endpoints:
        _target = endpoint.get("target", None)
        _path = endpoint.get("path", None)
        _custom_headers = endpoint.get("headers", None)
        _custom_headers = await set_env_variables_in_header(
            custom_headers=_custom_headers
        )
        _auth = endpoint.get("auth", None)
        _dependencies = None
        if _auth is not None and str(_auth).lower() == "true":
            if premium_user is not True:
                raise ValueError(
                    f"Error Setting Authentication on Pass Through Endpoint: {CommonProxyErrors.not_premium_user}"
                )
            _dependencies = [Depends(user_api_key_auth)]
            LiteLLMRoutes.openai_routes.value.append(_path)

        if _target is None:
            continue

        verbose_proxy_logger.debug("adding pass through endpoint: %s", _path)

        app.add_api_route(
            path=_path,
            endpoint=create_pass_through_route(_path, _target, _custom_headers),
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            dependencies=_dependencies,
        )

        verbose_proxy_logger.debug("Added new pass through endpoint: %s", _path)
