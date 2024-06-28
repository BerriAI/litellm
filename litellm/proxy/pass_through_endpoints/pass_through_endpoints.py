import ast
import traceback

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException

async_client = httpx.AsyncClient()


async def pass_through_request(request: Request, target: str, custom_headers: dict):
    try:

        url = httpx.URL(target)

        # Start with the original request headers
        headers = custom_headers
        # headers = dict(request.headers)

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
    from litellm.proxy.proxy_server import app

    for endpoint in pass_through_endpoints:
        _target = endpoint.get("target", None)
        _path = endpoint.get("path", None)
        _custom_headers = endpoint.get("headers", None)

        if _target is None:
            continue

        verbose_proxy_logger.debug("adding pass through endpoint: %s", _path)

        app.add_api_route(
            path=_path,
            endpoint=create_pass_through_route(_path, _target, _custom_headers),
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )

        verbose_proxy_logger.debug("Added new pass through endpoint: %s", _path)
