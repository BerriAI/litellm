import time, json, httpx, asyncio


class AsyncCustomHTTPTransport(httpx.AsyncHTTPTransport):
    """
    Async implementation of custom http transport
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if "images/generations" in request.url.path and request.url.params[
            "api-version"
        ] in [  # dall-e-3 starts from `2023-12-01-preview` so we should be able to avoid conflict
            "2023-06-01-preview",
            "2023-07-01-preview",
            "2023-08-01-preview",
            "2023-09-01-preview",
            "2023-10-01-preview",
        ]:
            request.url = request.url.copy_with(
                path="/openai/images/generations:submit"
            )
            response = await super().handle_async_request(request)
            operation_location_url = response.headers["operation-location"]
            request.url = httpx.URL(operation_location_url)
            request.method = "GET"
            response = await super().handle_async_request(request)
            await response.aread()

            timeout_secs: int = 120
            start_time = time.time()
            while response.json()["status"] not in ["succeeded", "failed"]:
                if time.time() - start_time > timeout_secs:
                    timeout = {
                        "error": {
                            "code": "Timeout",
                            "message": "Operation polling timed out.",
                        }
                    }
                    return httpx.Response(
                        status_code=400,
                        headers=response.headers,
                        content=json.dumps(timeout).encode("utf-8"),
                        request=request,
                    )

                await asyncio.sleep(int(response.headers.get("retry-after") or 10))
                response = await super().handle_async_request(request)
                await response.aread()

            if response.json()["status"] == "failed":
                error_data = response.json()
                return httpx.Response(
                    status_code=400,
                    headers=response.headers,
                    content=json.dumps(error_data).encode("utf-8"),
                    request=request,
                )

            result = response.json()["result"]
            return httpx.Response(
                status_code=200,
                headers=response.headers,
                content=json.dumps(result).encode("utf-8"),
                request=request,
            )
        return await super().handle_async_request(request)


class CustomHTTPTransport(httpx.HTTPTransport):
    """
    This class was written as a workaround to support dall-e-2 on openai > v1.x

    Refer to this issue for more: https://github.com/openai/openai-python/issues/692
    """

    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        if "images/generations" in request.url.path and request.url.params[
            "api-version"
        ] in [  # dall-e-3 starts from `2023-12-01-preview` so we should be able to avoid conflict
            "2023-06-01-preview",
            "2023-07-01-preview",
            "2023-08-01-preview",
            "2023-09-01-preview",
            "2023-10-01-preview",
        ]:
            request.url = request.url.copy_with(
                path="/openai/images/generations:submit"
            )
            response = super().handle_request(request)
            operation_location_url = response.headers["operation-location"]
            request.url = httpx.URL(operation_location_url)
            request.method = "GET"
            response = super().handle_request(request)
            response.read()
            timeout_secs: int = 120
            start_time = time.time()
            while response.json()["status"] not in ["succeeded", "failed"]:
                if time.time() - start_time > timeout_secs:
                    timeout = {
                        "error": {
                            "code": "Timeout",
                            "message": "Operation polling timed out.",
                        }
                    }
                    return httpx.Response(
                        status_code=400,
                        headers=response.headers,
                        content=json.dumps(timeout).encode("utf-8"),
                        request=request,
                    )
                time.sleep(int(response.headers.get("retry-after", None) or 10))
                response = super().handle_request(request)
                response.read()
            if response.json()["status"] == "failed":
                error_data = response.json()
                return httpx.Response(
                    status_code=400,
                    headers=response.headers,
                    content=json.dumps(error_data).encode("utf-8"),
                    request=request,
                )

            result = response.json()["result"]
            return httpx.Response(
                status_code=200,
                headers=response.headers,
                content=json.dumps(result).encode("utf-8"),
                request=request,
            )
        return super().handle_request(request)
