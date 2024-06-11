from typing import Optional
from fastapi import Request
import ast, json


async def _read_request_body(request: Optional[Request]) -> dict:
    """
    Asynchronous function to read the request body and parse it as JSON or literal data.

    Parameters:
    - request: The request object to read the body from

    Returns:
    - dict: Parsed request data as a dictionary
    """
    try:
        request_data: dict = {}
        if request is None:
            return request_data
        body = await request.body()

        if body == b"" or body is None:
            return request_data
        body_str = body.decode()
        try:
            request_data = ast.literal_eval(body_str)
        except:
            request_data = json.loads(body_str)
        return request_data
    except:
        return {}
