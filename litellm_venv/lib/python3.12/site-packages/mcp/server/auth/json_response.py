from typing import Any

from starlette.responses import JSONResponse


class PydanticJSONResponse(JSONResponse):
    # use pydantic json serialization instead of the stock `json.dumps`,
    # so that we can handle serializing pydantic models like AnyHttpUrl
    def render(self, content: Any) -> bytes:
        return content.model_dump_json(exclude_none=True).encode("utf-8")
