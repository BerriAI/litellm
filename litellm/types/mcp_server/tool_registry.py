from typing import Any, Callable, Dict

from pydantic import BaseModel


class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
