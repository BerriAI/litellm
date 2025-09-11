from typing import Dict, List, Literal, TypedDict, Union

from typing_extensions import NotRequired


class InputConfig(TypedDict, total=False):
    id: str
    description: str
    type: str
    password: bool


class StdioServerConfig(TypedDict):
    type: Literal["stdio"]
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: str


class HTTPServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: Dict[str, str]


class SSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: Dict[str, str]


ServerConfig = Union[StdioServerConfig, HTTPServerConfig, SSEServerConfig]


# AgentConfig root object
class AgentConfig(TypedDict):
    model: str
    provider: str
    apiKey: NotRequired[str]
    inputs: List[InputConfig]
    servers: List[ServerConfig]
