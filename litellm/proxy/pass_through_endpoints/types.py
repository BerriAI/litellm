from enum import Enum


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    GENERIC = "generic"
