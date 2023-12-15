from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Type, Any
from typing_extensions import TypedDict, Literal

from pydantic import Field

from .models import BaseModel
from .._compat import model_json


log: logging.Logger = logging.getLogger(__name__)


__all__ = ('Manifest',)


class Request(BaseModel):
    # JSON RPC protocol version
    jsonrpc: str = '2.0'

    # identifies a request
    id: int

    # request intention
    method: str

    # request payload
    params: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    id: int
    jsonrpc: str = '2.0'
    result: Optional[Dict[str, Any]] = None


class ErrorData(TypedDict):
    code: int
    message: str
    data: object


class ErrorResponse(BaseModel):
    id: int
    error: ErrorData
    jsonrpc: str = '2.0'


Response = Union[SuccessResponse, ErrorResponse]

EngineType = Literal[
    'prismaFmt',
    'queryEngine',
    'libqueryEngine',
    'migrationEngine',
    'introspectionEngine',
]


class Manifest(BaseModel):
    """Generator metadata"""

    prettyName: str = Field(alias='name')
    defaultOutput: Union[str, Path] = Field(alias='default_output')
    denylist: Optional[List[str]] = None
    requiresEngines: Optional[List[EngineType]] = Field(
        alias='requires_engines', default=None
    )
    requiresGenerators: Optional[List[str]] = Field(
        alias='requires_generators', default=None
    )


# TODO: proper types
method_mapping: Dict[str, Type[Request]] = {
    'getManifest': Request,
    'generate': Request,
}


def readline() -> Optional[str]:
    try:
        line = input()
    except EOFError:
        log.debug('Ignoring EOFError')
        return None

    return line


def parse(line: str) -> Request:
    data = json.loads(line)
    try:
        method = data['method']
    except (KeyError, TypeError):
        # TODO
        raise
    else:
        request_type = method_mapping.get(method)
        if request_type is None:
            raise RuntimeError(f'Unknown method: {method}')

    return request_type(**data)


def reply(response: Response) -> None:
    dumped = model_json(response) + '\n'
    print(dumped, file=sys.stderr, flush=True)
    log.debug('Replied with %s', dumped)
