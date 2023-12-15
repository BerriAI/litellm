from typing import Any

from ..errors import PrismaError
from ..http_abstract import AbstractResponse


__all__ = (
    'EngineError',
    'BinaryNotFoundError',
    'MismatchedVersionsError',
    'EngineConnectionError',
    'EngineRequestError',
    'AlreadyConnectedError',
    'NotConnectedError',
    'UnprocessableEntityError',
)


_AnyResponse = AbstractResponse[Any]


class EngineError(PrismaError):
    pass


class BinaryNotFoundError(EngineError):
    pass


class AlreadyConnectedError(EngineError):
    pass


class NotConnectedError(EngineError):
    pass


class MismatchedVersionsError(EngineError):
    got: str
    expected: str

    def __init__(self, *, expected: str, got: str):
        super().__init__(
            f'Expected query engine version `{expected}` but got `{got}`.\n'
            + 'If this is intentional then please set the PRISMA_PY_DEBUG_GENERATOR environment '
            + 'variable to 1 and try again.'
        )
        self.expected = expected
        self.got = got


class EngineConnectionError(EngineError):
    pass


class EngineRequestError(EngineError):
    response: _AnyResponse

    def __init__(self, response: _AnyResponse, body: str):
        self.response = response

        # TODO: better error message
        super().__init__(f'{response.status}: {body}')


class UnprocessableEntityError(EngineRequestError):
    def __init__(self, response: _AnyResponse):
        super().__init__(
            response,
            (
                'Error occurred, '
                'it is likely that the internal GraphQL query '
                'builder generated a malformed request.\n'
                'Please create an issue at https://github.com/RobertCraigie/prisma-client-py/issues'
            ),
        )
