from __future__ import annotations

from typing import Any, Optional


__all__ = (
    'PrismaError',
    'DataError',
    'UniqueViolationError',
    'ForeignKeyViolationError',
    'MissingRequiredValueError',
    'RawQueryError',
    'TableNotFoundError',
    'RecordNotFoundError',
    'HTTPClientClosedError',
    'ClientNotConnectedError',
    'PrismaWarning',
    'UnsupportedSubclassWarning',
)


class PrismaError(Exception):
    pass


class ClientNotRegisteredError(PrismaError):
    def __init__(self) -> None:
        super().__init__(
            'No client instance registered; You must call prisma.register(prisma.Prisma())'
        )


class ClientAlreadyRegisteredError(PrismaError):
    def __init__(self) -> None:
        super().__init__('A client has already been registered.')


class ClientNotConnectedError(PrismaError):
    def __init__(self) -> None:
        super().__init__(
            'Client is not connected to the query engine, '
            'you must call `connect()` before attempting to query data.'
        )


class HTTPClientClosedError(PrismaError):
    def __init__(self) -> None:
        super().__init__('Cannot make a request from a closed client.')


class UnsupportedDatabaseError(PrismaError):
    context: str
    database: str

    def __init__(self, database: str, context: str) -> None:
        super().__init__(f'{context} is not supported by {database}')
        self.database = database
        self.context = context


class DataError(PrismaError):
    data: Any
    code: Any
    meta: Any

    def __init__(self, data: Any, *, message: Optional[str] = None) -> None:
        self.data = data

        user_facing_error = data.get('user_facing_error', {})
        self.code = user_facing_error.get('error_code')
        self.meta = user_facing_error.get('meta')

        message = message or user_facing_error.get('message')
        super().__init__(message or 'An error occurred while processing data.')


class UniqueViolationError(DataError):
    pass


class ForeignKeyViolationError(DataError):
    pass


class MissingRequiredValueError(DataError):
    pass


class RawQueryError(DataError):
    def __init__(self, data: Any) -> None:
        try:
            super().__init__(
                data, message=data['user_facing_error']['meta']['message']
            )
        except KeyError:
            super().__init__(data)


class TableNotFoundError(DataError):
    def __init__(self, data: Any) -> None:
        super().__init__(data)
        self.table: Optional[str] = self.meta.get('table')


class FieldNotFoundError(DataError):
    # currently we cannot easily resolve the erroneous field as Prisma
    # returns different results for unknown fields in different situations
    # e.g. root query, nested query and mutation queries
    def __init__(self, data: Any, *, message: str | None = None) -> None:
        if message is None:
            meta = data.get('user_facing_error', {}).get('meta', {})
            if meta.get('kind') == 'Union':
                error = _pick_union_error(meta.get('errors', []))
            else:
                error = meta

            argument_path = error.get('argumentPath')
            selection_path = error.get('selectionPath')

            if argument_path:
                message = f'Could not find field at `{".".join(selection_path)}.{".".join(argument_path)}`'

        super().__init__(data, message=message)


class RecordNotFoundError(DataError):
    pass


class InputError(DataError):
    pass


class TransactionError(PrismaError):
    pass


class TransactionExpiredError(TransactionError):
    pass


class TransactionNotStartedError(TransactionError):
    def __init__(self) -> None:
        super().__init__(
            'Transaction has not been started yet.\n'
            'Transactions must be used within a context manager or started manually.'
        )


class BuilderError(PrismaError):
    pass


class InvalidModelError(BuilderError):
    def __init__(self, model: type) -> None:
        super().__init__(
            f'Expected the {model} type to have a `__prisma_model__` class variable set'
        )


class UnknownModelError(BuilderError):
    def __init__(self, model: str) -> None:
        super().__init__(f'Model: "{model}" does not exist.')


class UnknownRelationalFieldError(BuilderError):
    def __init__(self, model: str, field: str) -> None:
        super().__init__(
            f'Field: "{field}" either does not exist or is not a relational field on the {model} model'
        )


class GeneratorError(PrismaError):
    pass


class UnsupportedListTypeError(GeneratorError):
    type: str

    def __init__(self, typ: str) -> None:
        super().__init__(
            f'Cannot use {typ} as a list yet; Please create a '
            'feature request at https://github.com/RobertCraigie/prisma-client-py/issues/new'
        )
        self.type = typ


class PrismaWarning(Warning):
    pass


# Note: this is currently unused but not worth removing
class UnsupportedSubclassWarning(PrismaWarning):
    pass


# TODO: proper types
def _pick_union_error(errors: list[Any]) -> Any:
    # Note: uses the same heuristic as the TS client
    return max(
        errors,
        key=lambda e: (
            len(e.get('argumentPath', [])) + len(e.get('selectionPath'))
        ),
    )
