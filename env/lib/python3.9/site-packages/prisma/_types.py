from typing import Callable, Coroutine, TypeVar, Type, Tuple, Any
from pydantic import BaseModel
from typing_extensions import (
    TypeGuard as TypeGuard,
    TypedDict as TypedDict,
    Protocol as Protocol,
    Literal as Literal,
    get_args as get_args,
    runtime_checkable as runtime_checkable,
)

Method = Literal['GET', 'POST']

CallableT = TypeVar('CallableT', bound='FuncType')
BaseModelT = TypeVar('BaseModelT', bound=BaseModel)

# TODO: use a TypeVar everywhere
FuncType = Callable[..., object]
CoroType = Callable[..., Coroutine[Any, Any, object]]


@runtime_checkable
class InheritsGeneric(Protocol):
    __orig_bases__: Tuple['_GenericAlias']


class _GenericAlias(Protocol):
    __origin__: Type[object]


PrismaMethod = Literal[
    # raw queries
    'query_raw',
    'query_first',
    'execute_raw',
    # mutatitive queries
    'create',
    'delete',
    'update',
    'upsert',
    'create_many',
    'delete_many',
    'update_many',
    # read queries
    'count',
    'group_by',
    'find_many',
    'find_first',
    'find_first_or_raise',
    'find_unique',
    'find_unique_or_raise',
]
