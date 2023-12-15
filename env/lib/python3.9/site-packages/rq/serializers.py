import json
import pickle
from functools import partial
from typing import Optional, Type, Union

from .utils import import_attribute


class DefaultSerializer:
    dumps = partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
    loads = pickle.loads


class JSONSerializer:
    @staticmethod
    def dumps(*args, **kwargs):
        return json.dumps(*args, **kwargs).encode('utf-8')

    @staticmethod
    def loads(s, *args, **kwargs):
        return json.loads(s.decode('utf-8'), *args, **kwargs)


def resolve_serializer(serializer: Optional[Union[Type[DefaultSerializer], str]] = None) -> Type[DefaultSerializer]:
    """This function checks the user defined serializer for ('dumps', 'loads') methods
    It returns a default pickle serializer if not found else it returns a MySerializer
    The returned serializer objects implement ('dumps', 'loads') methods
    Also accepts a string path to serializer that will be loaded as the serializer.

    Args:
        serializer (Callable): The serializer to resolve.

    Returns:
        serializer (Callable): An object that implements the SerializerProtocol
    """
    if not serializer:
        return DefaultSerializer

    if isinstance(serializer, str):
        serializer = import_attribute(serializer)

    default_serializer_methods = ('dumps', 'loads')

    for instance_method in default_serializer_methods:
        if not hasattr(serializer, instance_method):
            raise NotImplementedError('Serializer should have (dumps, loads) methods.')

    return serializer
