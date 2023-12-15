"""
Miscellaneous helper functions.

The formatter for ANSI colored console output is heavily based on Pygments
terminal colorizing code, originally by Georg Brandl.
"""

import calendar
import datetime
import datetime as dt
import importlib
import logging
import numbers
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from redis import Redis

    from .queue import Queue

from redis.exceptions import ResponseError

from .exceptions import TimeoutFormatError

logger = logging.getLogger(__name__)


def compact(lst: List[Any]) -> List[Any]:
    """Excludes `None` values from a list-like object.

    Args:
        lst (list): A list (or list-like) oject

    Returns:
        object (list): The list without None values
    """
    return [item for item in lst if item is not None]


def as_text(v: Union[bytes, str]) -> str:
    """Converts a bytes value to a string using `utf-8`.

    Args:
        v (Union[bytes, str]): The value (bytes or string)

    Raises:
        ValueError: If the value is not bytes or string

    Returns:
        value (Optional[str]): Either the decoded string or None
    """
    if isinstance(v, bytes):
        return v.decode('utf-8')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


def decode_redis_hash(h) -> Dict[str, Any]:
    """Decodes the Redis hash, ensuring that keys are strings
    Most importantly, decodes bytes strings, ensuring the dict has str keys.

    Args:
        h (Dict[Any, Any]): The Redis hash

    Returns:
        Dict[str, Any]: The decoded Redis data (Dictionary)
    """
    return dict((as_text(k), h[k]) for k in h)


def import_attribute(name: str) -> Callable[..., Any]:
    """Returns an attribute from a dotted path name. Example: `path.to.func`.

    When the attribute we look for is a staticmethod, module name in its
    dotted path is not the last-before-end word

    E.g.: package_a.package_b.module_a.ClassA.my_static_method

    Thus we remove the bits from the end of the name until we can import it

    Args:
        name (str): The name (reference) to the path.

    Raises:
        ValueError: If no module is found or invalid attribute name.

    Returns:
        Any: An attribute (normally a Callable)
    """
    name_bits = name.split('.')
    module_name_bits, attribute_bits = name_bits[:-1], [name_bits[-1]]
    module = None
    while len(module_name_bits):
        try:
            module_name = '.'.join(module_name_bits)
            module = importlib.import_module(module_name)
            break
        except ImportError:
            attribute_bits.insert(0, module_name_bits.pop())

    if module is None:
        # maybe it's a builtin
        try:
            return __builtins__[name]
        except KeyError:
            raise ValueError('Invalid attribute name: %s' % name)

    attribute_name = '.'.join(attribute_bits)
    if hasattr(module, attribute_name):
        return getattr(module, attribute_name)
    # staticmethods
    attribute_name = attribute_bits.pop()
    attribute_owner_name = '.'.join(attribute_bits)
    try:
        attribute_owner = getattr(module, attribute_owner_name)
    except:  # noqa
        raise ValueError('Invalid attribute name: %s' % attribute_name)

    if not hasattr(attribute_owner, attribute_name):
        raise ValueError('Invalid attribute name: %s' % name)
    return getattr(attribute_owner, attribute_name)


def utcnow():
    return datetime.datetime.utcnow()


def now():
    """Return now in UTC"""
    return datetime.datetime.now(datetime.timezone.utc)


_TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


def utcformat(dt: dt.datetime) -> str:
    return dt.strftime(as_text(_TIMESTAMP_FORMAT))


def utcparse(string: str) -> dt.datetime:
    try:
        return datetime.datetime.strptime(string, _TIMESTAMP_FORMAT)
    except ValueError:
        # This catches any jobs remain with old datetime format
        return datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%SZ')


def first(iterable: Iterable, default=None, key=None):
    """Return first element of `iterable` that evaluates true, else return None
    (or an optional default value).

    >>> first([0, False, None, [], (), 42])
    42

    >>> first([0, False, None, [], ()]) is None
    True

    >>> first([0, False, None, [], ()], default='ohai')
    'ohai'

    >>> import re
    >>> m = first(re.match(regex, 'abc') for regex in ['b.*', 'a(.*)'])
    >>> m.group(1)
    'bc'

    The optional `key` argument specifies a one-argument predicate function
    like that used for `filter()`.  The `key` argument, if supplied, must be
    in keyword form.  For example:

    >>> first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0)
    4

    Args:
        iterable (t.Iterable): _description_
        default (_type_, optional): _description_. Defaults to None.
        key (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if key is None:
        for el in iterable:
            if el:
                return el
    else:
        for el in iterable:
            if key(el):
                return el

    return default


def is_nonstring_iterable(obj: Any) -> bool:
    """Returns whether the obj is an iterable, but not a string

    Args:
        obj (Any): _description_

    Returns:
        bool: _description_
    """
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def ensure_list(obj: Any) -> List:
    """When passed an iterable of objects, does nothing, otherwise, it returns
    a list with just that object in it.

    Args:
        obj (Any): _description_

    Returns:
        List: _description_
    """
    return obj if is_nonstring_iterable(obj) else [obj]


def current_timestamp() -> int:
    """Returns current UTC timestamp

    Returns:
        int: _description_
    """
    return calendar.timegm(datetime.datetime.utcnow().utctimetuple())


def backend_class(holder, default_name, override=None):
    """Get a backend class using its default attribute name or an override

    Args:
        holder (_type_): _description_
        default_name (_type_): _description_
        override (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if override is None:
        return getattr(holder, default_name)
    elif isinstance(override, str):
        return import_attribute(override)
    else:
        return override


def str_to_date(date_str: Optional[str]) -> Union[dt.datetime, Any]:
    if not date_str:
        return
    else:
        return utcparse(date_str.decode())


def parse_timeout(timeout: Union[int, float, str]) -> int:
    """Transfer all kinds of timeout format to an integer representing seconds"""
    if not isinstance(timeout, numbers.Integral) and timeout is not None:
        try:
            timeout = int(timeout)
        except ValueError:
            digit, unit = timeout[:-1], (timeout[-1:]).lower()
            unit_second = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}
            try:
                timeout = int(digit) * unit_second[unit]
            except (ValueError, KeyError):
                raise TimeoutFormatError(
                    'Timeout must be an integer or a string representing an integer, or '
                    'a string with format: digits + unit, unit can be "d", "h", "m", "s", '
                    'such as "1h", "23m".'
                )

    return timeout


def get_version(connection: 'Redis') -> Tuple[int, int, int]:
    """
    Returns tuple of Redis server version.
    This function also correctly handles 4 digit redis server versions.

    Args:
        connection (Redis): The Redis connection.

    Returns:
        version (Tuple[int, int, int]): A tuple representing the semantic versioning format (eg. (5, 0, 9))
    """
    try:
        # Getting the connection info for each job tanks performance, we can cache it on the connection object
        if not getattr(connection, "__rq_redis_server_version", None):
            setattr(
                connection,
                "__rq_redis_server_version",
                tuple(int(i) for i in connection.info("server")["redis_version"].split('.')[:3]),
            )
        return getattr(connection, "__rq_redis_server_version")
    except ResponseError:  # fakeredis doesn't implement Redis' INFO command
        return (5, 0, 9)


def ceildiv(a, b):
    """Ceiling division. Returns the ceiling of the quotient of a division operation

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    return -(-a // b)


def split_list(a_list: List[Any], segment_size: int):
    """Splits a list into multiple smaller lists having size `segment_size`

    Args:
        a_list (List[Any]): A list to split
        segment_size (int): The segment size to split into

    Yields:
        list: The splitted listed
    """
    for i in range(0, len(a_list), segment_size):
        yield a_list[i : i + segment_size]


def truncate_long_string(data: str, max_length: Optional[int] = None) -> str:
    """Truncate arguments with representation longer than max_length

    Args:
        data (str): The data to truncate
        max_length (Optional[int], optional): The max length. Defaults to None.

    Returns:
        truncated (str): The truncated string
    """
    if max_length is None:
        return data
    return (data[:max_length] + '...') if len(data) > max_length else data


def get_call_string(
    func_name: Optional[str], args: Any, kwargs: Dict[Any, Any], max_length: Optional[int] = None
) -> Optional[str]:
    """
    Returns a string representation of the call, formatted as a regular
    Python function invocation statement. If max_length is not None, truncate
    arguments with representation longer than max_length.

    Args:
        func_name (str): The funtion name
        args (Any): The function arguments
        kwargs (Dict[Any, Any]): The function kwargs
        max_length (int, optional): The max length. Defaults to None.

    Returns:
        str: A String representation of the function call.
    """
    if func_name is None:
        return None

    arg_list = [as_text(truncate_long_string(repr(arg), max_length)) for arg in args]

    list_kwargs = ['{0}={1}'.format(k, as_text(truncate_long_string(repr(v), max_length))) for k, v in kwargs.items()]
    arg_list += sorted(list_kwargs)
    args = ', '.join(arg_list)

    return '{0}({1})'.format(func_name, args)


def parse_names(queues_or_names: List[Union[str, 'Queue']]) -> List[str]:
    """Given a list of strings or queues, returns queue names"""
    from .queue import Queue

    names = []
    for queue_or_name in queues_or_names:
        if isinstance(queue_or_name, Queue):
            names.append(queue_or_name.name)
        else:
            names.append(str(queue_or_name))
    return names
