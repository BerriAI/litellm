import logging
from typing import Any  # noqa:F401
from typing import Dict  # noqa:F401
from typing import List  # noqa:F401
from typing import Optional  # noqa:F401
from typing import Text  # noqa:F401
from typing import Tuple  # noqa:F401
from typing import TypeVar  # noqa:F401
from typing import Union  # noqa:F401

from ddtrace.internal.constants import MAX_UINT_64BITS  # noqa:F401

from ..compat import ensure_text


VALUE_PLACEHOLDER = "?"
VALUE_MAX_LEN = 100
VALUE_TOO_LONG_MARK = "..."
CMD_MAX_LEN = 1000


T = TypeVar("T")


log = logging.getLogger(__name__)


def deep_getattr(obj, attr_string, default=None):
    # type: (Any, str, Optional[Any]) -> Optional[Any]
    """
    Returns the attribute of `obj` at the dotted path given by `attr_string`
    If no such attribute is reachable, returns `default`

    >>> deep_getattr(cass, 'cluster')
    <cassandra.cluster.Cluster object at 0xa20c350

    >>> deep_getattr(cass, 'cluster.metadata.partitioner')
    u'org.apache.cassandra.dht.Murmur3Partitioner'

    >>> deep_getattr(cass, 'i.dont.exist', default='default')
    'default'
    """
    attrs = attr_string.split(".")
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return default

    return obj


def asbool(value):
    # type: (Union[str, bool, None]) -> bool
    """Convert the given String to a boolean object.

    Accepted values are `True` and `1`.
    """
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    return value.lower() in ("true", "1")


def parse_tags_str(tags_str):
    # type: (Optional[str]) -> Dict[str, str]
    """Parse a string of tags typically provided via environment variables.

    The expected string is of the form::
        "key1:value1,key2:value2"
        "key1:value1 key2:value2"
        "key1,key2"
        "key1 key2"

    :param tags_str: A string of the above form to parse tags from.
    :return: A dict containing the tags that were parsed.
    """
    if not tags_str:
        return {}

    TAGSEP = ", "

    def parse_tags(tags):
        # type: (List[str]) -> Tuple[List[Tuple[str, str]], List[str]]
        parsed_tags = []
        invalids = []

        for tag in tags:
            key, sep, value = tag.partition(":")
            if not key.strip() or "," in key or (sep and not value):
                invalids.append(tag)
            elif sep:
                # parse key:val,key2:value2
                parsed_tags.append((key, value))
            else:
                # parse key,key2
                parsed_tags.append((key, ""))

        return parsed_tags, invalids

    tags_str = tags_str.strip(TAGSEP)

    # Take the maximal set of tags that can be parsed correctly for a given separator
    tag_list = []  # type: List[Tuple[str, str]]
    invalids = []
    for sep in TAGSEP:
        ts = tags_str.split(sep)
        tags, invs = parse_tags(ts)
        if len(tags) > len(tag_list):
            tag_list = tags
            invalids = invs
        elif len(tags) == len(tag_list) > 1:
            # Both separators produce the same number of tags.
            # DEV: This only works when checking between two separators.
            tag_list[:] = []
            invalids[:] = []

    if not tag_list:
        log.error(
            (
                "Failed to find separator for tag string: '%s'.\n"
                "Tag strings must be comma or space separated:\n"
                "  key1:value1,key2:value2\n"
                "  key1:value1 key2:value2"
            ),
            tags_str,
        )

    for tag in invalids:
        log.error("Malformed tag in tag pair '%s' from tag string '%s'.", tag, tags_str)

    return dict(tag_list)


def stringify_cache_args(args, value_max_len=VALUE_MAX_LEN, cmd_max_len=CMD_MAX_LEN):
    # type: (List[Any], int, int) -> Text
    """Convert a list of arguments into a space concatenated string

    This function is useful to convert a list of cache keys
    into a resource name or tag value with a max size limit.
    """
    length = 0
    out = []  # type: List[Text]
    for arg in args:
        try:
            if isinstance(arg, (bytes, str)):
                cmd = ensure_text(arg, errors="backslashreplace")
            else:
                cmd = str(arg)

            if len(cmd) > value_max_len:
                cmd = cmd[:value_max_len] + VALUE_TOO_LONG_MARK

            if length + len(cmd) > cmd_max_len:
                prefix = cmd[: cmd_max_len - length]
                out.append("%s%s" % (prefix, VALUE_TOO_LONG_MARK))
                break

            out.append(cmd)
            length += len(cmd)
        except Exception:
            out.append(VALUE_PLACEHOLDER)
            break

    return " ".join(out)


def is_sequence(obj):
    # type: (Any) -> bool
    try:
        return isinstance(obj, (list, tuple, set, frozenset))
    except TypeError:
        # Checking the type of Generic Subclasses raises a TypeError
        return False


def flatten_key_value(root_key, value):
    # type: (str, Any) -> Dict[str, Any]
    """Flattens attributes"""
    if not is_sequence(value):
        return {root_key: value}

    flattened = dict()
    for i, item in enumerate(value):
        key = f"{root_key}.{i}"
        if is_sequence(item):
            flattened.update(flatten_key_value(key, item))
        else:
            flattened[key] = item
    return flattened


def format_trace_id(trace_id: int) -> str:
    """Translate a trace ID to a string format supported by the backend."""
    return "{:032x}".format(trace_id) if trace_id > MAX_UINT_64BITS else str(trace_id)
