from sentry_sdk.consts import SPANDATA
from sentry_sdk.integrations.redis.consts import (
    _COMMANDS_INCLUDING_SENSITIVE_DATA,
    _MAX_NUM_ARGS,
    _MAX_NUM_COMMANDS,
    _MULTI_KEY_COMMANDS,
    _SINGLE_KEY_COMMANDS,
)
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.utils import SENSITIVE_DATA_SUBSTITUTE

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Optional, Sequence
    from sentry_sdk.tracing import Span


def _get_safe_command(name, args):
    # type: (str, Sequence[Any]) -> str
    command_parts = [name]

    for i, arg in enumerate(args):
        if i > _MAX_NUM_ARGS:
            break

        name_low = name.lower()

        if name_low in _COMMANDS_INCLUDING_SENSITIVE_DATA:
            command_parts.append(SENSITIVE_DATA_SUBSTITUTE)
            continue

        arg_is_the_key = i == 0
        if arg_is_the_key:
            command_parts.append(repr(arg))

        else:
            if should_send_default_pii():
                command_parts.append(repr(arg))
            else:
                command_parts.append(SENSITIVE_DATA_SUBSTITUTE)

    command = " ".join(command_parts)
    return command


def _safe_decode(key):
    # type: (Any) -> str
    if isinstance(key, bytes):
        try:
            return key.decode()
        except UnicodeDecodeError:
            return ""

    return str(key)


def _key_as_string(key):
    # type: (Any) -> str
    if isinstance(key, (dict, list, tuple)):
        key = ", ".join(_safe_decode(x) for x in key)
    elif isinstance(key, bytes):
        key = _safe_decode(key)
    elif key is None:
        key = ""
    else:
        key = str(key)

    return key


def _get_safe_key(method_name, args, kwargs):
    # type: (str, Optional[tuple[Any, ...]], Optional[dict[str, Any]]) -> Optional[tuple[str, ...]]
    """
    Gets the key (or keys) from the given method_name.
    The method_name could be a redis command or a django caching command
    """
    key = None

    if args is not None and method_name.lower() in _MULTI_KEY_COMMANDS:
        # for example redis "mget"
        key = tuple(args)

    elif args is not None and len(args) >= 1:
        # for example django "set_many/get_many" or redis "get"
        if isinstance(args[0], (dict, list, tuple)):
            key = tuple(args[0])
        else:
            key = (args[0],)

    elif kwargs is not None and "key" in kwargs:
        # this is a legacy case for older versions of Django
        if isinstance(kwargs["key"], (list, tuple)):
            if len(kwargs["key"]) > 0:
                key = tuple(kwargs["key"])
        else:
            if kwargs["key"] is not None:
                key = (kwargs["key"],)

    return key


def _parse_rediscluster_command(command):
    # type: (Any) -> Sequence[Any]
    return command.args


def _set_pipeline_data(
    span, is_cluster, get_command_args_fn, is_transaction, command_stack
):
    # type: (Span, bool, Any, bool, Sequence[Any]) -> None
    span.set_tag("redis.is_cluster", is_cluster)
    span.set_tag("redis.transaction", is_transaction)

    commands = []
    for i, arg in enumerate(command_stack):
        if i >= _MAX_NUM_COMMANDS:
            break

        command = get_command_args_fn(arg)
        commands.append(_get_safe_command(command[0], command[1:]))

    span.set_data(
        "redis.commands",
        {
            "count": len(command_stack),
            "first_ten": commands,
        },
    )


def _set_client_data(span, is_cluster, name, *args):
    # type: (Span, bool, str, *Any) -> None
    span.set_tag("redis.is_cluster", is_cluster)
    if name:
        span.set_tag("redis.command", name)
        span.set_tag(SPANDATA.DB_OPERATION, name)

    if name and args:
        name_low = name.lower()
        if (name_low in _SINGLE_KEY_COMMANDS) or (
            name_low in _MULTI_KEY_COMMANDS and len(args) == 1
        ):
            span.set_tag("redis.key", args[0])
