SPAN_ORIGIN = "auto.db.redis"

_SINGLE_KEY_COMMANDS = frozenset(
    ["decr", "decrby", "get", "incr", "incrby", "pttl", "set", "setex", "setnx", "ttl"],
)
_MULTI_KEY_COMMANDS = frozenset(
    [
        "del",
        "touch",
        "unlink",
        "mget",
    ],
)
_COMMANDS_INCLUDING_SENSITIVE_DATA = [
    "auth",
]
_MAX_NUM_ARGS = 10  # Trim argument lists to this many values
_MAX_NUM_COMMANDS = 10  # Trim command lists to this many values
_DEFAULT_MAX_DATA_SIZE = 1024
