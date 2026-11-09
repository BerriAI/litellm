"""Regenerates redis-py-kwargs.json, the Python facts SETTINGS in src/keys.rs answers.

Run it from the repository root in the project environment, which has the locked redis-py and litellm:

    uv run python litellm-rust/crates/redis-resolve/redis_py_kwargs.py

It reads every other redis-py version's signatures in a throwaway environment through
`uv run --no-project --with redis==<version>`, so nothing else is installed. Then update SETTINGS until
tests/catalog.rs passes.
"""

import inspect
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

VERSIONS: Final = ("5.3.1", "6.4.0", "7.4.1", "8.0.1")
PYTHON: Final = "3.12"
CATALOG: Final = Path(__file__).with_name("redis-py-kwargs.json")
# Keys litellm/_redis.py or its callers read that neither a redis-py signature nor the REDIS_* mapping names
LITELLM_ONLY: Final = frozenset(
    {"connection_class", "namespace", "sentinel_nodes", "sentinel_password", "service_name"}
)

META: Final = {
    "description": "What redis-py and litellm/_redis.py accept for every Redis kwarg the Rust resolver can be handed. SETTINGS in src/keys.rs has exactly one row per arg here",
    "redis_py_versions": list(VERSIONS),
    "python": PYTHON,
    "method": "python redis_py_kwargs.py next to this file, see its docstring. It reads inspect.getfullargspec(inspect.unwrap(cls.__init__)) args and kwonlyargs, minus self",
    "signatures": {
        "sync": "redis.Redis.__init__. _get_redis_kwargs uses it as the config allowlist and the REDIS_<KWARG> environment mapping",
        "async": "redis.asyncio.Redis.__init__, which the proxy constructs. A kwarg in sync but not async raises TypeError when the client is built",
        "connection": "redis.Connection or redis.asyncio.Connection MRO, the url allowlist from _get_redis_url_kwargs",
        "cluster": "redis.asyncio.RedisCluster.__init__, the allowlist from _get_redis_cluster_kwargs",
    },
    "environment": "true when _get_redis_env_kwarg_mapping reads the arg as REDIS_<ARG> under the locked redis-py",
    "redis_py": "each redis-py version whose signatures name the arg, with which ones. An empty object means only litellm reads it",
    "checked_by": "tests/unit/test_redis.py checks the installed redis-py's column and the REDIS_* names against this file, and tests/catalog.rs checks SETTINGS against it",
}


def _named_args(*classes: type) -> frozenset[str]:
    return frozenset(
        name
        for cls in classes
        for spec in (inspect.getfullargspec(inspect.unwrap(cls.__init__)),)
        for name in spec.args + spec.kwonlyargs
        if name != "self"
    )


def arg_signatures() -> dict[str, list[str]]:
    """Each kwarg the installed redis-py names, with the signatures that name it."""
    import redis
    import redis.asyncio as async_redis

    connection_classes: Final = (*redis.Connection.__mro__, *async_redis.Connection.__mro__)
    signatures: Final = {
        "sync": _named_args(redis.Redis),
        "async": _named_args(async_redis.Redis),
        "connection": _named_args(*(cls for cls in connection_classes if cls is not object)),
        "cluster": _named_args(async_redis.RedisCluster),
    }
    names: Final = sorted(frozenset().union(*signatures.values()))
    return {name: [kind for kind, named in signatures.items() if name in named] for name in names}


def _arg_signatures_of(version: str) -> Mapping[str, list[str]]:
    command: Final = ("uv", "run", "--no-project", "--python", PYTHON, "--with", f"redis=={version}")
    script: Final = ("python", "-I", __file__, "--arg-signatures")
    return json.loads(subprocess.run((*command, *script), check=True, capture_output=True, text=True).stdout)


def catalog() -> dict[str, object]:
    import redis

    from litellm._redis import _get_redis_env_kwarg_mapping

    environment: Final = frozenset(_get_redis_env_kwarg_mapping().values())
    by_version: Final = {version: _arg_signatures_of(version) for version in VERSIONS}
    names: Final = sorted(LITELLM_ONLY | environment | frozenset().union(*by_version.values()))
    return {
        "meta": {**META, "redis_py_locked": redis.__version__},
        "args": {
            name: {
                "environment": name in environment,
                "redis_py": {version: args[name] for version, args in by_version.items() if name in args},
            }
            for name in names
        },
    }


# one line per arg, so a version bump reads as a diff of the args it touched
def render(catalog: Mapping[str, object]) -> str:
    meta: Final = json.dumps(catalog["meta"], indent=2).replace("\n", "\n  ")
    args: Final = catalog["args"]
    assert isinstance(args, dict)
    rows: Final = ",\n".join(f"    {json.dumps(name)}: {json.dumps(arg)}" for name, arg in args.items())
    return f'{{\n  "meta": {meta},\n  "args": {{\n{rows}\n  }}\n}}\n'


if __name__ == "__main__":
    if sys.argv[1:] == ["--arg-signatures"]:
        sys.stdout.write(json.dumps(arg_signatures()))
    else:
        CATALOG.write_text(render(catalog()))
