import os
import inspect
import redis

def _get_redis_kwargs():
    arg_spec = inspect.getfullargspec(redis.Redis)

    # Only allow primitive arguments
    exclude_args = {
        "self",
        "connection_pool",
        "retry",
    }

    return [
        x for x in arg_spec.args if x not in exclude_args
    ]

def _get_redis_env_kwarg_mapping():
    PREFIX = "REDIS_"

    return {
        f"{PREFIX}{x.upper()}": x for x in _get_redis_kwargs()
    }


def _redis_kwargs_from_environment():
    mapping = _get_redis_env_kwarg_mapping()

    return {
        mapping[k]: v for k, v in os.environ.items() if k in mapping
    }


def get_redis_client(**env_overrides):
    redis_kwargs = {
        **_redis_kwargs_from_environment(),
        **env_overrides,
    }

    if "url" in redis_kwargs:
        redis_kwargs.pop("host", None)
        redis_kwargs.pop("port", None)
        redis_kwargs.pop("db", None)
        redis_kwargs.pop("password", None)
        
        return redis.Redis.from_url(**redis_kwargs)
    elif "host" not in redis_kwargs:
        raise ValueError("Either 'host' or 'url' must be specified for redis.")

    return redis.Redis(**redis_kwargs)