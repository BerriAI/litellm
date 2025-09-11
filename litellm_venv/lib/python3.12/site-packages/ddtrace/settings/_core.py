import os
from typing import Any  # noqa:F401
from typing import Callable  # noqa:F401
from typing import List  # noqa:F401
from typing import Optional  # noqa:F401
from typing import Union  # noqa:F401

from envier.env import EnvVariable
from envier.env import _normalized

from ddtrace.internal.telemetry import telemetry_writer


def report_telemetry(env: Any) -> None:
    for name, e in list(env.__class__.__dict__.items()):
        if isinstance(e, EnvVariable) and not e.private:
            env_name = env._full_prefix + _normalized(e.name)
            env_val = e(env, env._full_prefix)
            raw_val = env.source.get(env_name)
            if env_name in env.source and env_val == e._cast(e.type, raw_val, env):
                source = "env_var"
            elif env_val == e.default:
                source = "default"
            else:
                source = "unknown"
            telemetry_writer.add_configuration(env_name, env_val, source)


def get_config(
    envs: Union[str, List[str]],
    default: Any = None,
    modifier: Optional[Callable[[Any], Any]] = None,
    report_telemetry=True,
):
    if isinstance(envs, str):
        envs = [envs]
    val = default
    source = "default"
    effective_env = envs[0]
    for env in envs:
        if env in os.environ:
            val = os.environ[env]
            if modifier:
                val = modifier(val)
            source = "env_var"
            effective_env = env
            break
    if report_telemetry:
        telemetry_writer.add_configuration(effective_env, val, source)
    return val
