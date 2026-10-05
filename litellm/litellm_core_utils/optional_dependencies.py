from importlib import import_module
from typing import Final

from litellm._logging import verbose_logger


class MissingOptionalDependencyError(ImportError):
    pass


def require_optional_dependency(module: str, extra: str, capability: str) -> None:
    try:
        import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        message: Final = f'{capability} requires {module}. Install support with pip install "litellm[{extra}]"'
        verbose_logger.error(message)
        raise MissingOptionalDependencyError(message) from error
