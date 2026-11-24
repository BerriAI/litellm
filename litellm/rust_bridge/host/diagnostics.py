from __future__ import annotations

import traceback


def log_environment_fallback(name: str, error: Exception) -> None:
    from litellm._logging import verbose_logger

    verbose_logger.error(
        "Defaulting to os.environ value for key=%s. An exception occurred - %s.\n\n%s",
        name,
        error,
        "".join(traceback.format_exception(error)),
    )
