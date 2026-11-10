import os
import warnings

import dotenv


def _dev_env_hot_reload_enabled() -> bool:
    """The proxy exports this flag when started with ``--reload``. A reloaded
    worker is a fresh process that inherits the reloader's environment, so an
    edited ``.env`` value stays masked by the stale inherited one unless we
    let the file win; overriding makes the edit take effect on reload."""
    return os.getenv("LITELLM_DEV_ENV_HOT_RELOAD") == "True"


def apply_process_defaults() -> None:
    warnings.filterwarnings("ignore", message=".*conflict with protected namespace.*")
    # Suppress Pydantic 2.11+ deprecation warning about accessing model_fields on instances
    # This warning can accumulate during streaming and cause memory leaks
    warnings.filterwarnings("ignore", message=".*Accessing the.*attribute on the instance is deprecated.*")
    # ReadOnly on TypedDict fields is repo-wide static discipline (LIT012); pydantic warns it
    # cannot enforce it at runtime, which floods proxy boot once such a type is schema-walked
    warnings.filterwarnings("ignore", message=".*`ReadOnly` qualifier.*")
    if os.getenv("LITELLM_MODE", "DEV") == "DEV":
        dotenv.load_dotenv(override=_dev_env_hot_reload_enabled())
