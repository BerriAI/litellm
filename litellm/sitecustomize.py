import os

if os.environ.get("LITELLM_ECS_LOGS", "").lower() == "true":
    try:
        from litellm._logging import _turn_on_ecs  # pyright: ignore[reportPrivateUsage]  # public entry point

        _turn_on_ecs()
    except Exception:
        pass
