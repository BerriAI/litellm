"""
ECS logging early-startup hook for litellm.

Python executes sitecustomize.py from the active site-packages directory
before any user code, making it the right place to configure logging format
before the first log record is emitted.

To enable ECS-compliant log output without touching application code:

    1. Find your environment's site-packages path:
           python -c "import site; print(site.getsitepackages()[0])"

    2. If no sitecustomize.py exists there yet, copy this file:
           cp litellm/sitecustomize.py <site-packages>/sitecustomize.py

       If one already exists, append this import to it:
           echo "import litellm.sitecustomize" >> <site-packages>/sitecustomize.py

    3. Set the environment variable before starting your process:
           export LITELLM_ECS_LOGS=true

Alternatively, call litellm._logging._turn_on_ecs() directly at the top of
your application's entrypoint if you prefer not to touch site-packages.
"""

import os

if os.environ.get("LITELLM_ECS_LOGS", "").lower() == "true":
    try:
        from litellm._logging import _turn_on_ecs

        _turn_on_ecs()
    except Exception:
        pass
