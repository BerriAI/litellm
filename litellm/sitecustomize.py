"""
ECS logging early-startup hook template for litellm.

This file is a deploy-time template. It will NOT auto-execute from its current
location inside the litellm package. Python only auto-runs sitecustomize.py if
it exists as a top-level module in a site-packages directory.

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
