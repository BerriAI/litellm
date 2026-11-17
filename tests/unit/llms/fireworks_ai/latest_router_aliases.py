from typing import Final

# https://api.fireworks.ai/v1/serverless/models?format=nested, read 2026-10-29
LATEST_ROUTER_ALIAS_TARGETS: Final = (
    ("accounts/fireworks/routers/minimax-latest", "accounts/fireworks/models/minimax-m3"),
    ("accounts/fireworks/routers/kimi-latest", "accounts/fireworks/models/kimi-k3"),
    ("accounts/fireworks/routers/kimi-fast-latest", "accounts/fireworks/routers/kimi-k3-fast"),
    ("accounts/fireworks/routers/qwen-max-latest", "accounts/fireworks/models/qwen3p8-max"),
    ("accounts/fireworks/routers/glm-latest", "accounts/fireworks/models/glm-5p3"),
    ("accounts/fireworks/routers/glm-fast-latest", "accounts/fireworks/routers/glm-5p3-fast"),
    ("accounts/fireworks/routers/glm-flash-latest", "accounts/fireworks/models/glm-5p3-flash"),
    ("accounts/fireworks/routers/deepseek-flash-latest", "accounts/fireworks/models/deepseek-v4p1-flash-1017"),
)
