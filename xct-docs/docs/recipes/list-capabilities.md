---
sidebar_position: 1
---

# List capabilities

Show everything the current caller can use, in one call.

```python
from xct_litellm import XctClient

xct = XctClient(base_url="https://api.xct.test", access_token=token, app_id="xct-chat")
caps = xct.capabilities.list()

# caps["models"]        → [{"id":"gpt-4o", "capabilities":{...}, ...}, ...]
# caps["agents"]        → [{"agent_id":"...", "agent_name":"...", "is_public":..., ...}]
# caps["mcps"]          → [{"server_id":"...", "transport":"http", "tools_count":..., ...}]
# caps["skills"]        → [{"skill_id":"...", "display_title":"...", "is_public":..., ...}]
# caps["access_groups"] → []   (placeholder; future expansion)
# caps["caller"]        → {"key_id":"hashed", "team_id":"...", "user_id":"...", "app_id":"..."}
# caps["schema_version"]→ "2026-05-19"
```

Result is cached **per (token, app_id)** for 60s. Permission changes
propagate at most that fast.

Need only public stuff (no auth)? Use
`GET /.well-known/xct-capabilities` instead.
