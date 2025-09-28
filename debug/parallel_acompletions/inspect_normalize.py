#!/usr/bin/env python3
from litellm.router_utils.parallel_acompletion import _normalize_requests

raw = [{"model": "demo", "messages": [["role", "user"]], "temperature": 0.1}]
print(_normalize_requests(raw))
