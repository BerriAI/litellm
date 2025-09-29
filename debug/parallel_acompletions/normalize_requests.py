#!/usr/bin/env python3
from litellm.router_utils.parallel_acompletion import RouterParallelRequest, _normalize_requests

reqs = [{"model": "m", "messages": [["role", "user"]]}]
print(_normalize_requests(reqs))
