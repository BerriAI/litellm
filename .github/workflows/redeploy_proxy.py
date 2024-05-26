"""

redeploy_proxy.py
"""

import os
import requests

# send a get request to this endpoint
deploy_hook1 = os.getenv("LOAD_TEST_REDEPLOY_URL1")
response = requests.get(deploy_hook1, timeout=20)
