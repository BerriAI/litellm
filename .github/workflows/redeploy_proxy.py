"""

redeploy_proxy.py
"""

import os
import requests
import time

# send a get request to this endpoint
deploy_hook1 = os.getenv("LOAD_TEST_REDEPLOY_URL1")
response = requests.get(deploy_hook1, timeout=20)


deploy_hook2 = os.getenv("LOAD_TEST_REDEPLOY_URL2")
response = requests.get(deploy_hook2, timeout=20)

print("SENT GET REQUESTS to re-deploy proxy")
print("sleeeping.... for 60s")
time.sleep(60)
