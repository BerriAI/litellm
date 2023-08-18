#### What this tests ####
#    This tests the timeout decorator

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from litellm import timeout


@timeout(10)
def stop_after_10_s(force_timeout=60):
    print("Stopping after 10 seconds")
    time.sleep(10)
    return


start_time = time.time()

try:
    stop_after_10_s(force_timeout=1)
except Exception as e:
    print(e)
    pass

end_time = time.time()

print(f"total time: {end_time-start_time}")
