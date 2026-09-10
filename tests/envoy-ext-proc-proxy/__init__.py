import os
import sys

sys.path.insert(  # test-quality-ok: importing non-default component
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../envoy-ext-proc-proxy")),
)
