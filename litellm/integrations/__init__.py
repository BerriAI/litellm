from . import *

# LIT-2577: install v3 rate-limit remaining-headers fallback on PrometheusLogger
# at package import time. See `_prometheus_v3_remaining_fix.py` for the
# full rationale; the import below has the side-effect of running the patch.
from . import _prometheus_v3_remaining_fix  # noqa: F401
