# This module supports an optional feature.  It may not even load on all platforms or configurations.
# In ddtrace/settings/profiling.py, this module is imported and the is_available attribute is checked to determine
# whether the feature is available. If not, then the feature is disabled and all downstream consumption is
# suppressed.
is_available = False
failure_msg = ""


try:
    from ._ddup import *  # noqa: F403, F401

    is_available = True

except Exception as e:
    failure_msg = str(e)
