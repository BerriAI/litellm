"""
ddtrace.vendor
==============
Install vendored dependencies under a different top level package to avoid importing `ddtrace/__init__.py`
whenever a dependency is imported. Doing this allows us to have a little more control over import order.


Dependencies
============

dogstatsd
---------

Website: https://datadogpy.readthedocs.io/en/latest/
Source: https://github.com/DataDog/datadogpy
Version: 8e11af2 (0.39.1)
License: Copyright (c) 2020, Datadog <info@datadoghq.com>

Notes:
  `dogstatsd/__init__.py` was updated to include a copy of the `datadogpy` license: https://github.com/DataDog/datadogpy/blob/master/LICENSE
  Only `datadog.dogstatsd` module was vendored to avoid unnecessary dependencies
  `datadog/util/compat.py` was copied to `dogstatsd/compat.py`
  `datadog/util/format.py` was copied to `dogstatsd/format.py`
  version fixed to 8e11af2
  removed type imports
  removed unnecessary compat utils


monotonic
---------

Website: https://pypi.org/project/monotonic/
Source: https://github.com/atdt/monotonic
Version: 1.5
License: Apache License 2.0

Notes:
  The source `monotonic.py` was added as `monotonic/__init__.py`

  No other changes were made

debtcollector
-------------

Website: https://docs.openstack.org/debtcollector/latest/index.html
Source: https://github.com/openstack/debtcollector
Version: 2.5.0
License: Apache License 2.0

Notes:
   Removed dependency on `pbr` and manually set `__version__`


psutil
------

Website: https://github.com/giampaolo/psutil
Source: https://github.com/giampaolo/psutil
Version: 5.6.7
License: BSD 3


contextvars
-------------

Source: https://github.com/MagicStack/contextvars
Version: 2.4
License: Apache License 2.0

Notes:
  - removal of metaclass usage
  - formatting
  - use a plain old dict instead of immutables.Map
  - removal of `*` syntax


sqlcommenter
------------

Source: https://github.com/open-telemetry/opentelemetry-sqlcommenter/blob/2f8841add68358069ebf1c0ee560ab3e98a59aa9/python/sqlcommenter-python/opentelemetry/sqlcommenter/__init__.py
License: Apache License 2.0


packaging
---------

Source: https://github.com/pypa/packaging
Version: 17.1
License: Apache License 2.0

Notes:
  - We only vendor the packaging.version sub-module as this is all we currently
    need.


ply
---------

Source: https://github.com/dabeaz/ply
Version: 3.11
License: BSD-3-Clause

Notes:
  - jsonpath-ng dependency
    Did a "pip install jsonpath-ng"
    Then went and looked at the contents of the ply packages
    yacc.py and lex.py files here.
    Didn't copy: cpp.py, ctokens.py, ygen.py (didn't see them used)

    
jsonpath-ng
---------

Source: https://github.com/h2non/jsonpath-ng
Version: 1.6.1
License: Apache License 2.0

Notes:
  - Copied ply into vendors as well.
    Changed "-" to "_" as was causing errors when importing.
"""

# Initialize `ddtrace.vendor.datadog.base.log` logger with our custom rate limited logger
# DEV: This helps ensure if there are connection issues we do not spam their logs
# DEV: Overwrite `base.log` instead of `get_logger('datadog.dogstatsd')` so we do
#      not conflict with any non-vendored datadog.dogstatsd logger
from ..internal.logger import get_logger
from .dogstatsd import base


base.log = get_logger("ddtrace.vendor.dogstatsd")
