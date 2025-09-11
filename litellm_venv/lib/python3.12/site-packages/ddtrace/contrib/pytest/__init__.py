"""
The pytest integration traces test executions.

Enabling
~~~~~~~~

Enable traced execution of tests using ``pytest`` runner by
running ``pytest --ddtrace`` or by modifying any configuration
file read by pytest (``pytest.ini``, ``setup.cfg``, ...)::

    [pytest]
    ddtrace = 1


If you need to disable it, the option ``--no-ddtrace`` will take
precedence over ``--ddtrace`` and (``pytest.ini``, ``setup.cfg``, ...)

You can enable all integrations by using the ``--ddtrace-patch-all`` option
alongside ``--ddtrace`` or by adding this to your configuration::

    [pytest]
    ddtrace = 1
    ddtrace-patch-all = 1


.. note::
   The ddtrace plugin for pytest has the side effect of importing the ddtrace
   package and starting a global tracer.

   If this is causing issues for your pytest runs where traced execution of
   tests is not enabled, you can deactivate the plugin::

     [pytest]
     addopts = -p no:ddtrace

   See the `pytest documentation
   <https://docs.pytest.org/en/7.1.x/how-to/plugins.html#deactivating-unregistering-a-plugin-by-name>`_
   for more details.


Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.pytest["service"]

   The service name reported by default for pytest traces.

   This option can also be set with the integration specific ``DD_PYTEST_SERVICE`` environment
   variable, or more generally with the `DD_SERVICE` environment variable.

   Default: Name of the repository being tested, otherwise ``"pytest"`` if the repository name cannot be found.


.. py:data:: ddtrace.config.pytest["operation_name"]

   The operation name reported by default for pytest traces.

   This option can also be set with the ``DD_PYTEST_OPERATION_NAME`` environment
   variable.

   Default: ``"pytest.test"``
"""

import os

from ddtrace import config


# pytest default settings
config._add(
    "pytest",
    dict(
        _default_service="pytest",
        operation_name=os.getenv("DD_PYTEST_OPERATION_NAME", default="pytest.test"),
    ),
)


def get_version():
    # type: () -> str
    import pytest

    return pytest.__version__


__all__ = ["get_version"]
