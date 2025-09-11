"""
The pytest-bdd integration traces executions of scenarios and steps.

Enabling
~~~~~~~~

Please follow the instructions for enabling `pytest` integration.

.. note::
   The ddtrace.pytest_bdd plugin for pytest-bdd has the side effect of importing
   the ddtrace package and starting a global tracer.

   If this is causing issues for your pytest-bdd runs where traced execution of
   tests is not enabled, you can deactivate the plugin::

     [pytest]
     addopts = -p no:ddtrace.pytest_bdd

   See the `pytest documentation
   <https://docs.pytest.org/en/7.1.x/how-to/plugins.html#deactivating-unregistering-a-plugin-by-name>`_
   for more details.

"""

from ddtrace import config


# pytest-bdd default settings
config._add(
    "pytest_bdd",
    dict(
        _default_service="pytest_bdd",
    ),
)


def get_version():
    # type: () -> str
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata  # type: ignore[no-redef]

    return str(importlib_metadata.version("pytest-bdd"))


__all__ = ["get_version"]
