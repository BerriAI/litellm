"""
`Datadog Source Code Integration`__ is supported for Git by the addition of the
repository URL and commit hash in the Python package metadata field
``Project-URL`` with name ``source_code_link``.

Format of ``source_code_link``: ``<repository url>#<commit hash>``


setuptools
----------

The ``ddtrace`` provides automatic instrumentation of ``setuptools`` to embed
the source code link into the project metadata. ``ddtrace`` has to be installed
as a build dependency.

Packages with ``pyproject.toml`` can update the build system requirements::

  [build-system]
  requires = ["setuptools", "ddtrace"]
  build-backend = "setuptools.build_meta"


The instrumentation of ``setuptools`` can be automatically enabled to embed the
source code link with a one-line import in ``setup.py`` (before setuptools import)::

   import ddtrace.sourcecode.setuptools_auto
   from setuptools import setup

   setup(
       name="mypackage",
       version="0.0.1",
       #...
   )

.. __: https://docs.datadoghq.com/integrations/guide/source-code-integration/
"""
