# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helper functions for verifying versions of installed modules."""

import sys
from typing import Any

import packaging.version

from google.cloud.bigquery import exceptions


_MIN_PYARROW_VERSION = packaging.version.Version("3.0.0")
_MIN_BQ_STORAGE_VERSION = packaging.version.Version("2.0.0")
_BQ_STORAGE_OPTIONAL_READ_SESSION_VERSION = packaging.version.Version("2.6.0")
_MIN_PANDAS_VERSION = packaging.version.Version("1.1.0")

_MIN_PANDAS_VERSION_RANGE = packaging.version.Version("1.5.0")
_MIN_PYARROW_VERSION_RANGE = packaging.version.Version("10.0.1")


class PyarrowVersions:
    """Version comparisons for pyarrow package."""

    def __init__(self):
        self._installed_version = None

    @property
    def installed_version(self) -> packaging.version.Version:
        """Return the parsed version of pyarrow."""
        if self._installed_version is None:
            import pyarrow  # type: ignore

            self._installed_version = packaging.version.parse(
                # Use 0.0.0, since it is earlier than any released version.
                # Legacy versions also have the same property, but
                # creating a LegacyVersion has been deprecated.
                # https://github.com/pypa/packaging/issues/321
                getattr(pyarrow, "__version__", "0.0.0")
            )

        return self._installed_version

    @property
    def use_compliant_nested_type(self) -> bool:
        return self.installed_version.major >= 4

    def try_import(self, raise_if_error: bool = False) -> Any:
        """Verifies that a recent enough version of pyarrow extra is installed.

        The function assumes that pyarrow extra is installed, and should thus
        be used in places where this assumption holds.

        Because `pip` can install an outdated version of this extra despite
        the constraints in `setup.py`, the calling code can use this helper
        to verify the version compatibility at runtime.

        Returns:
            The ``pyarrow`` module or ``None``.

        Raises:
            exceptions.LegacyPyarrowError:
                If the pyarrow package is outdated and ``raise_if_error`` is
                ``True``.
        """
        try:
            import pyarrow
        except ImportError as exc:
            if raise_if_error:
                raise exceptions.LegacyPyarrowError(
                    "pyarrow package not found. Install pyarrow version >="
                    f" {_MIN_PYARROW_VERSION}."
                ) from exc
            return None

        if self.installed_version < _MIN_PYARROW_VERSION:
            if raise_if_error:
                msg = (
                    "Dependency pyarrow is outdated, please upgrade"
                    f" it to version >= {_MIN_PYARROW_VERSION}"
                    f" (version found: {self.installed_version})."
                )
                raise exceptions.LegacyPyarrowError(msg)
            return None

        return pyarrow


PYARROW_VERSIONS = PyarrowVersions()


class BQStorageVersions:
    """Version comparisons for google-cloud-bigqueyr-storage package."""

    def __init__(self):
        self._installed_version = None

    @property
    def installed_version(self) -> packaging.version.Version:
        """Return the parsed version of google-cloud-bigquery-storage."""
        if self._installed_version is None:
            from google.cloud import bigquery_storage

            self._installed_version = packaging.version.parse(
                # Use 0.0.0, since it is earlier than any released version.
                # Legacy versions also have the same property, but
                # creating a LegacyVersion has been deprecated.
                # https://github.com/pypa/packaging/issues/321
                getattr(bigquery_storage, "__version__", "0.0.0")
            )

        return self._installed_version  # type: ignore

    @property
    def is_read_session_optional(self) -> bool:
        """True if read_session is optional to rows().

        See: https://github.com/googleapis/python-bigquery-storage/pull/228
        """
        return self.installed_version >= _BQ_STORAGE_OPTIONAL_READ_SESSION_VERSION

    def try_import(self, raise_if_error: bool = False) -> Any:
        """Tries to import the bigquery_storage module, and returns results
        accordingly. It also verifies the module version is recent enough.

        If the import succeeds, returns the ``bigquery_storage`` module.

        If the import fails,
        returns ``None`` when ``raise_if_error == False``,
        raises Error when ``raise_if_error == True``.

        Returns:
            The ``bigquery_storage`` module or ``None``.

        Raises:
            exceptions.BigQueryStorageNotFoundError:
                If google-cloud-bigquery-storage is not installed
            exceptions.LegacyBigQueryStorageError:
                If google-cloud-bigquery-storage package is outdated
        """
        try:
            from google.cloud import bigquery_storage  # type: ignore
        except ImportError:
            if raise_if_error:
                msg = (
                    "Package google-cloud-bigquery-storage not found. "
                    "Install google-cloud-bigquery-storage version >= "
                    f"{_MIN_BQ_STORAGE_VERSION}."
                )
                raise exceptions.BigQueryStorageNotFoundError(msg)
            return None

        if self.installed_version < _MIN_BQ_STORAGE_VERSION:
            if raise_if_error:
                msg = (
                    "Dependency google-cloud-bigquery-storage is outdated, "
                    f"please upgrade it to version >= {_MIN_BQ_STORAGE_VERSION} "
                    f"(version found: {self.installed_version})."
                )
                raise exceptions.LegacyBigQueryStorageError(msg)
            return None

        return bigquery_storage


BQ_STORAGE_VERSIONS = BQStorageVersions()


class PandasVersions:
    """Version comparisons for pandas package."""

    def __init__(self):
        self._installed_version = None

    @property
    def installed_version(self) -> packaging.version.Version:
        """Return the parsed version of pandas"""
        if self._installed_version is None:
            import pandas  # type: ignore

            self._installed_version = packaging.version.parse(
                # Use 0.0.0, since it is earlier than any released version.
                # Legacy versions also have the same property, but
                # creating a LegacyVersion has been deprecated.
                # https://github.com/pypa/packaging/issues/321
                getattr(pandas, "__version__", "0.0.0")
            )

        return self._installed_version

    def try_import(self, raise_if_error: bool = False) -> Any:
        """Verify that a recent enough version of pandas extra is installed.
        The function assumes that pandas extra is installed, and should thus
        be used in places where this assumption holds.
        Because `pip` can install an outdated version of this extra despite
        the constraints in `setup.py`, the calling code can use this helper
        to verify the version compatibility at runtime.
        Returns:
            The ``pandas`` module or ``None``.
        Raises:
            exceptions.LegacyPandasError:
                If the pandas package is outdated and ``raise_if_error`` is
                ``True``.
        """
        try:
            import pandas
        except ImportError as exc:
            if raise_if_error:
                raise exceptions.LegacyPandasError(
                    "pandas package not found. Install pandas version >="
                    f" {_MIN_PANDAS_VERSION}"
                ) from exc
            return None

        if self.installed_version < _MIN_PANDAS_VERSION:
            if raise_if_error:
                msg = (
                    "Dependency pandas is outdated, please upgrade"
                    f" it to version >= {_MIN_PANDAS_VERSION}"
                    f" (version found: {self.installed_version})."
                )
                raise exceptions.LegacyPandasError(msg)
            return None

        return pandas


PANDAS_VERSIONS = PandasVersions()

# Since RANGE support in pandas requires specific versions
# of both pyarrow and pandas, we make this a separate
# constant instead of as a property of PANDAS_VERSIONS
# or PYARROW_VERSIONS.
SUPPORTS_RANGE_PYARROW = (
    PANDAS_VERSIONS.try_import() is not None
    and PANDAS_VERSIONS.installed_version >= _MIN_PANDAS_VERSION_RANGE
    and PYARROW_VERSIONS.try_import() is not None
    and PYARROW_VERSIONS.installed_version >= _MIN_PYARROW_VERSION_RANGE
)


def extract_runtime_version():
    # Retrieve the version information
    version_info = sys.version_info

    # Extract the major, minor, and micro components
    major = version_info.major
    minor = version_info.minor
    micro = version_info.micro

    # Display the version number in a clear format
    return major, minor, micro
