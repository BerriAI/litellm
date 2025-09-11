# Copyright 2022 Google LLC
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


class BigQueryError(Exception):
    """Base class for all custom exceptions defined by the BigQuery client."""


class LegacyBigQueryStorageError(BigQueryError):
    """Raised when too old a version of BigQuery Storage extra is detected at runtime."""


class LegacyPyarrowError(BigQueryError):
    """Raised when too old a version of pyarrow package is detected at runtime."""


class BigQueryStorageNotFoundError(BigQueryError):
    """Raised when BigQuery Storage extra is not installed when trying to
    import it.
    """


class LegacyPandasError(BigQueryError):
    """Raised when too old a version of pandas package is detected at runtime."""
