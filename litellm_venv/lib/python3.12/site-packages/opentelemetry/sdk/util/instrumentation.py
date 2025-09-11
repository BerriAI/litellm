# Copyright The OpenTelemetry Authors
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
from json import dumps
from typing import Optional

from deprecated import deprecated


class InstrumentationInfo:
    """Immutable information about an instrumentation library module.

    See `opentelemetry.trace.TracerProvider.get_tracer` for the meaning of these
    properties.
    """

    __slots__ = ("_name", "_version", "_schema_url")

    @deprecated(version="1.11.1", reason="You should use InstrumentationScope")
    def __init__(
        self,
        name: str,
        version: Optional[str] = None,
        schema_url: Optional[str] = None,
    ):
        self._name = name
        self._version = version
        if schema_url is None:
            schema_url = ""
        self._schema_url = schema_url

    def __repr__(self):
        return (
            f"{type(self).__name__}({self._name}, {self._version}, {self._schema_url})"
        )

    def __hash__(self):
        return hash((self._name, self._version, self._schema_url))

    def __eq__(self, value):
        return type(value) is type(self) and (
            self._name,
            self._version,
            self._schema_url,
        ) == (value._name, value._version, value._schema_url)

    def __lt__(self, value):
        if type(value) is not type(self):
            return NotImplemented
        return (self._name, self._version, self._schema_url) < (
            value._name,
            value._version,
            value._schema_url,
        )

    @property
    def schema_url(self) -> Optional[str]:
        return self._schema_url

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def name(self) -> str:
        return self._name


class InstrumentationScope:
    """A logical unit of the application code with which the emitted telemetry can be
    associated.

    See `opentelemetry.trace.TracerProvider.get_tracer` for the meaning of these
    properties.
    """

    __slots__ = ("_name", "_version", "_schema_url")

    def __init__(
        self,
        name: str,
        version: Optional[str] = None,
        schema_url: Optional[str] = None,
    ) -> None:
        self._name = name
        self._version = version
        if schema_url is None:
            schema_url = ""
        self._schema_url = schema_url

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._name}, {self._version}, {self._schema_url})"
        )

    def __hash__(self) -> int:
        return hash((self._name, self._version, self._schema_url))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, InstrumentationScope):
            return NotImplemented
        return (self._name, self._version, self._schema_url) == (
            value._name,
            value._version,
            value._schema_url,
        )

    def __lt__(self, value: object) -> bool:
        if not isinstance(value, InstrumentationScope):
            return NotImplemented
        return (self._name, self._version, self._schema_url) < (
            value._name,
            value._version,
            value._schema_url,
        )

    @property
    def schema_url(self) -> Optional[str]:
        return self._schema_url

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def name(self) -> str:
        return self._name

    def to_json(self, indent=4) -> str:
        return dumps(
            {
                "name": self._name,
                "version": self._version,
                "schema_url": self._schema_url,
            },
            indent=indent,
        )
