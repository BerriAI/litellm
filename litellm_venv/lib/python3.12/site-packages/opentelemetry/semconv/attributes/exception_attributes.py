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


EXCEPTION_ESCAPED = "exception.escaped"
"""
SHOULD be set to true if the exception event is recorded at a point where it is known that the exception is escaping the scope of the span.
Note: An exception is considered to have escaped (or left) the scope of a span,
    if that span is ended while the exception is still logically "in flight".
    This may be actually "in flight" in some languages (e.g. if the exception
    is passed to a Context manager's `__exit__` method in Python) but will
    usually be caught at the point of recording the exception in most languages.

    It is usually not possible to determine at the point where an exception is thrown
    whether it will escape the scope of a span.
    However, it is trivial to know that an exception
    will escape, if one checks for an active exception just before ending the span,
    as done in the [example for recording span exceptions](#recording-an-exception).

    It follows that an exception may still escape the scope of the span
    even if the `exception.escaped` attribute was not set or set to false,
    since the event might have been recorded at a time where it was not
    clear whether the exception will escape.
"""

EXCEPTION_MESSAGE = "exception.message"
"""
The exception message.
"""

EXCEPTION_STACKTRACE = "exception.stacktrace"
"""
A stacktrace as a string in the natural representation for the language runtime. The representation is to be determined and documented by each language SIG.
"""

EXCEPTION_TYPE = "exception.type"
"""
The type of the exception (its fully-qualified class name, if applicable). The dynamic type of the exception should be preferred over the static type in languages that support it.
"""
