# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class MessageRule:
    """A marshal for converting between a descriptor and proto.Message."""

    def __init__(self, descriptor: type, wrapper: type):
        self._descriptor = descriptor
        self._wrapper = wrapper

    def to_python(self, value, *, absent: bool = None):
        if isinstance(value, self._descriptor):
            return self._wrapper.wrap(value)
        return value

    def to_proto(self, value):
        if isinstance(value, self._wrapper):
            return self._wrapper.pb(value)
        if isinstance(value, dict) and not self.is_map:
            # We need to use the wrapper's marshaling to handle
            # potentially problematic nested messages.
            try:
                # Try the fast path first.
                return self._descriptor(**value)
            except (TypeError, ValueError, AttributeError) as ex:
                # If we have a TypeError, ValueError or AttributeError,
                # try the slow path in case the error
                # was:
                # - an int64/string issue.
                # - a missing key issue in case a key only exists with a `_` suffix.
                #   See related issue: https://github.com/googleapis/python-api-core/issues/227.
                # - a missing key issue due to nested struct. See: https://github.com/googleapis/proto-plus-python/issues/424.
                # - a missing key issue due to nested duration. See: https://github.com/googleapis/google-cloud-python/issues/13350.
                return self._wrapper(value)._pb
        return value

    @property
    def is_map(self):
        """Return True if the descriptor is a map entry, False otherwise."""
        desc = self._descriptor.DESCRIPTOR
        return desc.has_options and desc.GetOptions().map_entry
