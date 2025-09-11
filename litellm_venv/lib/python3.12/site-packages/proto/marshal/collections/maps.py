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

import collections

from proto.utils import cached_property
from google.protobuf.message import Message


class MapComposite(collections.abc.MutableMapping):
    """A view around a mutable sequence in protocol buffers.

    This implements the full Python MutableMapping interface, but all methods
    modify the underlying field container directly.
    """

    @cached_property
    def _pb_type(self):
        """Return the protocol buffer type for this sequence."""
        # Huzzah, another hack. Still less bad than RepeatedComposite.
        return type(self.pb.GetEntryClass()().value)

    def __init__(self, sequence, *, marshal):
        """Initialize a wrapper around a protobuf map.

        Args:
            sequence: A protocol buffers map.
            marshal (~.MarshalRegistry): An instantiated marshal, used to
                convert values going to and from this map.
        """
        self._pb = sequence
        self._marshal = marshal

    def __contains__(self, key):
        # Protocol buffers is so permissive that querying for the existence
        # of a key will in of itself create it.
        #
        # By taking a tuple of the keys and querying that, we avoid sending
        # the lookup to protocol buffers and therefore avoid creating the key.
        return key in tuple(self.keys())

    def __getitem__(self, key):
        # We handle raising KeyError ourselves, because otherwise protocol
        # buffers will create the key if it does not exist.
        if key not in self:
            raise KeyError(key)
        return self._marshal.to_python(self._pb_type, self.pb[key])

    def __setitem__(self, key, value):
        pb_value = self._marshal.to_proto(self._pb_type, value, strict=True)
        # Directly setting a key is not allowed; however, protocol buffers
        # is so permissive that querying for the existence of a key will in
        # of itself create it.
        #
        # Therefore, we create a key that way (clearing any fields that may
        # be set) and then merge in our values.
        self.pb[key].Clear()
        self.pb[key].MergeFrom(pb_value)

    def __delitem__(self, key):
        self.pb.pop(key)

    def __len__(self):
        return len(self.pb)

    def __iter__(self):
        return iter(self.pb)

    @property
    def pb(self):
        return self._pb
