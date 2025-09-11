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
import copy
from typing import Iterable

from proto.utils import cached_property


class Repeated(collections.abc.MutableSequence):
    """A view around a mutable sequence in protocol buffers.

    This implements the full Python MutableSequence interface, but all methods
    modify the underlying field container directly.
    """

    def __init__(self, sequence, *, marshal, proto_type=None):
        """Initialize a wrapper around a protobuf repeated field.

        Args:
            sequence: A protocol buffers repeated field.
            marshal (~.MarshalRegistry): An instantiated marshal, used to
                convert values going to and from this map.
        """
        self._pb = sequence
        self._marshal = marshal
        self._proto_type = proto_type

    def __copy__(self):
        """Copy this object and return the copy."""
        return type(self)(self.pb[:], marshal=self._marshal)

    def __delitem__(self, key):
        """Delete the given item."""
        del self.pb[key]

    def __eq__(self, other):
        if hasattr(other, "pb"):
            return tuple(self.pb) == tuple(other.pb)
        return tuple(self.pb) == tuple(other) if isinstance(other, Iterable) else False

    def __getitem__(self, key):
        """Return the given item."""
        return self.pb[key]

    def __len__(self):
        """Return the length of the sequence."""
        return len(self.pb)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return repr([*self])

    def __setitem__(self, key, value):
        self.pb[key] = value

    def insert(self, index: int, value):
        """Insert ``value`` in the sequence before ``index``."""
        self.pb.insert(index, value)

    def sort(self, *, key: str = None, reverse: bool = False):
        """Stable sort *IN PLACE*."""
        self.pb.sort(key=key, reverse=reverse)

    @property
    def pb(self):
        return self._pb


class RepeatedComposite(Repeated):
    """A view around a mutable sequence of messages in protocol buffers.

    This implements the full Python MutableSequence interface, but all methods
    modify the underlying field container directly.
    """

    @cached_property
    def _pb_type(self):
        """Return the protocol buffer type for this sequence."""
        # Provide the marshal-given proto_type, if any.
        # Used for RepeatedComposite of Enum.
        if self._proto_type is not None:
            return self._proto_type

        # There is no public-interface mechanism to determine the type
        # of what should go in the list (and the C implementation seems to
        # have no exposed mechanism at all).
        #
        # If the list has members, use the existing list members to
        # determine the type.
        if len(self.pb) > 0:
            return type(self.pb[0])

        # We have no members in the list, so we get the type from the attributes.
        if hasattr(self.pb, "_message_descriptor") and hasattr(
            self.pb._message_descriptor, "_concrete_class"
        ):
            return self.pb._message_descriptor._concrete_class

        # Fallback logic in case attributes are not available
        # In order to get the type, we create a throw-away copy and add a
        # blank member to it.
        canary = copy.deepcopy(self.pb).add()
        return type(canary)

    def __eq__(self, other):
        if super().__eq__(other):
            return True
        return (
            tuple([i for i in self]) == tuple(other)
            if isinstance(other, Iterable)
            else False
        )

    def __getitem__(self, key):
        return self._marshal.to_python(self._pb_type, self.pb[key])

    def __setitem__(self, key, value):
        # The underlying protocol buffer does not define __setitem__, so we
        # have to implement all the operations on our own.

        # If ``key`` is an integer, as in list[index] = value:
        if isinstance(key, int):
            if -len(self) <= key < len(self):
                self.pop(key)  # Delete the old item.
                self.insert(key, value)  # Insert the new item in its place.
            else:
                raise IndexError("list assignment index out of range")

        # If ``key`` is a slice object, as in list[start:stop:step] = [values]:
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))

            if not isinstance(value, collections.abc.Iterable):
                raise TypeError("can only assign an iterable")

            if step == 1:  # Is not an extended slice.
                # Assign all the new values to the sliced part, replacing the
                # old values, if any, and unconditionally inserting those
                # values whose indices already exceed the slice length.
                for index, item in enumerate(value):
                    if start + index < stop:
                        self.pop(start + index)
                    self.insert(start + index, item)

                # If there are less values than the length of the slice, remove
                # the remaining elements so that the slice adapts to the
                # newly provided values.
                for _ in range(stop - start - len(value)):
                    self.pop(start + len(value))

            else:  # Is an extended slice.
                indices = range(start, stop, step)

                if len(value) != len(indices):  # XXX: Use PEP 572 on 3.8+
                    raise ValueError(
                        f"attempt to assign sequence of size "
                        f"{len(value)} to extended slice of size "
                        f"{len(indices)}"
                    )

                # Assign each value to its index, calling this function again
                # with individual integer indexes that get processed above.
                for index, item in zip(indices, value):
                    self[index] = item

        else:
            raise TypeError(
                f"list indices must be integers or slices, not {type(key).__name__}"
            )

    def insert(self, index: int, value):
        """Insert ``value`` in the sequence before ``index``."""
        pb_value = self._marshal.to_proto(self._pb_type, value)
        self.pb.insert(index, pb_value)
