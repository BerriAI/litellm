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
import inspect
import logging

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import reflection

from proto.marshal.rules.message import MessageRule

log = logging.getLogger("_FileInfo")


class _FileInfo(
    collections.namedtuple(
        "_FileInfo",
        ["descriptor", "messages", "enums", "name", "nested", "nested_enum"],
    )
):
    registry = {}  # Mapping[str, '_FileInfo']

    @classmethod
    def maybe_add_descriptor(cls, filename, package):
        descriptor = cls.registry.get(filename)
        if not descriptor:
            descriptor = cls.registry[filename] = cls(
                descriptor=descriptor_pb2.FileDescriptorProto(
                    name=filename,
                    package=package,
                    syntax="proto3",
                ),
                enums=collections.OrderedDict(),
                messages=collections.OrderedDict(),
                name=filename,
                nested={},
                nested_enum={},
            )

        return descriptor

    @staticmethod
    def proto_file_name(name):
        return "{0}.proto".format(name.replace(".", "/"))

    def _get_manifest(self, new_class):
        module = inspect.getmodule(new_class)
        if hasattr(module, "__protobuf__"):
            return frozenset(module.__protobuf__.manifest)

        return frozenset()

    def _get_remaining_manifest(self, new_class):
        return self._get_manifest(new_class) - {new_class.__name__}

    def _calculate_salt(self, new_class, fallback):
        manifest = self._get_manifest(new_class)
        if manifest and new_class.__name__ not in manifest:
            log.warning(
                "proto-plus module {module} has a declared manifest but {class_name} is not in it".format(
                    module=inspect.getmodule(new_class).__name__,
                    class_name=new_class.__name__,
                )
            )

        return "" if new_class.__name__ in manifest else (fallback or "").lower()

    def generate_file_pb(self, new_class, fallback_salt=""):
        """Generate the descriptors for all protos in the file.

        This method takes the file descriptor attached to the parent
        message and generates the immutable descriptors for all of the
        messages in the file descriptor. (This must be done in one fell
        swoop for immutability and to resolve proto cross-referencing.)

        This is run automatically when the last proto in the file is
        generated, as determined by the module's __all__ tuple.
        """
        pool = descriptor_pool.Default()

        # Salt the filename in the descriptor.
        # This allows re-use of the filename by other proto messages if
        # needed (e.g. if __all__ is not used).
        salt = self._calculate_salt(new_class, fallback_salt)
        self.descriptor.name = "{name}.proto".format(
            name="_".join([self.descriptor.name[:-6], salt]).rstrip("_"),
        )

        # Add the file descriptor.
        pool.Add(self.descriptor)

        # Adding the file descriptor to the pool created a descriptor for
        # each message; go back through our wrapper messages and associate
        # them with the internal protobuf version.
        for full_name, proto_plus_message in self.messages.items():
            # Get the descriptor from the pool, and create the protobuf
            # message based on it.
            descriptor = pool.FindMessageTypeByName(full_name)
            pb_message = reflection.GeneratedProtocolMessageType(
                descriptor.name,
                (message.Message,),
                {"DESCRIPTOR": descriptor, "__module__": None},
            )

            # Register the message with the marshal so it is wrapped
            # appropriately.
            #
            # We do this here (rather than at class creation) because it
            # is not until this point that we have an actual protobuf
            # message subclass, which is what we need to use.
            proto_plus_message._meta._pb = pb_message
            proto_plus_message._meta.marshal.register(
                pb_message, MessageRule(pb_message, proto_plus_message)
            )

            # Iterate over any fields on the message and, if their type
            # is a message still referenced as a string, resolve the reference.
            for field in proto_plus_message._meta.fields.values():
                if field.message and isinstance(field.message, str):
                    field.message = self.messages[field.message]
                elif field.enum and isinstance(field.enum, str):
                    field.enum = self.enums[field.enum]

        # Same thing for enums
        for full_name, proto_plus_enum in self.enums.items():
            descriptor = pool.FindEnumTypeByName(full_name)
            proto_plus_enum._meta.pb = descriptor

        # We no longer need to track this file's info; remove it from
        # the module's registry and from this object.
        self.registry.pop(self.name)

    def ready(self, new_class):
        """Return True if a file descriptor may added, False otherwise.

        This determine if all the messages that we plan to create have been
        created, as best as we are able.

        Since messages depend on one another, we create descriptor protos
        (which reference each other using strings) and wait until we have
        built everything that is going to be in the module, and then
        use the descriptor protos to instantiate the actual descriptors in
        one fell swoop.

        Args:
            new_class (~.MessageMeta): The new class currently undergoing
                creation.
        """
        # If there are any nested descriptors that have not been assigned to
        # the descriptors that should contain them, then we are not ready.
        if len(self.nested) or len(self.nested_enum):
            return False

        # If there are any unresolved fields (fields with a composite message
        # declared as a string), ensure that the corresponding message is
        # declared.
        for field in self.unresolved_fields:
            if (field.message and field.message not in self.messages) or (
                field.enum and field.enum not in self.enums
            ):
                return False

        # If the module in which this class is defined provides a
        # __protobuf__ property, it may have a manifest.
        #
        # Do not generate the file descriptor until every member of the
        # manifest has been populated.
        module = inspect.getmodule(new_class)
        manifest = self._get_remaining_manifest(new_class)

        # We are ready if all members have been populated.
        return all(hasattr(module, i) for i in manifest)

    @property
    def unresolved_fields(self):
        """Return fields with referencing message types as strings."""
        for proto_plus_message in self.messages.values():
            for field in proto_plus_message._meta.fields.values():
                if (field.message and isinstance(field.message, str)) or (
                    field.enum and isinstance(field.enum, str)
                ):
                    yield field
