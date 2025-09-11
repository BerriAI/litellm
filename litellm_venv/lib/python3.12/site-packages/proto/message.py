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
import collections.abc
import copy
import re
from typing import Any, Dict, List, Optional, Type
import warnings

import google.protobuf
from google.protobuf import descriptor_pb2
from google.protobuf import message
from google.protobuf.json_format import MessageToDict, MessageToJson, Parse

from proto import _file_info
from proto import _package_info
from proto.fields import Field
from proto.fields import MapField
from proto.fields import RepeatedField
from proto.marshal import Marshal
from proto.primitives import ProtoType
from proto.utils import has_upb


PROTOBUF_VERSION = google.protobuf.__version__

_upb = has_upb()  # Important to cache result here.


class MessageMeta(type):
    """A metaclass for building and registering Message subclasses."""

    def __new__(mcls, name, bases, attrs):
        # Do not do any special behavior for Message itself.
        if not bases:
            return super().__new__(mcls, name, bases, attrs)

        # Get the essential information about the proto package, and where
        # this component belongs within the file.
        package, marshal = _package_info.compile(name, attrs)

        # Determine the local path of this proto component within the file.
        local_path = tuple(attrs.get("__qualname__", name).split("."))

        # Sanity check: We get the wrong full name if a class is declared
        # inside a function local scope; correct this.
        if "<locals>" in local_path:
            ix = local_path.index("<locals>")
            local_path = local_path[: ix - 1] + local_path[ix + 1 :]

        # Determine the full name in protocol buffers.
        full_name = ".".join((package,) + local_path).lstrip(".")

        # Special case: Maps. Map fields are special; they are essentially
        # shorthand for a nested message and a repeated field of that message.
        # Decompose each map into its constituent form.
        # https://developers.google.com/protocol-buffers/docs/proto3#maps
        map_fields = {}
        for key, field in attrs.items():
            if not isinstance(field, MapField):
                continue

            # Determine the name of the entry message.
            msg_name = "{pascal_key}Entry".format(
                pascal_key=re.sub(
                    r"_\w",
                    lambda m: m.group()[1:].upper(),
                    key,
                ).replace(key[0], key[0].upper(), 1),
            )

            # Create the "entry" message (with the key and value fields).
            #
            # Note: We instantiate an ordered dictionary here and then
            # attach key and value in order to ensure that the fields are
            # iterated in the correct order when the class is created.
            # This is only an issue in Python 3.5, where the order is
            # random (and the wrong order causes the pool to refuse to add
            # the descriptor because reasons).
            entry_attrs = collections.OrderedDict(
                {
                    "__module__": attrs.get("__module__", None),
                    "__qualname__": "{prefix}.{name}".format(
                        prefix=attrs.get("__qualname__", name),
                        name=msg_name,
                    ),
                    "_pb_options": {"map_entry": True},
                }
            )
            entry_attrs["key"] = Field(field.map_key_type, number=1)
            entry_attrs["value"] = Field(
                field.proto_type,
                number=2,
                enum=field.enum,
                message=field.message,
            )
            map_fields[msg_name] = MessageMeta(msg_name, (Message,), entry_attrs)

            # Create the repeated field for the entry message.
            map_fields[key] = RepeatedField(
                ProtoType.MESSAGE,
                number=field.number,
                message=map_fields[msg_name],
            )

        # Add the new entries to the attrs
        attrs.update(map_fields)

        # Okay, now we deal with all the rest of the fields.
        # Iterate over all the attributes and separate the fields into
        # their own sequence.
        fields = []
        new_attrs = {}
        oneofs = collections.OrderedDict()
        proto_imports = set()
        index = 0
        for key, field in attrs.items():
            # Sanity check: If this is not a field, do nothing.
            if not isinstance(field, Field):
                # The field objects themselves should not be direct attributes.
                new_attrs[key] = field
                continue

            # Add data that the field requires that we do not take in the
            # constructor because we can derive it from the metaclass.
            # (The goal is to make the declaration syntax as nice as possible.)
            field.mcls_data = {
                "name": key,
                "parent_name": full_name,
                "index": index,
                "package": package,
            }

            # Add the field to the list of fields.
            fields.append(field)
            # If this field is part of a "oneof", ensure the oneof itself
            # is represented.
            if field.oneof:
                # Keep a running tally of the index of each oneof, and assign
                # that index to the field's descriptor.
                oneofs.setdefault(field.oneof, len(oneofs))
                field.descriptor.oneof_index = oneofs[field.oneof]

            # If this field references a message, it may be from another
            # proto file; ensure we know about the import (to faithfully
            # construct our file descriptor proto).
            if field.message and not isinstance(field.message, str):
                field_msg = field.message
                if hasattr(field_msg, "pb") and callable(field_msg.pb):
                    field_msg = field_msg.pb()
                # Sanity check: The field's message may not yet be defined if
                # it was a Message defined in the same file, and the file
                # descriptor proto has not yet been generated.
                #
                # We do nothing in this situation; everything will be handled
                # correctly when the file descriptor is created later.
                if field_msg:
                    proto_imports.add(field_msg.DESCRIPTOR.file.name)

            # Same thing, but for enums.
            elif field.enum and not isinstance(field.enum, str):
                field_enum = (
                    field.enum._meta.pb
                    if hasattr(field.enum, "_meta")
                    else field.enum.DESCRIPTOR
                )

                if field_enum:
                    proto_imports.add(field_enum.file.name)

            # Increment the field index counter.
            index += 1

        # As per descriptor.proto, all synthetic oneofs must be ordered after
        # 'real' oneofs.
        opt_attrs = {}
        for field in fields:
            if field.optional:
                field.oneof = "_{}".format(field.name)
                field.descriptor.oneof_index = oneofs[field.oneof] = len(oneofs)
                opt_attrs[field.name] = field.name

        # Generating a metaclass dynamically provides class attributes that
        # instances can't see. This provides idiomatically named constants
        # that enable the following pattern to check for field presence:
        #
        # class MyMessage(proto.Message):
        #     field = proto.Field(proto.INT32, number=1, optional=True)
        #
        # m = MyMessage()
        # MyMessage.field in m
        if opt_attrs:
            mcls = type("AttrsMeta", (mcls,), opt_attrs)

        # Determine the filename.
        # We determine an appropriate proto filename based on the
        # Python module.
        filename = _file_info._FileInfo.proto_file_name(
            new_attrs.get("__module__", name.lower())
        )

        # Get or create the information about the file, including the
        # descriptor to which the new message descriptor shall be added.
        file_info = _file_info._FileInfo.maybe_add_descriptor(filename, package)

        # Ensure any imports that would be necessary are assigned to the file
        # descriptor proto being created.
        for proto_import in proto_imports:
            if proto_import not in file_info.descriptor.dependency:
                file_info.descriptor.dependency.append(proto_import)

        # Retrieve any message options.
        opts = descriptor_pb2.MessageOptions(**new_attrs.pop("_pb_options", {}))

        # Create the underlying proto descriptor.
        desc = descriptor_pb2.DescriptorProto(
            name=name,
            field=[i.descriptor for i in fields],
            oneof_decl=[
                descriptor_pb2.OneofDescriptorProto(name=i) for i in oneofs.keys()
            ],
            options=opts,
        )

        # If any descriptors were nested under this one, they need to be
        # attached as nested types here.
        child_paths = [p for p in file_info.nested.keys() if local_path == p[:-1]]
        for child_path in child_paths:
            desc.nested_type.add().MergeFrom(file_info.nested.pop(child_path))

        # Same thing, but for enums
        child_paths = [p for p in file_info.nested_enum.keys() if local_path == p[:-1]]
        for child_path in child_paths:
            desc.enum_type.add().MergeFrom(file_info.nested_enum.pop(child_path))

        # Add the descriptor to the file if it is a top-level descriptor,
        # or to a "holding area" for nested messages otherwise.
        if len(local_path) == 1:
            file_info.descriptor.message_type.add().MergeFrom(desc)
        else:
            file_info.nested[local_path] = desc

        # Create the MessageInfo instance to be attached to this message.
        new_attrs["_meta"] = _MessageInfo(
            fields=fields,
            full_name=full_name,
            marshal=marshal,
            options=opts,
            package=package,
        )

        # Run the superclass constructor.
        cls = super().__new__(mcls, name, bases, new_attrs)

        # The info class and fields need a reference to the class just created.
        cls._meta.parent = cls
        for field in cls._meta.fields.values():
            field.parent = cls

        # Add this message to the _FileInfo instance; this allows us to
        # associate the descriptor with the message once the descriptor
        # is generated.
        file_info.messages[full_name] = cls

        # Generate the descriptor for the file if it is ready.
        if file_info.ready(new_class=cls):
            file_info.generate_file_pb(new_class=cls, fallback_salt=full_name)

        # Done; return the class.
        return cls

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        return collections.OrderedDict()

    @property
    def meta(cls):
        return cls._meta

    def __dir__(self):
        try:
            names = set(dir(type))
            names.update(
                (
                    "meta",
                    "pb",
                    "wrap",
                    "serialize",
                    "deserialize",
                    "to_json",
                    "from_json",
                    "to_dict",
                    "copy_from",
                )
            )
            desc = self.pb().DESCRIPTOR
            names.update(t.name for t in desc.nested_types)
            names.update(e.name for e in desc.enum_types)

            return names
        except AttributeError:
            return dir(type)

    def pb(cls, obj=None, *, coerce: bool = False):
        """Return the underlying protobuf Message class or instance.

        Args:
            obj: If provided, and an instance of ``cls``, return the
                underlying protobuf instance.
            coerce (bool): If provided, will attempt to coerce ``obj`` to
                ``cls`` if it is not already an instance.
        """
        if obj is None:
            return cls.meta.pb
        if not isinstance(obj, cls):
            if coerce:
                obj = cls(obj)
            else:
                raise TypeError(
                    "%r is not an instance of %s"
                    % (
                        obj,
                        cls.__name__,
                    )
                )
        return obj._pb

    def wrap(cls, pb):
        """Return a Message object that shallowly wraps the descriptor.

        Args:
            pb: A protocol buffer object, such as would be returned by
                :meth:`pb`.
        """
        # Optimized fast path.
        instance = cls.__new__(cls)
        super(cls, instance).__setattr__("_pb", pb)
        return instance

    def serialize(cls, instance) -> bytes:
        """Return the serialized proto.

        Args:
            instance: An instance of this message type, or something
                compatible (accepted by the type's constructor).

        Returns:
            bytes: The serialized representation of the protocol buffer.
        """
        return cls.pb(instance, coerce=True).SerializeToString()

    def deserialize(cls, payload: bytes) -> "Message":
        """Given a serialized proto, deserialize it into a Message instance.

        Args:
            payload (bytes): The serialized proto.

        Returns:
            ~.Message: An instance of the message class against which this
            method was called.
        """
        return cls.wrap(cls.pb().FromString(payload))

    def _warn_if_including_default_value_fields_is_used_protobuf_5(
        cls, including_default_value_fields: Optional[bool]
    ) -> None:
        """
        Warn Protobuf 5.x+ users that `including_default_value_fields` is deprecated if it is set.

        Args:
            including_default_value_fields (Optional(bool)): The value of `including_default_value_fields` set by the user.
        """
        if (
            PROTOBUF_VERSION[0] not in ("3", "4")
            and including_default_value_fields is not None
        ):
            warnings.warn(
                """The argument `including_default_value_fields` has been removed from
                Protobuf 5.x. Please use `always_print_fields_with_no_presence` instead.
                """,
                DeprecationWarning,
            )

    def _raise_if_print_fields_values_are_set_and_differ(
        cls,
        always_print_fields_with_no_presence: Optional[bool],
        including_default_value_fields: Optional[bool],
    ) -> None:
        """
        Raise Exception if both `always_print_fields_with_no_presence` and `including_default_value_fields` are set
            and the values differ.

        Args:
            always_print_fields_with_no_presence (Optional(bool)): The value of `always_print_fields_with_no_presence` set by the user.
            including_default_value_fields (Optional(bool)): The value of `including_default_value_fields` set by the user.
        Returns:
            None
        Raises:
            ValueError: if both `always_print_fields_with_no_presence` and `including_default_value_fields` are set and
                the values differ.
        """
        if (
            always_print_fields_with_no_presence is not None
            and including_default_value_fields is not None
            and always_print_fields_with_no_presence != including_default_value_fields
        ):
            raise ValueError(
                "Arguments `always_print_fields_with_no_presence` and `including_default_value_fields` must match"
            )

    def _normalize_print_fields_without_presence(
        cls,
        always_print_fields_with_no_presence: Optional[bool],
        including_default_value_fields: Optional[bool],
    ) -> bool:
        """
        Return true if fields with no presence should be included in the results.
        By default, fields with no presence will be included in the results
        when both `always_print_fields_with_no_presence` and
        `including_default_value_fields` are not set

        Args:
            always_print_fields_with_no_presence (Optional(bool)): The value of `always_print_fields_with_no_presence` set by the user.
            including_default_value_fields (Optional(bool)): The value of `including_default_value_fields` set by the user.
        Returns:
            None
        Raises:
            ValueError: if both `always_print_fields_with_no_presence` and `including_default_value_fields` are set and
                the values differ.
        """

        cls._warn_if_including_default_value_fields_is_used_protobuf_5(
            including_default_value_fields
        )
        cls._raise_if_print_fields_values_are_set_and_differ(
            always_print_fields_with_no_presence, including_default_value_fields
        )
        # Default to True if neither `always_print_fields_with_no_presence` or `including_default_value_fields` is set
        return (
            (
                always_print_fields_with_no_presence is None
                and including_default_value_fields is None
            )
            or always_print_fields_with_no_presence
            or including_default_value_fields
        )

    def to_json(
        cls,
        instance,
        *,
        use_integers_for_enums=True,
        including_default_value_fields=None,
        preserving_proto_field_name=False,
        sort_keys=False,
        indent=2,
        float_precision=None,
        always_print_fields_with_no_presence=None,
    ) -> str:
        """Given a message instance, serialize it to json

        Args:
            instance: An instance of this message type, or something
                compatible (accepted by the type's constructor).
            use_integers_for_enums (Optional(bool)): An option that determines whether enum
                values should be represented by strings (False) or integers (True).
                Default is True.
            including_default_value_fields (Optional(bool)): Deprecated. Use argument
                `always_print_fields_with_no_presence` instead. An option that
                determines whether the default field values should be included in the results.
                This value must match `always_print_fields_with_no_presence`,
                if both arguments are explicitly set.
            preserving_proto_field_name (Optional(bool)): An option that
                determines whether field name representations preserve
                proto case (snake_case) or use lowerCamelCase. Default is False.
            sort_keys (Optional(bool)): If True, then the output will be sorted by field names.
                Default is False.
            indent (Optional(int)): The JSON object will be pretty-printed with this indent level.
                An indent level of 0 or negative will only insert newlines.
                Pass None for the most compact representation without newlines.
            float_precision (Optional(int)): If set, use this to specify float field valid digits.
                Default is None.
            always_print_fields_with_no_presence (Optional(bool)): If True, fields without
                presence (implicit presence scalars, repeated fields, and map fields) will
                always be serialized. Any field that supports presence is not affected by
                this option (including singular message fields and oneof fields).
                This value must match `including_default_value_fields`,
                if both arguments are explicitly set.
        Returns:
            str: The json string representation of the protocol buffer.
        """

        print_fields = cls._normalize_print_fields_without_presence(
            always_print_fields_with_no_presence, including_default_value_fields
        )

        if PROTOBUF_VERSION[0] in ("3", "4"):
            return MessageToJson(
                cls.pb(instance),
                use_integers_for_enums=use_integers_for_enums,
                including_default_value_fields=print_fields,
                preserving_proto_field_name=preserving_proto_field_name,
                sort_keys=sort_keys,
                indent=indent,
                float_precision=float_precision,
            )
        else:
            # The `including_default_value_fields` argument was removed from protobuf 5.x
            # and replaced with `always_print_fields_with_no_presence` which very similar but has
            # handles optional fields consistently by not affecting them.
            # The old flag accidentally had inconsistent behavior between proto2
            # optional and proto3 optional fields.
            return MessageToJson(
                cls.pb(instance),
                use_integers_for_enums=use_integers_for_enums,
                always_print_fields_with_no_presence=print_fields,
                preserving_proto_field_name=preserving_proto_field_name,
                sort_keys=sort_keys,
                indent=indent,
                float_precision=float_precision,
            )

    def from_json(cls, payload, *, ignore_unknown_fields=False) -> "Message":
        """Given a json string representing an instance,
        parse it into a message.

        Args:
            payload: A json string representing a message.
            ignore_unknown_fields (Optional(bool)): If True, do not raise errors
                for unknown fields.

        Returns:
            ~.Message: An instance of the message class against which this
            method was called.
        """
        instance = cls()
        Parse(payload, instance._pb, ignore_unknown_fields=ignore_unknown_fields)
        return instance

    def to_dict(
        cls,
        instance,
        *,
        use_integers_for_enums=True,
        preserving_proto_field_name=True,
        including_default_value_fields=None,
        float_precision=None,
        always_print_fields_with_no_presence=None,
    ) -> Dict[str, Any]:
        """Given a message instance, return its representation as a python dict.

        Args:
            instance: An instance of this message type, or something
                compatible (accepted by the type's constructor).
            use_integers_for_enums (Optional(bool)): An option that determines whether enum
                values should be represented by strings (False) or integers (True).
                Default is True.
            preserving_proto_field_name (Optional(bool)): An option that
                determines whether field name representations preserve
                proto case (snake_case) or use lowerCamelCase. Default is True.
            including_default_value_fields (Optional(bool)): Deprecated. Use argument
                `always_print_fields_with_no_presence` instead. An option that
                determines whether the default field values should be included in the results.
                This value must match `always_print_fields_with_no_presence`,
                if both arguments are explicitly set.
            float_precision (Optional(int)): If set, use this to specify float field valid digits.
                Default is None.
            always_print_fields_with_no_presence (Optional(bool)): If True, fields without
                presence (implicit presence scalars, repeated fields, and map fields) will
                always be serialized. Any field that supports presence is not affected by
                this option (including singular message fields and oneof fields). This value
                must match `including_default_value_fields`, if both arguments are explicitly set.

        Returns:
            dict: A representation of the protocol buffer using pythonic data structures.
                  Messages and map fields are represented as dicts,
                  repeated fields are represented as lists.
        """

        print_fields = cls._normalize_print_fields_without_presence(
            always_print_fields_with_no_presence, including_default_value_fields
        )

        if PROTOBUF_VERSION[0] in ("3", "4"):
            return MessageToDict(
                cls.pb(instance),
                including_default_value_fields=print_fields,
                preserving_proto_field_name=preserving_proto_field_name,
                use_integers_for_enums=use_integers_for_enums,
                float_precision=float_precision,
            )
        else:
            # The `including_default_value_fields` argument was removed from protobuf 5.x
            # and replaced with `always_print_fields_with_no_presence` which very similar but has
            # handles optional fields consistently by not affecting them.
            # The old flag accidentally had inconsistent behavior between proto2
            # optional and proto3 optional fields.
            return MessageToDict(
                cls.pb(instance),
                always_print_fields_with_no_presence=print_fields,
                preserving_proto_field_name=preserving_proto_field_name,
                use_integers_for_enums=use_integers_for_enums,
                float_precision=float_precision,
            )

    def copy_from(cls, instance, other):
        """Equivalent for protobuf.Message.CopyFrom

        Args:
            instance: An instance of this message type
            other: (Union[dict, ~.Message):
                A dictionary or message to reinitialize the values for this message.
        """
        if isinstance(other, cls):
            # Just want the underlying proto.
            other = Message.pb(other)
        elif isinstance(other, cls.pb()):
            # Don't need to do anything.
            pass
        elif isinstance(other, collections.abc.Mapping):
            # Coerce into a proto
            other = cls._meta.pb(**other)
        else:
            raise TypeError(
                "invalid argument type to copy to {}: {}".format(
                    cls.__name__, other.__class__.__name__
                )
            )

        # Note: we can't just run self.__init__ because this may be a message field
        # for a higher order proto; the memory layout for protos is NOT LIKE the
        # python memory model. We cannot rely on just setting things by reference.
        # Non-trivial complexity is (partially) hidden by the protobuf runtime.
        cls.pb(instance).CopyFrom(other)


class Message(metaclass=MessageMeta):
    """The abstract base class for a message.

    Args:
        mapping (Union[dict, ~.Message]): A dictionary or message to be
            used to determine the values for this message.
        ignore_unknown_fields (Optional(bool)): If True, do not raise errors for
            unknown fields. Only applied if `mapping` is a mapping type or there
            are keyword parameters.
        kwargs (dict): Keys and values corresponding to the fields of the
            message.
    """

    def __init__(
        self,
        mapping=None,
        *,
        ignore_unknown_fields=False,
        **kwargs,
    ):
        # We accept several things for `mapping`:
        #   * An instance of this class.
        #   * An instance of the underlying protobuf descriptor class.
        #   * A dict
        #   * Nothing (keyword arguments only).
        if mapping is None:
            if not kwargs:
                # Special fast path for empty construction.
                super().__setattr__("_pb", self._meta.pb())
                return

            mapping = kwargs
        elif isinstance(mapping, self._meta.pb):
            # Make a copy of the mapping.
            # This is a constructor for a new object, so users will assume
            # that it will not have side effects on the arguments being
            # passed in.
            #
            # The `wrap` method on the metaclass is the public API for taking
            # ownership of the passed in protobuf object.
            mapping = copy.deepcopy(mapping)
            if kwargs:
                mapping.MergeFrom(self._meta.pb(**kwargs))

            super().__setattr__("_pb", mapping)
            return
        elif isinstance(mapping, type(self)):
            # Just use the above logic on mapping's underlying pb.
            self.__init__(mapping=mapping._pb, **kwargs)
            return
        elif isinstance(mapping, collections.abc.Mapping):
            # Can't have side effects on mapping.
            mapping = copy.copy(mapping)
            # kwargs entries take priority for duplicate keys.
            mapping.update(kwargs)
        else:
            # Sanity check: Did we get something not a map? Error if so.
            raise TypeError(
                "Invalid constructor input for %s: %r"
                % (
                    self.__class__.__name__,
                    mapping,
                )
            )

        params = {}
        # Update the mapping to address any values that need to be
        # coerced.
        marshal = self._meta.marshal
        for key, value in mapping.items():
            (key, pb_type) = self._get_pb_type_from_key(key)
            if pb_type is None:
                if ignore_unknown_fields:
                    continue

                raise ValueError(
                    "Unknown field for {}: {}".format(self.__class__.__name__, key)
                )

            pb_value = marshal.to_proto(pb_type, value)

            if pb_value is not None:
                params[key] = pb_value

        # Create the internal protocol buffer.
        super().__setattr__("_pb", self._meta.pb(**params))

    def _get_pb_type_from_key(self, key):
        """Given a key, return the corresponding pb_type.

        Args:
            key(str): The name of the field.

        Returns:
            A tuple containing a key and pb_type. The pb_type will be
            the composite type of the field, or the primitive type if a primitive.
            If no corresponding field exists, return None.
        """

        pb_type = None

        try:
            pb_type = self._meta.fields[key].pb_type
        except KeyError:
            # Underscores may be appended to field names
            # that collide with python or proto-plus keywords.
            # In case a key only exists with a `_` suffix, coerce the key
            # to include the `_` suffix. It's not possible to
            # natively define the same field with a trailing underscore in protobuf.
            # See related issue
            # https://github.com/googleapis/python-api-core/issues/227
            if f"{key}_" in self._meta.fields:
                key = f"{key}_"
                pb_type = self._meta.fields[key].pb_type

        return (key, pb_type)

    def __dir__(self):
        desc = type(self).pb().DESCRIPTOR
        names = {f_name for f_name in self._meta.fields.keys()}
        names.update(m.name for m in desc.nested_types)
        names.update(e.name for e in desc.enum_types)
        names.update(dir(object()))
        # Can't think of a better way of determining
        # the special methods than manually listing them.
        names.update(
            (
                "__bool__",
                "__contains__",
                "__dict__",
                "__getattr__",
                "__getstate__",
                "__module__",
                "__setstate__",
                "__weakref__",
            )
        )

        return names

    def __bool__(self):
        """Return True if any field is truthy, False otherwise."""
        return any(k in self and getattr(self, k) for k in self._meta.fields.keys())

    def __contains__(self, key):
        """Return True if this field was set to something non-zero on the wire.

        In most cases, this method will return True when ``__getattr__``
        would return a truthy value and False when it would return a falsy
        value, so explicitly calling this is not useful.

        The exception case is empty messages explicitly set on the wire,
        which are falsy from ``__getattr__``. This method allows to
        distinguish between an explicitly provided empty message and the
        absence of that message, which is useful in some edge cases.

        The most common edge case is the use of ``google.protobuf.BoolValue``
        to get a boolean that distinguishes between ``False`` and ``None``
        (or the same for a string, int, etc.). This library transparently
        handles that case for you, but this method remains available to
        accommodate cases not automatically covered.

        Args:
            key (str): The name of the field.

        Returns:
            bool: Whether the field's value corresponds to a non-empty
                wire serialization.
        """
        pb_value = getattr(self._pb, key)
        try:
            # Protocol buffers "HasField" is unfriendly; it only works
            # against composite, non-repeated fields, and raises ValueError
            # against any repeated field or primitive.
            #
            # There is no good way to test whether it is valid to provide
            # a field to this method, so sadly we are stuck with a
            # somewhat inefficient try/except.
            return self._pb.HasField(key)
        except ValueError:
            return bool(pb_value)

    def __delattr__(self, key):
        """Delete the value on the given field.

        This is generally equivalent to setting a falsy value.
        """
        self._pb.ClearField(key)

    def __eq__(self, other):
        """Return True if the messages are equal, False otherwise."""
        # If these are the same type, use internal protobuf's equality check.
        if isinstance(other, type(self)):
            return self._pb == other._pb

        # If the other type is the target protobuf object, honor that also.
        if isinstance(other, self._meta.pb):
            return self._pb == other

        # Ask the other object.
        return NotImplemented

    def __getattr__(self, key):
        """Retrieve the given field's value.

        In protocol buffers, the presence of a field on a message is
        sufficient for it to always be "present".

        For primitives, a value of the correct type will always be returned
        (the "falsy" values in protocol buffers consistently match those
        in Python). For repeated fields, the falsy value is always an empty
        sequence.

        For messages, protocol buffers does distinguish between an empty
        message and absence, but this distinction is subtle and rarely
        relevant. Therefore, this method always returns an empty message
        (following the official implementation). To check for message
        presence, use ``key in self`` (in other words, ``__contains__``).

        .. note::

            Some well-known protocol buffer types
            (e.g. ``google.protobuf.Timestamp``) will be converted to
            their Python equivalents. See the ``marshal`` module for
            more details.
        """
        (key, pb_type) = self._get_pb_type_from_key(key)
        if pb_type is None:
            raise AttributeError(
                "Unknown field for {}: {}".format(self.__class__.__name__, key)
            )
        pb_value = getattr(self._pb, key)
        marshal = self._meta.marshal
        return marshal.to_python(pb_type, pb_value, absent=key not in self)

    def __ne__(self, other):
        """Return True if the messages are unequal, False otherwise."""
        return not self == other

    def __repr__(self):
        return repr(self._pb)

    def __setattr__(self, key, value):
        """Set the value on the given field.

        For well-known protocol buffer types which are marshalled, either
        the protocol buffer object or the Python equivalent is accepted.
        """
        if key[0] == "_":
            return super().__setattr__(key, value)
        marshal = self._meta.marshal
        (key, pb_type) = self._get_pb_type_from_key(key)
        if pb_type is None:
            raise AttributeError(
                "Unknown field for {}: {}".format(self.__class__.__name__, key)
            )

        pb_value = marshal.to_proto(pb_type, value)

        # Clear the existing field.
        # This is the only way to successfully write nested falsy values,
        # because otherwise MergeFrom will no-op on them.
        self._pb.ClearField(key)

        # Merge in the value being set.
        if pb_value is not None:
            self._pb.MergeFrom(self._meta.pb(**{key: pb_value}))

    def __getstate__(self):
        """Serialize for pickling."""
        return self._pb.SerializeToString()

    def __setstate__(self, value):
        """Deserialization for pickling."""
        new_pb = self._meta.pb().FromString(value)
        super().__setattr__("_pb", new_pb)


class _MessageInfo:
    """Metadata about a message.

    Args:
        fields (Tuple[~.fields.Field]): The fields declared on the message.
        package (str): The proto package.
        full_name (str): The full name of the message.
        file_info (~._FileInfo): The file descriptor and messages for the
            file containing this message.
        marshal (~.Marshal): The marshal instance to which this message was
            automatically registered.
        options (~.descriptor_pb2.MessageOptions): Any options that were
            set on the message.
    """

    def __init__(
        self,
        *,
        fields: List[Field],
        package: str,
        full_name: str,
        marshal: Marshal,
        options: descriptor_pb2.MessageOptions,
    ) -> None:
        self.package = package
        self.full_name = full_name
        self.options = options
        self.fields = collections.OrderedDict((i.name, i) for i in fields)
        self.fields_by_number = collections.OrderedDict((i.number, i) for i in fields)
        self.marshal = marshal
        self._pb = None

    @property
    def pb(self) -> Type[message.Message]:
        """Return the protobuf message type for this descriptor.

        If a field on the message references another message which has not
        loaded, then this method returns None.
        """
        return self._pb


__all__ = ("Message",)
