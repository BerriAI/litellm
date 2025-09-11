# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# pylint: disable=docstring-missing-return, docstring-missing-rtype

"""Input/output utilities.

Includes:
 - i/o-specific constants
 - i/o-specific exceptions
 - schema validation
 - leaf value encoding and decoding
 - datum reader/writer stuff (?)

Also includes a generic representation for data, which uses the
following mapping:
 - Schema records are implemented as dict.
 - Schema arrays are implemented as list.
 - Schema maps are implemented as dict.
 - Schema strings are implemented as unicode.
 - Schema bytes are implemented as str.
 - Schema ints are implemented as int.
 - Schema longs are implemented as long.
 - Schema floats are implemented as float.
 - Schema doubles are implemented as float.
 - Schema booleans are implemented as bool.
"""

import json
import logging
import struct
import sys

from ..avro import schema

PY3 = sys.version_info[0] == 3

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Constants

STRUCT_FLOAT = struct.Struct("<f")  # little-endian float
STRUCT_DOUBLE = struct.Struct("<d")  # little-endian double

# ------------------------------------------------------------------------------
# Exceptions


class SchemaResolutionException(schema.AvroException):
    def __init__(self, fail_msg, writer_schema=None):
        pretty_writers = json.dumps(json.loads(str(writer_schema)), indent=2)
        if writer_schema:
            fail_msg += f"\nWriter's Schema: {pretty_writers}"
        schema.AvroException.__init__(self, fail_msg)


# ------------------------------------------------------------------------------
# Decoder


class BinaryDecoder(object):
    """Read leaf values."""

    def __init__(self, reader):
        """
        reader is a Python object on which we can call read, seek, and tell.
        """
        self._reader = reader

    @property
    def reader(self):
        """Reports the reader used by this decoder."""
        return self._reader

    def read(self, n):
        """Read n bytes.

        :param int n: Number of bytes to read.
        :return: The next n bytes from the input.
        :rtype: bytes
        """
        assert n >= 0, n
        input_bytes = self.reader.read(n)
        if n > 0 and not input_bytes:
            raise StopIteration
        assert len(input_bytes) == n, input_bytes
        return input_bytes

    @staticmethod
    def read_null():
        """
        null is written as zero bytes
        """
        return None

    def read_boolean(self):
        """
        a boolean is written as a single byte
        whose value is either 0 (false) or 1 (true).
        """
        b = ord(self.read(1))
        if b == 1:
            return True
        if b == 0:
            return False
        fail_msg = f"Invalid value for boolean: {b}"
        raise schema.AvroException(fail_msg)

    def read_int(self):
        """
        int and long values are written using variable-length, zig-zag coding.
        """
        return self.read_long()

    def read_long(self):
        """
        int and long values are written using variable-length, zig-zag coding.
        """
        b = ord(self.read(1))
        n = b & 0x7F
        shift = 7
        while (b & 0x80) != 0:
            b = ord(self.read(1))
            n |= (b & 0x7F) << shift
            shift += 7
        datum = (n >> 1) ^ -(n & 1)
        return datum

    def read_float(self):
        """
        A float is written as 4 bytes.
        The float is converted into a 32-bit integer using a method equivalent to
        Java's floatToIntBits and then encoded in little-endian format.
        """
        return STRUCT_FLOAT.unpack(self.read(4))[0]

    def read_double(self):
        """
        A double is written as 8 bytes.
        The double is converted into a 64-bit integer using a method equivalent to
        Java's doubleToLongBits and then encoded in little-endian format.
        """
        return STRUCT_DOUBLE.unpack(self.read(8))[0]

    def read_bytes(self):
        """
        Bytes are encoded as a long followed by that many bytes of data.
        """
        nbytes = self.read_long()
        assert nbytes >= 0, nbytes
        return self.read(nbytes)

    def read_utf8(self):
        """
        A string is encoded as a long followed by
        that many bytes of UTF-8 encoded character data.
        """
        input_bytes = self.read_bytes()
        if PY3:
            try:
                return input_bytes.decode("utf-8")
            except UnicodeDecodeError as exn:
                logger.error(
                    "Invalid UTF-8 input bytes: %r", input_bytes
                )  # pylint: disable=do-not-log-raised-errors
                raise exn
        else:
            # PY2
            return unicode(input_bytes, "utf-8")  # pylint: disable=undefined-variable

    def skip_null(self):
        pass

    def skip_boolean(self):
        self.skip(1)

    def skip_int(self):
        self.skip_long()

    def skip_long(self):
        b = ord(self.read(1))
        while (b & 0x80) != 0:
            b = ord(self.read(1))

    def skip_float(self):
        self.skip(4)

    def skip_double(self):
        self.skip(8)

    def skip_bytes(self):
        self.skip(self.read_long())

    def skip_utf8(self):
        self.skip_bytes()

    def skip(self, n):
        self.reader.seek(self.reader.tell() + n)


# ------------------------------------------------------------------------------
# DatumReader


class DatumReader(object):
    """Deserialize Avro-encoded data into a Python data structure."""

    def __init__(self, writer_schema=None):
        """
        As defined in the Avro specification, we call the schema encoded
        in the data the "writer's schema".
        """
        self._writer_schema = writer_schema

    # read/write properties
    def set_writer_schema(self, writer_schema):
        self._writer_schema = writer_schema

    writer_schema = property(lambda self: self._writer_schema, set_writer_schema)

    def read(self, decoder):
        return self.read_data(self.writer_schema, decoder)

    def read_data(self, writer_schema, decoder):
        # function dispatch for reading data based on type of writer's schema
        if writer_schema.type == "null":
            result = decoder.read_null()
        elif writer_schema.type == "boolean":
            result = decoder.read_boolean()
        elif writer_schema.type == "string":
            result = decoder.read_utf8()
        elif writer_schema.type == "int":
            result = decoder.read_int()
        elif writer_schema.type == "long":
            result = decoder.read_long()
        elif writer_schema.type == "float":
            result = decoder.read_float()
        elif writer_schema.type == "double":
            result = decoder.read_double()
        elif writer_schema.type == "bytes":
            result = decoder.read_bytes()
        elif writer_schema.type == "fixed":
            result = self.read_fixed(writer_schema, decoder)
        elif writer_schema.type == "enum":
            result = self.read_enum(writer_schema, decoder)
        elif writer_schema.type == "array":
            result = self.read_array(writer_schema, decoder)
        elif writer_schema.type == "map":
            result = self.read_map(writer_schema, decoder)
        elif writer_schema.type in ["union", "error_union"]:
            result = self.read_union(writer_schema, decoder)
        elif writer_schema.type in ["record", "error", "request"]:
            result = self.read_record(writer_schema, decoder)
        else:
            fail_msg = f"Cannot read unknown schema type: {writer_schema.type}"
            raise schema.AvroException(fail_msg)
        return result

    def skip_data(self, writer_schema, decoder):
        if writer_schema.type == "null":
            result = decoder.skip_null()
        elif writer_schema.type == "boolean":
            result = decoder.skip_boolean()
        elif writer_schema.type == "string":
            result = decoder.skip_utf8()
        elif writer_schema.type == "int":
            result = decoder.skip_int()
        elif writer_schema.type == "long":
            result = decoder.skip_long()
        elif writer_schema.type == "float":
            result = decoder.skip_float()
        elif writer_schema.type == "double":
            result = decoder.skip_double()
        elif writer_schema.type == "bytes":
            result = decoder.skip_bytes()
        elif writer_schema.type == "fixed":
            result = self.skip_fixed(writer_schema, decoder)
        elif writer_schema.type == "enum":
            result = self.skip_enum(decoder)
        elif writer_schema.type == "array":
            self.skip_array(writer_schema, decoder)
            result = None
        elif writer_schema.type == "map":
            self.skip_map(writer_schema, decoder)
            result = None
        elif writer_schema.type in ["union", "error_union"]:
            result = self.skip_union(writer_schema, decoder)
        elif writer_schema.type in ["record", "error", "request"]:
            self.skip_record(writer_schema, decoder)
            result = None
        else:
            fail_msg = f"Unknown schema type: {writer_schema.type}"
            raise schema.AvroException(fail_msg)
        return result

    # Fixed instances are encoded using the number of bytes declared in the schema.
    @staticmethod
    def read_fixed(writer_schema, decoder):
        return decoder.read(writer_schema.size)

    @staticmethod
    def skip_fixed(writer_schema, decoder):
        return decoder.skip(writer_schema.size)

    # An enum is encoded by a int, representing the zero-based position of the symbol in the schema.
    @staticmethod
    def read_enum(writer_schema, decoder):
        # read data
        index_of_symbol = decoder.read_int()
        if index_of_symbol >= len(writer_schema.symbols):
            fail_msg = f"Can't access enum index {index_of_symbol} for enum with {len(writer_schema.symbols)} symbols"
            raise SchemaResolutionException(fail_msg, writer_schema)
        read_symbol = writer_schema.symbols[index_of_symbol]
        return read_symbol

    @staticmethod
    def skip_enum(decoder):
        return decoder.skip_int()

    # Arrays are encoded as a series of blocks.

    # Each block consists of a long count value, followed by that many array items.
    # A block with count zero indicates the end of the array. Each item is encoded per the array's item schema.

    # If a block's count is negative, then the count is followed immediately by a long block size,
    # indicating the number of bytes in the block.
    # The actual count in this case is the absolute value of the count written.
    def read_array(self, writer_schema, decoder):
        read_items = []
        block_count = decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_count = -block_count
                decoder.read_long()
            for _ in range(block_count):
                read_items.append(self.read_data(writer_schema.items, decoder))
            block_count = decoder.read_long()
        return read_items

    def skip_array(self, writer_schema, decoder):
        block_count = decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_size = decoder.read_long()
                decoder.skip(block_size)
            else:
                for _ in range(block_count):
                    self.skip_data(writer_schema.items, decoder)
            block_count = decoder.read_long()

    # Maps are encoded as a series of blocks.

    # Each block consists of a long count value, followed by that many key/value pairs.
    # A block with count zero indicates the end of the map. Each item is encoded per the map's value schema.

    # If a block's count is negative, then the count is followed immediately by a long block size,
    # indicating the number of bytes in the block.
    # The actual count in this case is the absolute value of the count written.
    def read_map(self, writer_schema, decoder):
        read_items = {}
        block_count = decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_count = -block_count
                decoder.read_long()
            for _ in range(block_count):
                key = decoder.read_utf8()
                read_items[key] = self.read_data(writer_schema.values, decoder)
            block_count = decoder.read_long()
        return read_items

    def skip_map(self, writer_schema, decoder):
        block_count = decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_size = decoder.read_long()
                decoder.skip(block_size)
            else:
                for _ in range(block_count):
                    decoder.skip_utf8()
                    self.skip_data(writer_schema.values, decoder)
            block_count = decoder.read_long()

    # A union is encoded by first writing a long value indicating
    # the zero-based position within the union of the schema of its value.
    # The value is then encoded per the indicated schema within the union.
    def read_union(self, writer_schema, decoder):
        # schema resolution
        index_of_schema = int(decoder.read_long())
        if index_of_schema >= len(writer_schema.schemas):
            fail_msg = (
                f"Can't access branch index {index_of_schema} "
                f"for union with {len(writer_schema.schemas)} branches"
            )
            raise SchemaResolutionException(fail_msg, writer_schema)
        selected_writer_schema = writer_schema.schemas[index_of_schema]

        # read data
        return self.read_data(selected_writer_schema, decoder)

    def skip_union(self, writer_schema, decoder):
        index_of_schema = int(decoder.read_long())
        if index_of_schema >= len(writer_schema.schemas):
            fail_msg = (
                f"Can't access branch index {index_of_schema} "
                f"for union with {len(writer_schema.schemas)} branches"
            )
            raise SchemaResolutionException(fail_msg, writer_schema)
        return self.skip_data(writer_schema.schemas[index_of_schema], decoder)

    # A record is encoded by encoding the values of its fields
    # in the order that they are declared. In other words, a record
    # is encoded as just the concatenation of the encodings of its fields.
    # Field values are encoded per their schema.

    # Schema Resolution:
    #     * the ordering of fields may be different: fields are matched by name.
    #     * schemas for fields with the same name in both records are resolved
    #     recursively.
    #     * if the writer's record contains a field with a name not present in the
    #     reader's record, the writer's value for that field is ignored.
    #     * if the reader's record schema has a field that contains a default value,
    #     and writer's schema does not have a field with the same name, then the
    #     reader should use the default value from its field.
    #     * if the reader's record schema has a field with no default value, and
    #     writer's schema does not have a field with the same name, then the
    #     field's value is unset.
    def read_record(self, writer_schema, decoder):
        # schema resolution
        read_record = {}
        for field in writer_schema.fields:
            field_val = self.read_data(field.type, decoder)
            read_record[field.name] = field_val
        return read_record

    def skip_record(self, writer_schema, decoder):
        for field in writer_schema.fields:
            self.skip_data(field.type, decoder)
