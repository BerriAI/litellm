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

import logging
import sys

from ..avro import schema

from .avro_io import STRUCT_FLOAT, STRUCT_DOUBLE, SchemaResolutionException

PY3 = sys.version_info[0] == 3

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Decoder


class AsyncBinaryDecoder(object):
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

    async def read(self, n):
        """Read n bytes.

        :param int n: Number of bytes to read.
        :return: The next n bytes from the input.
        :rtype: bytes
        """
        assert n >= 0, n
        input_bytes = await self.reader.read(n)
        if n > 0 and not input_bytes:
            raise StopAsyncIteration
        assert len(input_bytes) == n, input_bytes
        return input_bytes

    @staticmethod
    def read_null():
        """
        null is written as zero bytes
        """
        return None

    async def read_boolean(self):
        """
        a boolean is written as a single byte
        whose value is either 0 (false) or 1 (true).
        """
        b = ord(await self.read(1))
        if b == 1:
            return True
        if b == 0:
            return False
        fail_msg = f"Invalid value for boolean: {b}"
        raise schema.AvroException(fail_msg)

    async def read_int(self):
        """
        int and long values are written using variable-length, zig-zag coding.
        """
        return await self.read_long()

    async def read_long(self):
        """
        int and long values are written using variable-length, zig-zag coding.
        """
        b = ord(await self.read(1))
        n = b & 0x7F
        shift = 7
        while (b & 0x80) != 0:
            b = ord(await self.read(1))
            n |= (b & 0x7F) << shift
            shift += 7
        datum = (n >> 1) ^ -(n & 1)
        return datum

    async def read_float(self):
        """
        A float is written as 4 bytes.
        The float is converted into a 32-bit integer using a method equivalent to
        Java's floatToIntBits and then encoded in little-endian format.
        """
        return STRUCT_FLOAT.unpack(await self.read(4))[0]

    async def read_double(self):
        """
        A double is written as 8 bytes.
        The double is converted into a 64-bit integer using a method equivalent to
        Java's doubleToLongBits and then encoded in little-endian format.
        """
        return STRUCT_DOUBLE.unpack(await self.read(8))[0]

    async def read_bytes(self):
        """
        Bytes are encoded as a long followed by that many bytes of data.
        """
        nbytes = await self.read_long()
        assert nbytes >= 0, nbytes
        return await self.read(nbytes)

    async def read_utf8(self):
        """
        A string is encoded as a long followed by
        that many bytes of UTF-8 encoded character data.
        """
        input_bytes = await self.read_bytes()
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

    async def skip_boolean(self):
        await self.skip(1)

    async def skip_int(self):
        await self.skip_long()

    async def skip_long(self):
        b = ord(await self.read(1))
        while (b & 0x80) != 0:
            b = ord(await self.read(1))

    async def skip_float(self):
        await self.skip(4)

    async def skip_double(self):
        await self.skip(8)

    async def skip_bytes(self):
        await self.skip(await self.read_long())

    async def skip_utf8(self):
        await self.skip_bytes()

    async def skip(self, n):
        await self.reader.seek(await self.reader.tell() + n)


# ------------------------------------------------------------------------------
# DatumReader


class AsyncDatumReader(object):
    """Deserialize Avro-encoded data into a Python data structure."""

    def __init__(self, writer_schema=None):
        """
        As defined in the Avro specification, we call the schema encoded
        in the data the "writer's schema", and the schema expected by the
        reader the "reader's schema".
        """
        self._writer_schema = writer_schema

    # read/write properties
    def set_writer_schema(self, writer_schema):
        self._writer_schema = writer_schema

    writer_schema = property(lambda self: self._writer_schema, set_writer_schema)

    async def read(self, decoder):
        return await self.read_data(self.writer_schema, decoder)

    async def read_data(self, writer_schema, decoder):
        # function dispatch for reading data based on type of writer's schema
        if writer_schema.type == "null":
            result = decoder.read_null()
        elif writer_schema.type == "boolean":
            result = await decoder.read_boolean()
        elif writer_schema.type == "string":
            result = await decoder.read_utf8()
        elif writer_schema.type == "int":
            result = await decoder.read_int()
        elif writer_schema.type == "long":
            result = await decoder.read_long()
        elif writer_schema.type == "float":
            result = await decoder.read_float()
        elif writer_schema.type == "double":
            result = await decoder.read_double()
        elif writer_schema.type == "bytes":
            result = await decoder.read_bytes()
        elif writer_schema.type == "fixed":
            result = await self.read_fixed(writer_schema, decoder)
        elif writer_schema.type == "enum":
            result = await self.read_enum(writer_schema, decoder)
        elif writer_schema.type == "array":
            result = await self.read_array(writer_schema, decoder)
        elif writer_schema.type == "map":
            result = await self.read_map(writer_schema, decoder)
        elif writer_schema.type in ["union", "error_union"]:
            result = await self.read_union(writer_schema, decoder)
        elif writer_schema.type in ["record", "error", "request"]:
            result = await self.read_record(writer_schema, decoder)
        else:
            fail_msg = f"Cannot read unknown schema type: {writer_schema.type}"
            raise schema.AvroException(fail_msg)
        return result

    async def skip_data(self, writer_schema, decoder):
        if writer_schema.type == "null":
            result = decoder.skip_null()
        elif writer_schema.type == "boolean":
            result = await decoder.skip_boolean()
        elif writer_schema.type == "string":
            result = await decoder.skip_utf8()
        elif writer_schema.type == "int":
            result = await decoder.skip_int()
        elif writer_schema.type == "long":
            result = await decoder.skip_long()
        elif writer_schema.type == "float":
            result = await decoder.skip_float()
        elif writer_schema.type == "double":
            result = await decoder.skip_double()
        elif writer_schema.type == "bytes":
            result = await decoder.skip_bytes()
        elif writer_schema.type == "fixed":
            result = await self.skip_fixed(writer_schema, decoder)
        elif writer_schema.type == "enum":
            result = await self.skip_enum(decoder)
        elif writer_schema.type == "array":
            await self.skip_array(writer_schema, decoder)
            result = None
        elif writer_schema.type == "map":
            await self.skip_map(writer_schema, decoder)
            result = None
        elif writer_schema.type in ["union", "error_union"]:
            result = await self.skip_union(writer_schema, decoder)
        elif writer_schema.type in ["record", "error", "request"]:
            await self.skip_record(writer_schema, decoder)
            result = None
        else:
            fail_msg = f"Unknown schema type: {writer_schema.type}"
            raise schema.AvroException(fail_msg)
        return result

    # Fixed instances are encoded using the number of bytes declared in the schema.
    @staticmethod
    async def read_fixed(writer_schema, decoder):
        return await decoder.read(writer_schema.size)

    @staticmethod
    async def skip_fixed(writer_schema, decoder):
        return await decoder.skip(writer_schema.size)

    # An enum is encoded by a int, representing the zero-based position of the symbol in the schema.
    @staticmethod
    async def read_enum(writer_schema, decoder):
        # read data
        index_of_symbol = await decoder.read_int()
        if index_of_symbol >= len(writer_schema.symbols):
            fail_msg = f"Can't access enum index {index_of_symbol} for enum with {len(writer_schema.symbols)} symbols"
            raise SchemaResolutionException(fail_msg, writer_schema)
        read_symbol = writer_schema.symbols[index_of_symbol]
        return read_symbol

    @staticmethod
    async def skip_enum(decoder):
        return await decoder.skip_int()

    # Arrays are encoded as a series of blocks.

    # Each block consists of a long count value, followed by that many array items.
    # A block with count zero indicates the end of the array. Each item is encoded per the array's item schema.

    # If a block's count is negative, then the count is followed immediately by a long block size,
    # indicating the number of bytes in the block.
    # The actual count in this case is the absolute value of the count written.
    async def read_array(self, writer_schema, decoder):
        read_items = []
        block_count = await decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_count = -block_count
                await decoder.read_long()
            for _ in range(block_count):
                read_items.append(await self.read_data(writer_schema.items, decoder))
            block_count = await decoder.read_long()
        return read_items

    async def skip_array(self, writer_schema, decoder):
        block_count = await decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_size = await decoder.read_long()
                await decoder.skip(block_size)
            else:
                for _ in range(block_count):
                    await self.skip_data(writer_schema.items, decoder)
            block_count = await decoder.read_long()

    # Maps are encoded as a series of blocks.

    # Each block consists of a long count value, followed by that many key/value pairs.
    # A block with count zero indicates the end of the map. Each item is encoded per the map's value schema.

    # If a block's count is negative, then the count is followed immediately by a long block size,
    # indicating the number of bytes in the block.
    # The actual count in this case is the absolute value of the count written.
    async def read_map(self, writer_schema, decoder):
        read_items = {}
        block_count = await decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_count = -block_count
                await decoder.read_long()
            for _ in range(block_count):
                key = await decoder.read_utf8()
                read_items[key] = await self.read_data(writer_schema.values, decoder)
            block_count = await decoder.read_long()
        return read_items

    async def skip_map(self, writer_schema, decoder):
        block_count = await decoder.read_long()
        while block_count != 0:
            if block_count < 0:
                block_size = await decoder.read_long()
                await decoder.skip(block_size)
            else:
                for _ in range(block_count):
                    await decoder.skip_utf8()
                    await self.skip_data(writer_schema.values, decoder)
            block_count = await decoder.read_long()

    # A union is encoded by first writing a long value indicating
    # the zero-based position within the union of the schema of its value.
    # The value is then encoded per the indicated schema within the union.
    async def read_union(self, writer_schema, decoder):
        # schema resolution
        index_of_schema = int(await decoder.read_long())
        if index_of_schema >= len(writer_schema.schemas):
            fail_msg = (
                f"Can't access branch index {index_of_schema} "
                f"for union with {len(writer_schema.schemas)} branches"
            )
            raise SchemaResolutionException(fail_msg, writer_schema)
        selected_writer_schema = writer_schema.schemas[index_of_schema]

        # read data
        return await self.read_data(selected_writer_schema, decoder)

    async def skip_union(self, writer_schema, decoder):
        index_of_schema = int(await decoder.read_long())
        if index_of_schema >= len(writer_schema.schemas):
            fail_msg = (
                f"Can't access branch index {index_of_schema} "
                f"for union with {len(writer_schema.schemas)} branches"
            )
            raise SchemaResolutionException(fail_msg, writer_schema)
        return await self.skip_data(writer_schema.schemas[index_of_schema], decoder)

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
    async def read_record(self, writer_schema, decoder):
        # schema resolution
        read_record = {}
        for field in writer_schema.fields:
            field_val = await self.read_data(field.type, decoder)
            read_record[field.name] = field_val
        return read_record

    async def skip_record(self, writer_schema, decoder):
        for field in writer_schema.fields:
            await self.skip_data(field.type, decoder)
