# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# pylint: disable=docstring-missing-return, docstring-missing-rtype

"""Read/Write Avro File Object Containers."""

import io
import logging
import sys
import zlib

from ..avro import avro_io
from ..avro import schema

PY3 = sys.version_info[0] == 3

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Constants

# Version of the container file:
VERSION = 1

if PY3:
    MAGIC = b"Obj" + bytes([VERSION])
    MAGIC_SIZE = len(MAGIC)
else:
    MAGIC = "Obj" + chr(VERSION)
    MAGIC_SIZE = len(MAGIC)

# Size of the synchronization marker, in number of bytes:
SYNC_SIZE = 16

# Schema of the container header:
META_SCHEMA = schema.parse(
    """
{
  "type": "record", "name": "org.apache.avro.file.Header",
  "fields": [{
    "name": "magic",
    "type": {"type": "fixed", "name": "magic", "size": %(magic_size)d}
  }, {
    "name": "meta",
    "type": {"type": "map", "values": "bytes"}
  }, {
    "name": "sync",
    "type": {"type": "fixed", "name": "sync", "size": %(sync_size)d}
  }]
}
"""
    % {
        "magic_size": MAGIC_SIZE,
        "sync_size": SYNC_SIZE,
    }
)

# Codecs supported by container files:
VALID_CODECS = frozenset(["null", "deflate"])

# Metadata key associated to the schema:
SCHEMA_KEY = "avro.schema"


# ------------------------------------------------------------------------------
# Exceptions


class DataFileException(schema.AvroException):
    """Problem reading or writing file object containers."""


# ------------------------------------------------------------------------------


class DataFileReader(object):  # pylint: disable=too-many-instance-attributes
    """Read files written by DataFileWriter."""

    def __init__(self, reader, datum_reader, **kwargs):
        """Initializes a new data file reader.

        Args:
          reader: Open file to read from.
          datum_reader: Avro datum reader.
        """
        self._reader = reader
        self._raw_decoder = avro_io.BinaryDecoder(reader)
        self._header_reader = kwargs.pop("header_reader", None)
        self._header_decoder = (
            None
            if self._header_reader is None
            else avro_io.BinaryDecoder(self._header_reader)
        )
        self._datum_decoder = None  # Maybe reset at every block.
        self._datum_reader = datum_reader

        # In case self._reader only has partial content(without header).
        # seek(0, 0) to make sure read the (partial)content from beginning.
        self._reader.seek(0, 0)

        # read the header: magic, meta, sync
        self._read_header()

        # ensure codec is valid
        avro_codec_raw = self.get_meta("avro.codec")
        if avro_codec_raw is None:
            self.codec = "null"
        else:
            self.codec = avro_codec_raw.decode("utf-8")
        if self.codec not in VALID_CODECS:
            raise DataFileException(f"Unknown codec: {self.codec}.")

        # get ready to read
        self._block_count = 0

        # object_position is to support reading from current position in the future read,
        # no need to downloading from the beginning of avro.
        if hasattr(self._reader, "object_position"):
            self.reader.track_object_position()

        self._cur_object_index = 0
        # header_reader indicates reader only has partial content. The reader doesn't have block header,
        # so we read use the block count stored last time.
        # Also ChangeFeed only has codec==null, so use _raw_decoder is good.
        if self._header_reader is not None:
            self._datum_decoder = self._raw_decoder

        self.datum_reader.writer_schema = schema.parse(
            self.get_meta(SCHEMA_KEY).decode("utf-8")
        )

    def __enter__(self):
        return self

    def __exit__(self, data_type, value, traceback):
        # Perform a close if there's no exception
        if data_type is None:
            self.close()

    def __iter__(self):
        return self

    # read-only properties
    @property
    def reader(self):
        return self._reader

    @property
    def raw_decoder(self):
        return self._raw_decoder

    @property
    def datum_decoder(self):
        return self._datum_decoder

    @property
    def datum_reader(self):
        return self._datum_reader

    @property
    def sync_marker(self):
        return self._sync_marker

    @property
    def meta(self):
        return self._meta

    # read/write properties
    @property
    def block_count(self):
        return self._block_count

    def get_meta(self, key):
        """Reports the value of a given metadata key.

        :param str key: Metadata key to report the value of.
        :return: Value associated to the metadata key, as bytes.
        :rtype: bytes
        """
        return self._meta.get(key)

    def _read_header(self):
        header_reader = self._header_reader if self._header_reader else self._reader
        header_decoder = (
            self._header_decoder if self._header_decoder else self._raw_decoder
        )

        # seek to the beginning of the file to get magic block
        header_reader.seek(0, 0)

        # read header into a dict
        header = self.datum_reader.read_data(META_SCHEMA, header_decoder)

        # check magic number
        if header.get("magic") != MAGIC:
            fail_msg = (
                f"Not an Avro data file: {header.get('magic')} doesn't match {MAGIC!r}."
            )
            raise schema.AvroException(fail_msg)

        # set metadata
        self._meta = header["meta"]

        # set sync marker
        self._sync_marker = header["sync"]

    def _read_block_header(self):
        self._block_count = self.raw_decoder.read_long()
        if self.codec == "null":
            # Skip a long; we don't need to use the length.
            self.raw_decoder.skip_long()
            self._datum_decoder = self._raw_decoder
        elif self.codec == "deflate":
            # Compressed data is stored as (length, data), which
            # corresponds to how the "bytes" type is encoded.
            data = self.raw_decoder.read_bytes()
            # -15 is the log of the window size; negative indicates
            # "raw" (no zlib headers) decompression.  See zlib.h.
            uncompressed = zlib.decompress(data, -15)
            self._datum_decoder = avro_io.BinaryDecoder(io.BytesIO(uncompressed))
        else:
            raise DataFileException(f"Unknown codec: {self.codec!r}")

    def _skip_sync(self):
        """
        Read the length of the sync marker; if it matches the sync marker,
        return True. Otherwise, seek back to where we started and return False.
        """
        proposed_sync_marker = self.reader.read(SYNC_SIZE)
        if SYNC_SIZE > 0 and not proposed_sync_marker:
            raise StopIteration
        if proposed_sync_marker != self.sync_marker:
            self.reader.seek(-SYNC_SIZE, 1)

    def __next__(self):
        """Return the next datum in the file."""
        if self.block_count == 0:
            self._skip_sync()

            # object_position is to support reading from current position in the future read,
            # no need to downloading from the beginning of avro file with this attr.
            if hasattr(self._reader, "object_position"):
                self.reader.track_object_position()
            self._cur_object_index = 0

            self._read_block_header()

        datum = self.datum_reader.read(self.datum_decoder)
        self._block_count -= 1
        self._cur_object_index += 1

        # object_position is to support reading from current position in the future read,
        # This will track the index of the next item to be read.
        # This will also track the offset before the next sync marker.
        if hasattr(self._reader, "object_position"):
            if self.block_count == 0:
                # the next event to be read is at index 0 in the new chunk of blocks,
                self.reader.track_object_position()
                self.reader.set_object_index(0)
            else:
                self.reader.set_object_index(self._cur_object_index)

        return datum

    def close(self):
        """Close this reader."""
        self.reader.close()
