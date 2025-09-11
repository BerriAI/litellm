# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from io import BytesIO
from typing import Any, Dict, Generator, IO, Iterable, Optional, Type, TYPE_CHECKING

from ._shared.avro.avro_io import DatumReader
from ._shared.avro.datafile import DataFileReader

if TYPE_CHECKING:
    from ._models import BlobQueryError


class BlobQueryReader:  # pylint: disable=too-many-instance-attributes
    """A streaming object to read query results."""

    name: str
    """The name of the blob being queried."""
    container: str
    """The name of the container where the blob is."""
    response_headers: Dict[str, Any]
    """The response_headers of the quick query request."""
    record_delimiter: str
    """The delimiter used to separate lines, or records with the data. The `records`
    method will return these lines via a generator."""

    def __init__(
        self,
        name: str = None,  # type: ignore [assignment]
        container: str = None,  # type: ignore [assignment]
        errors: Any = None,
        record_delimiter: str = "\n",
        encoding: Optional[str] = None,
        headers: Dict[str, Any] = None,  # type: ignore [assignment]
        response: Any = None,
        error_cls: Type["BlobQueryError"] = None,  # type: ignore [assignment]
    ) -> None:
        self.name = name
        self.container = container
        self.response_headers = headers
        self.record_delimiter = record_delimiter
        self._size = 0
        self._bytes_processed = 0
        self._errors = errors
        self._encoding = encoding
        self._parsed_results = DataFileReader(
            QuickQueryStreamer(response), DatumReader()
        )
        self._first_result = self._process_record(next(self._parsed_results))
        self._error_cls = error_cls

    def __len__(self) -> int:
        return self._size

    def _process_record(self, result: Dict[str, Any]) -> Optional[bytes]:
        self._size = result.get("totalBytes", self._size)
        self._bytes_processed = result.get("bytesScanned", self._bytes_processed)
        if "data" in result:
            return result.get("data")
        if "fatal" in result:
            error = self._error_cls(
                error=result["name"],
                is_fatal=result["fatal"],
                description=result["description"],
                position=result["position"],
            )
            if self._errors:
                self._errors(error)
        return None

    def _iter_stream(self) -> Generator[bytes, None, None]:
        if self._first_result is not None:
            yield self._first_result
        for next_result in self._parsed_results:
            processed_result = self._process_record(next_result)
            if processed_result is not None:
                yield processed_result

    def readall(self) -> bytes:
        """Return all query results.

        This operation is blocking until all data is downloaded.

        :return: The query results.
        :rtype: bytes
        """
        stream = BytesIO()
        self.readinto(stream)
        data = stream.getvalue()
        if self._encoding:
            return data.decode(self._encoding)  # type: ignore [return-value]
        return data

    def readinto(self, stream: IO) -> None:
        """Download the query result to a stream.

        :param IO stream:
            The stream to download to. This can be an open file-handle,
            or any writable stream.
        :return: None
        """
        for record in self._iter_stream():
            stream.write(record)

    def records(self) -> Iterable[bytes]:
        """Returns a record generator for the query result.

        Records will be returned line by line.

        :return: A record generator for the query result.
        :rtype: Iterable[bytes]
        """
        delimiter = self.record_delimiter.encode("utf-8")
        for record_chunk in self._iter_stream():
            for record in record_chunk.split(delimiter):
                if self._encoding:
                    yield record.decode(self._encoding)  # type: ignore [misc]
                else:
                    yield record


class QuickQueryStreamer:
    """File-like streaming iterator."""

    def __init__(self, generator):
        self.generator = generator
        self.iterator = iter(generator)
        self._buf = b""
        self._point = 0
        self._download_offset = 0
        self._buf_start = 0
        self.file_length = None

    def __len__(self):
        return self.file_length

    def __iter__(self):
        return self.iterator

    @staticmethod
    def seekable():
        return True

    def __next__(self):
        next_part = next(self.iterator)
        self._download_offset += len(next_part)
        return next_part

    def tell(self):
        return self._point

    def seek(self, offset, whence=0):
        if whence == 0:
            self._point = offset
        elif whence == 1:
            self._point += offset
        else:
            raise ValueError("whence must be 0, or 1")
        if self._point < 0:  # pylint: disable=consider-using-max-builtin
            self._point = 0  # XXX is this right?

    def read(self, size):
        try:
            # keep reading from the generator until the buffer of this stream has enough data to read
            while self._point + size > self._download_offset:
                self._buf += self.__next__()
        except StopIteration:
            self.file_length = self._download_offset

        start_point = self._point

        # EOF
        self._point = min(self._point + size, self._download_offset)

        relative_start = start_point - self._buf_start
        if relative_start < 0:
            raise ValueError("Buffer has dumped too much data")
        relative_end = relative_start + size
        data = self._buf[relative_start:relative_end]

        # dump the extra data in buffer
        # buffer start--------------------16bytes----current read position
        dumped_size = max(relative_end - 16 - relative_start, 0)
        self._buf_start += dumped_size
        self._buf = self._buf[dumped_size:]

        return data
