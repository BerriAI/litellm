# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import Any, Dict, IO, Iterable, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from azure.storage.blob import BlobQueryReader


class DataLakeFileQueryReader:
    """A streaming object to read query results."""

    name: str
    """The name of the blob being queried."""
    file_system: str
    """The name of the file system being queried."""
    response_headers: Dict[str, Any]
    """The response_headers of the quick query request."""
    record_delimiter: str
    """The delimiter used to separate lines, or records with the data. The `records`
        method will return these lines via a generator."""

    def __init__(self, blob_query_reader: "BlobQueryReader") -> None:
        self.name = blob_query_reader.name
        self.file_system = blob_query_reader.container
        self.response_headers = blob_query_reader.response_headers
        self.record_delimiter = blob_query_reader.record_delimiter
        self._bytes_processed = 0
        self._blob_query_reader = blob_query_reader

    def __len__(self) -> int:
        return len(self._blob_query_reader)

    def readall(self) -> Union[bytes, str]:
        """Return all query results.

        This operation is blocking until all data is downloaded.
        If encoding has been configured - this will be used to decode individual
        records are they are received.

        :returns: All query results.
        :rtype: Union[bytes, str]
        """
        return self._blob_query_reader.readall()

    def readinto(self, stream: IO) -> None:
        """Download the query result to a stream.

        :param IO stream:
            The stream to download to. This can be an open file-handle,
            or any writable stream.
        :returns: None
        """
        self._blob_query_reader.readinto(stream)

    def records(self) -> Iterable[Union[bytes, str]]:
        """Returns a record generator for the query result.

        Records will be returned line by line.
        If encoding has been configured - this will be used to decode individual
        records are they are received.

        :returns: A record generator for the query result.
        :rtype: Iterable[Union[bytes, str]]
        """
        return self._blob_query_reader.records()
