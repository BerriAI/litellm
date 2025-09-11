# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import inspect
import sys
from io import BytesIO
from typing import IO

from .._encryption import _GCM_REGION_DATA_LENGTH, encrypt_data_v2


class GCMBlobEncryptionStream:
    """
    An async stream that performs AES-GCM encryption on the given data as
    it's streamed. Data is read and encrypted in regions. The stream
    will use the same encryption key and will generate a guaranteed unique
    nonce for each encryption region.
    """

    def __init__(
        self,
        content_encryption_key: bytes,
        data_stream: IO[bytes],
    ) -> None:
        """
        :param bytes content_encryption_key: The encryption key to use.
        :param IO[bytes] data_stream: The data stream to read data from.
        """
        self.content_encryption_key = content_encryption_key
        self.data_stream = data_stream

        self.offset = 0
        self.current = b""
        self.nonce_counter = 0

    async def read(self, size: int = -1) -> bytes:
        """
        Read data from the stream. Specify -1 to read all available data.

        :param int size: The amount of data to read. Defaults to -1 for all data.
        :return: The bytes read.
        :rtype: bytes
        """
        result = BytesIO()
        remaining = sys.maxsize if size == -1 else size

        while remaining > 0:
            # Start by reading from current
            if len(self.current) > 0:
                read = min(remaining, len(self.current))
                result.write(self.current[:read])

                self.current = self.current[read:]
                self.offset += read
                remaining -= read

            if remaining > 0:
                # Read one region of data and encrypt it
                data = self.data_stream.read(_GCM_REGION_DATA_LENGTH)
                if inspect.isawaitable(data):
                    data = await data

                if len(data) == 0:
                    # No more data to read
                    break

                self.current = encrypt_data_v2(
                    data, self.nonce_counter, self.content_encryption_key
                )
                # IMPORTANT: Must increment the nonce each time.
                self.nonce_counter += 1

        return result.getvalue()
