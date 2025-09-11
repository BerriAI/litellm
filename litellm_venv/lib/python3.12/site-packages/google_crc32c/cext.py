# Copyright 2020 Google LLC
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

import struct

# NOTE: ``__config__`` **must** be the first import because it (may)
#       modify the search path used to locate shared libraries.
import google_crc32c.__config__  # type: ignore
from google_crc32c._crc32c import extend  # type: ignore
from google_crc32c._crc32c import value  # type: ignore
from google_crc32c._checksum import CommonChecksum


class Checksum(CommonChecksum):
    """Hashlib-alike helper for CRC32C operations.

    Args:
        initial_value (Optional[bytes]): the initial chunk of data from
            which the CRC32C checksum is computed.  Defaults to b''.
    """

    __slots__ = ("_crc",)

    def __init__(self, initial_value=b""):
        self._crc = value(initial_value)

    def update(self, chunk):
        """Update the checksum with a new chunk of data.

        Args:
            chunk (Optional[bytes]): a chunk of data used to extend
                the CRC32C checksum.
        """
        self._crc = extend(self._crc, chunk)
