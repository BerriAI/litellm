# Copyright 2013 Donald Stufft and individual contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


def _sodium_init() -> None:
    ensure(
        lib.sodium_init() != -1,
        "Could not initialize sodium",
        raising=exc.RuntimeError,
    )


def sodium_init() -> None:
    """
    Initializes sodium, picking the best implementations available for this
    machine.
    """
    ffi.init_once(_sodium_init, "libsodium")
