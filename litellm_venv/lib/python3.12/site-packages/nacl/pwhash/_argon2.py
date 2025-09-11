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

import nacl.bindings

_argon2_strbytes_plus_one = nacl.bindings.crypto_pwhash_STRBYTES

PWHASH_SIZE = _argon2_strbytes_plus_one - 1
SALTBYTES = nacl.bindings.crypto_pwhash_SALTBYTES

PASSWD_MIN = nacl.bindings.crypto_pwhash_PASSWD_MIN
PASSWD_MAX = nacl.bindings.crypto_pwhash_PASSWD_MAX

PWHASH_SIZE = _argon2_strbytes_plus_one - 1

BYTES_MAX = nacl.bindings.crypto_pwhash_BYTES_MAX
BYTES_MIN = nacl.bindings.crypto_pwhash_BYTES_MIN

ALG_ARGON2I13 = nacl.bindings.crypto_pwhash_ALG_ARGON2I13
ALG_ARGON2ID13 = nacl.bindings.crypto_pwhash_ALG_ARGON2ID13
ALG_ARGON2_DEFAULT = nacl.bindings.crypto_pwhash_ALG_DEFAULT


def verify(password_hash: bytes, password: bytes) -> bool:
    """
    Takes a modular crypt encoded argon2i or argon2id stored password hash
    and checks if the user provided password will hash to the same string
    when using the stored parameters

    :param password_hash: password hash serialized in modular crypt() format
    :type password_hash: bytes
    :param password: user provided password
    :type password: bytes
    :rtype: boolean

    .. versionadded:: 1.2
    """
    return nacl.bindings.crypto_pwhash_str_verify(password_hash, password)
