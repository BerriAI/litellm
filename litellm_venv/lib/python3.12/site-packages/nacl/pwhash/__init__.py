# Copyright 2017 Donald Stufft and individual contributors
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


from nacl.exceptions import CryptPrefixError

from . import _argon2, argon2i, argon2id, scrypt

STRPREFIX = argon2id.STRPREFIX

PWHASH_SIZE = argon2id.PWHASH_SIZE

assert _argon2.ALG_ARGON2_DEFAULT == _argon2.ALG_ARGON2ID13
# since version 1.0.15 of libsodium

PASSWD_MIN = argon2id.PASSWD_MIN
PASSWD_MAX = argon2id.PASSWD_MAX
MEMLIMIT_MAX = argon2id.MEMLIMIT_MAX
MEMLIMIT_MIN = argon2id.MEMLIMIT_MIN
OPSLIMIT_MAX = argon2id.OPSLIMIT_MAX
OPSLIMIT_MIN = argon2id.OPSLIMIT_MIN
OPSLIMIT_INTERACTIVE = argon2id.OPSLIMIT_INTERACTIVE
MEMLIMIT_INTERACTIVE = argon2id.MEMLIMIT_INTERACTIVE
OPSLIMIT_MODERATE = argon2id.OPSLIMIT_MODERATE
MEMLIMIT_MODERATE = argon2id.MEMLIMIT_MODERATE
OPSLIMIT_SENSITIVE = argon2id.OPSLIMIT_SENSITIVE
MEMLIMIT_SENSITIVE = argon2id.MEMLIMIT_SENSITIVE

str = argon2id.str

assert argon2i.ALG != argon2id.ALG

SCRYPT_SALTBYTES = scrypt.SALTBYTES
SCRYPT_PWHASH_SIZE = scrypt.PWHASH_SIZE
SCRYPT_OPSLIMIT_INTERACTIVE = scrypt.OPSLIMIT_INTERACTIVE
SCRYPT_MEMLIMIT_INTERACTIVE = scrypt.MEMLIMIT_INTERACTIVE
SCRYPT_OPSLIMIT_SENSITIVE = scrypt.OPSLIMIT_SENSITIVE
SCRYPT_MEMLIMIT_SENSITIVE = scrypt.MEMLIMIT_SENSITIVE


kdf_scryptsalsa208sha256 = scrypt.kdf
scryptsalsa208sha256_str = scrypt.str
verify_scryptsalsa208sha256 = scrypt.verify


def verify(password_hash: bytes, password: bytes) -> bool:
    """
    Takes a modular crypt encoded stored password hash derived using one
    of the algorithms supported by `libsodium` and checks if the user provided
    password will hash to the same string when using the parameters saved
    in the stored hash
    """
    if password_hash.startswith(argon2id.STRPREFIX):
        return argon2id.verify(password_hash, password)
    elif password_hash.startswith(argon2i.STRPREFIX):
        return argon2id.verify(password_hash, password)
    elif scrypt.AVAILABLE and password_hash.startswith(scrypt.STRPREFIX):
        return scrypt.verify(password_hash, password)
    else:
        raise (CryptPrefixError("given password_hash is not in a supported format"))
