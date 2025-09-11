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


# We create a clone of various builtin Exception types which additionally
# inherit from CryptoError. Below, we refer to the parent types via the
# `builtins` namespace, so mypy can distinguish between (e.g.)
# `nacl.exceptions.RuntimeError` and `builtins.RuntimeError`.
import builtins
from typing import Type


class CryptoError(Exception):
    """
    Base exception for all nacl related errors
    """


class BadSignatureError(CryptoError):
    """
    Raised when the signature was forged or otherwise corrupt.
    """


class RuntimeError(builtins.RuntimeError, CryptoError):
    pass


class AssertionError(builtins.AssertionError, CryptoError):
    pass


class TypeError(builtins.TypeError, CryptoError):
    pass


class ValueError(builtins.ValueError, CryptoError):
    pass


class InvalidkeyError(CryptoError):
    pass


class CryptPrefixError(InvalidkeyError):
    pass


class UnavailableError(RuntimeError):
    """
    is a subclass of :class:`~nacl.exceptions.RuntimeError`, raised when
    trying to call functions not available in a minimal build of
    libsodium.
    """

    pass


def ensure(cond: bool, *args: object, **kwds: Type[Exception]) -> None:
    """
    Return if a condition is true, otherwise raise a caller-configurable
    :py:class:`Exception`
    :param bool cond: the condition to be checked
    :param sequence args: the arguments to be passed to the exception's
                          constructor
    The only accepted named parameter is `raising` used to configure the
    exception to be raised if `cond` is not `True`
    """
    _CHK_UNEXP = "check_condition() got an unexpected keyword argument {0}"

    raising = kwds.pop("raising", AssertionError)
    if kwds:
        raise TypeError(_CHK_UNEXP.format(repr(kwds.popitem()[0])))

    if cond is True:
        return
    raise raising(*args)
