"""
Implementation of Fowler/Noll/Vo hash algorithm in pure Python.
See http://isthe.com/chongo/tech/comp/fnv/
"""
import sys


FNV_64_PRIME = 0x100000001B3
FNV1_64_INIT = 0xCBF29CE484222325


def no_op(c):
    return c


if sys.version_info[0] == 3:
    _get_byte = no_op
else:
    _get_byte = ord


def fnv(data, hval_init, fnv_prime, fnv_size):
    # type: (bytes, int, int, int) -> int
    """
    Core FNV hash algorithm used in FNV0 and FNV1.
    """
    hval = hval_init
    for byte in data:
        hval = (hval * fnv_prime) % fnv_size
        hval = hval ^ _get_byte(byte)
    return hval


def fnv1_64(data):
    # type: (bytes) -> int
    """
    Returns the 64 bit FNV-1 hash value for the given data.
    """
    return fnv(data, FNV1_64_INIT, FNV_64_PRIME, 2**64)
