# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
import codecs
from base64 import b64encode, b64decode

from cryptography.hazmat.primitives.asymmetric import utils


def _bytes_to_int(b):
    if not b or not isinstance(b, bytes):
        raise ValueError("b must be non-empty byte string")

    return int(codecs.encode(b, "hex"), 16)


def _int_to_bytes(i):
    h = hex(i)
    if len(h) > 1 and h[0:2] == "0x":
        h = h[2:]

    # need to strip L in python 2.x
    h = h.strip("L")

    if len(h) % 2:
        h = "0" + h
    return codecs.decode(h, "hex")


def _bstr_to_b64url(bstr):
    """Serialize bytes into base-64 string.

    :param bytes bstr: Object to be serialized.

    :returns: The base-64 URL encoded string.
    :rtype: str
    """
    encoded = b64encode(bstr).decode()
    return encoded.strip("=").replace("+", "-").replace("/", "_")


def _str_to_b64url(s):
    """Serialize str into base-64 string.

    :param str s: Object to be serialized.

    :returns: The base-64 URL encoded string.
    :rtype: str
    """
    return _bstr_to_b64url(s.encode(encoding="utf8"))


def _b64_to_bstr(b64str):
    """Deserialize base-64 encoded string into string.

    :param str b64str: response string to be deserialized.

    :returns: The decoded bytes.
    :rtype: bytes

    :raises: TypeError if string format invalid.
    """
    padding = "=" * (3 - (len(b64str) + 3) % 4)
    b64str = b64str + padding
    encoded = b64str.replace("-", "+").replace("_", "/")
    return b64decode(encoded)


def _b64_to_str(b64str):
    """Deserialize base-64 encoded string into string.

    :param str b64str: response string to be deserialized.

    :returns: The decoded string.
    :rtype: str

    :raises: TypeError if string format invalid.
    """
    return _b64_to_bstr(b64str).decode("utf8")


def _int_to_fixed_length_bigendian_bytes(i, length):
    """Convert an integer to a bigendian byte string left-padded with zeroes to a fixed length.

    :param int i: The integer to convert.
    :param int length: The length of the desired byte string.

    :returns: A bigendian byte string of length `length`, representing integer `i`.
    :rtype: bytes
    """

    b = _int_to_bytes(i)

    if len(b) > length:
        raise ValueError(f"{i} is too large to be represented by {length} bytes")

    if len(b) < length:
        b = (b"\0" * (length - len(b))) + b

    return b


def ecdsa_to_asn1_der(signature):
    """ASN.1 DER encode an ECDSA signature.

    :param bytes signature: ECDSA signature encoded according to RFC 7518, i.e. the concatenated big-endian bytes of
      two integers (as produced by Key Vault)

    :returns: signature, ASN.1 DER encoded (as expected by ``cryptography``)
    :rtype: bytes
    """
    mid = len(signature) // 2
    r = _bytes_to_int(signature[:mid])
    s = _bytes_to_int(signature[mid:])
    return utils.encode_dss_signature(r, s)


def asn1_der_to_ecdsa(signature, algorithm):
    """Convert an ASN.1 DER encoded signature to ECDSA encoding.

    :param bytes signature: an ASN.1 DER encoded ECDSA signature (as produced by ``cryptography``)
    :param _Ecdsa algorithm: signing algorithm which produced ``signature``

    :returns: signature encoded according to RFC 7518 (as expected by Key Vault)
    :rtype: bytes
    """
    r, s = utils.decode_dss_signature(signature)
    r_bytes = _int_to_fixed_length_bigendian_bytes(r, algorithm.coordinate_length)
    s_bytes = _int_to_fixed_length_bigendian_bytes(s, algorithm.coordinate_length)
    return r_bytes + s_bytes
