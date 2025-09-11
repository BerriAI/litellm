# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import base64
import hashlib
import hmac

try:
    from urllib.parse import quote, unquote
except ImportError:
    from urllib2 import quote, unquote  # type: ignore


def url_quote(url):
    return quote(url)


def url_unquote(url):
    return unquote(url)


def encode_base64(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    encoded = base64.b64encode(data)
    return encoded.decode("utf-8")


def decode_base64_to_bytes(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.b64decode(data)


def decode_base64_to_text(data):
    decoded_bytes = decode_base64_to_bytes(data)
    return decoded_bytes.decode("utf-8")


def sign_string(key, string_to_sign, key_is_base64=True):
    if key_is_base64:
        key = decode_base64_to_bytes(key)
    else:
        if isinstance(key, str):
            key = key.encode("utf-8")
    if isinstance(string_to_sign, str):
        string_to_sign = string_to_sign.encode("utf-8")
    signed_hmac_sha256 = hmac.HMAC(key, string_to_sign, hashlib.sha256)
    digest = signed_hmac_sha256.digest()
    encoded_digest = encode_base64(digest)
    return encoded_digest
