# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

import typing

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs7

def serialize_certificates(
    certs: list[x509.Certificate],
    encoding: serialization.Encoding,
) -> bytes: ...
def encrypt_and_serialize(
    builder: pkcs7.PKCS7EnvelopeBuilder,
    encoding: serialization.Encoding,
    options: typing.Iterable[pkcs7.PKCS7Options],
) -> bytes: ...
def sign_and_serialize(
    builder: pkcs7.PKCS7SignatureBuilder,
    encoding: serialization.Encoding,
    options: typing.Iterable[pkcs7.PKCS7Options],
) -> bytes: ...
def load_pem_pkcs7_certificates(
    data: bytes,
) -> list[x509.Certificate]: ...
def load_der_pkcs7_certificates(
    data: bytes,
) -> list[x509.Certificate]: ...
