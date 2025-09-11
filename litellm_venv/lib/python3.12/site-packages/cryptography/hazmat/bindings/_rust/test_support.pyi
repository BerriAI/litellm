# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import pkcs7

class TestCertificate:
    not_after_tag: int
    not_before_tag: int
    issuer_value_tags: list[int]
    subject_value_tags: list[int]

def test_parse_certificate(data: bytes) -> TestCertificate: ...
def pkcs7_decrypt(
    encoding: serialization.Encoding,
    msg: bytes,
    pkey: serialization.pkcs7.PKCS7PrivateKeyTypes,
    cert_recipient: x509.Certificate,
    options: list[pkcs7.PKCS7Options],
) -> bytes: ...
def pkcs7_verify(
    encoding: serialization.Encoding,
    sig: bytes,
    msg: bytes | None,
    certs: list[x509.Certificate],
    options: list[pkcs7.PKCS7Options],
) -> None: ...
