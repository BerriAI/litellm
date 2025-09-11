# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


# pylint: disable=enum-must-be-uppercase
class KeyWrapAlgorithm(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Key wrapping algorithms"""

    aes_128 = "A128KW"
    aes_192 = "A192KW"
    aes_256 = "A256KW"
    # [Not recommended] Microsoft recommends using RSA_OAEP_256 or stronger algorithms for enhanced security.
    # Microsoft does *not* recommend RSA_OAEP, which is included solely for backwards compatibility.
    # RSA_OAEP utilizes SHA1, which has known collision problems.
    rsa_oaep = "RSA-OAEP"
    rsa_oaep_256 = "RSA-OAEP-256"
    # [Not recommended] Microsoft recommends using RSA_OAEP_256 or stronger algorithms for enhanced security.
    # Microsoft does *not* recommend RSA_1_5, which is included solely for backwards compatibility.
    # Cryptographic standards no longer consider RSA with the PKCS#1 v1.5 padding scheme secure for encryption.
    rsa1_5 = "RSA1_5"
    ckm_aes_key_wrap = "CKM_AES_KEY_WRAP"
    ckm_aes_key_wrap_pad = "CKM_AES_KEY_WRAP_PAD"


class EncryptionAlgorithm(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Encryption algorithms"""

    # [Not recommended] Microsoft recommends using RSA_OAEP_256 or stronger algorithms for enhanced security.
    # Microsoft does *not* recommend RSA_OAEP, which is included solely for backwards compatibility.
    # RSA_OAEP utilizes SHA1, which has known collision problems.
    rsa_oaep = "RSA-OAEP"
    rsa_oaep_256 = "RSA-OAEP-256"
    # [Not recommended] Microsoft recommends using RSA_OAEP_256 or stronger algorithms for enhanced security.
    # Microsoft does *not* recommend RSA_1_5, which is included solely for backwards compatibility.
    # Cryptographic standards no longer consider RSA with the PKCS#1 v1.5 padding scheme secure for encryption.
    rsa1_5 = "RSA1_5"
    a128_gcm = "A128GCM"
    a192_gcm = "A192GCM"
    a256_gcm = "A256GCM"
    a128_cbc = "A128CBC"
    a192_cbc = "A192CBC"
    a256_cbc = "A256CBC"
    a128_cbcpad = "A128CBCPAD"
    a192_cbcpad = "A192CBCPAD"
    a256_cbcpad = "A256CBCPAD"


class SignatureAlgorithm(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Signature algorithms, described in https://tools.ietf.org/html/rfc7518"""

    ps256 = "PS256"  #: RSASSA-PSS using SHA-256 and MGF1 with SHA-256
    ps384 = "PS384"  #: RSASSA-PSS using SHA-384 and MGF1 with SHA-384
    ps512 = "PS512"  #: RSASSA-PSS using SHA-512 and MGF1 with SHA-512
    rs256 = "RS256"  #: RSASSA-PKCS1-v1_5 using SHA-256
    rs384 = "RS384"  #: RSASSA-PKCS1-v1_5 using SHA-384
    rs512 = "RS512"  #: RSASSA-PKCS1-v1_5 using SHA-512
    es256 = "ES256"  #: ECDSA using P-256 and SHA-256
    es384 = "ES384"  #: ECDSA using P-384 and SHA-384
    es512 = "ES512"  #: ECDSA using P-521 and SHA-512
    es256_k = "ES256K"  #: ECDSA using P-256K and SHA-256
    hs256 = "HS256"  #: HMAC using SHA-256
    hs384 = "HS384"  #: HMAC using SHA-384
    hs512 = "HS512"  #: HMAC using SHA-512
