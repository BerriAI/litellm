# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

# pylint: disable=enum-must-be-uppercase

from enum import Enum

from azure.core import CaseInsensitiveEnumMeta


class KeyCurveName(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Supported elliptic curves"""

    p_256 = "P-256"  #: The NIST P-256 elliptic curve, AKA SECG curve SECP256R1.
    p_384 = "P-384"  #: The NIST P-384 elliptic curve, AKA SECG curve SECP384R1.
    p_521 = "P-521"  #: The NIST P-521 elliptic curve, AKA SECG curve SECP521R1.
    p_256_k = "P-256K"  #: The SECG SECP256K1 elliptic curve.


class KeyExportEncryptionAlgorithm(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Supported algorithms for protecting exported key material"""

    ckm_rsa_aes_key_wrap = "CKM_RSA_AES_KEY_WRAP"
    rsa_aes_key_wrap_256 = "RSA_AES_KEY_WRAP_256"
    rsa_aes_key_wrap_384 = "RSA_AES_KEY_WRAP_384"


class KeyOperation(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Supported key operations"""

    encrypt = "encrypt"
    decrypt = "decrypt"
    import_key = "import"
    sign = "sign"
    verify = "verify"
    wrap_key = "wrapKey"
    unwrap_key = "unwrapKey"
    export = "export"


class KeyRotationPolicyAction(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The action that will be executed in a key rotation policy"""

    rotate = "Rotate"  #: Rotate the key based on the key policy.
    notify = "Notify"  #: Trigger Event Grid events.

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        raise ValueError(f"{value} is not a valid KeyRotationPolicyAction")


class KeyType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Supported key types"""

    ec = "EC"  #: Elliptic Curve
    ec_hsm = "EC-HSM"  #: Elliptic Curve with a private key which is not exportable from the HSM
    rsa = "RSA"  #: RSA (https://tools.ietf.org/html/rfc3447)
    rsa_hsm = "RSA-HSM"  #: RSA with a private key which is not exportable from the HSM
    oct = "oct"  #: Octet sequence (used to represent symmetric keys)
    oct_hsm = "oct-HSM"  #: Octet sequence with a private key which is not exportable from the HSM

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        raise ValueError(f"{value} is not a valid KeyType")
