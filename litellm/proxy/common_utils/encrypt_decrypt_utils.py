import base64
import os

from litellm._logging import verbose_proxy_logger

LITELLM_SALT_KEY = os.getenv("LITELLM_SALT_KEY", None)
verbose_proxy_logger.debug(
    "LITELLM_SALT_KEY is None using master_key to encrypt/decrypt secrets stored in DB"
)


def encrypt_value_helper(value: str):
    from litellm.proxy.proxy_server import master_key

    signing_key = LITELLM_SALT_KEY
    if LITELLM_SALT_KEY is None:
        signing_key = master_key

    try:
        if isinstance(value, str):
            encrypted_value = encrypt_value(value=value, signing_key=signing_key)  # type: ignore
            encrypted_value = base64.b64encode(encrypted_value).decode("utf-8")

            return encrypted_value

        verbose_proxy_logger.debug(
            f"Invalid value type passed to encrypt_value: {type(value)} for Value: {value}\n Value must be a string"
        )
        # if it's not a string - do not encrypt it and return the value
        return value
    except Exception as e:
        raise e


def decrypt_value_helper(value: str):
    from litellm.proxy.proxy_server import master_key

    signing_key = LITELLM_SALT_KEY
    if LITELLM_SALT_KEY is None:
        signing_key = master_key

    try:
        if isinstance(value, str):
            decoded_b64 = base64.b64decode(value)
            value = decrypt_value(value=decoded_b64, signing_key=signing_key)  # type: ignore
            return value

        # if it's not str - do not decrypt it, return the value
        return value
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error decrypting value, Did your master_key/salt key change recently? : {value}\nError: {str(e)}\nSet permanent salt key - https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
        )
        # [Non-Blocking Exception. - this should not block decrypting other values]
        pass


def encrypt_value(value: str, signing_key: str):
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(signing_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # encode message #
    value_bytes = value.encode("utf-8")

    encrypted = box.encrypt(value_bytes)

    return encrypted


def decrypt_value(value: bytes, signing_key: str) -> str:
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(signing_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # Convert the bytes object to a string
    plaintext = box.decrypt(value)

    plaintext = plaintext.decode("utf-8")  # type: ignore
    return plaintext  # type: ignore
