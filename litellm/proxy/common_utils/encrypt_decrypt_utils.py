import base64

from litellm._logging import verbose_proxy_logger


def encrypt_value_helper(value: str):
    from litellm.proxy.proxy_server import master_key

    try:
        if isinstance(value, str):
            encrypted_value = encrypt_value(value=value, master_key=master_key)  # type: ignore
            encrypted_value = base64.b64encode(encrypted_value).decode("utf-8")

            return encrypted_value

        raise ValueError(
            f"Invalid value type passed to encrypt_value: {type(value)} for Value: {value}\n Value must be a string"
        )
    except Exception as e:
        raise e


def decrypt_value_helper(value: str):
    from litellm.proxy.proxy_server import master_key

    try:
        if isinstance(value, str):
            decoded_b64 = base64.b64decode(value)
            value = decrypt_value(value=decoded_b64, master_key=master_key)  # type: ignore
            return value
    except Exception as e:
        verbose_proxy_logger.error(f"Error decrypting value: {value}\nError: {str(e)}")
        # [Non-Blocking Exception. - this should not block decrypting other values]
        pass


def encrypt_value(value: str, master_key: str):
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(master_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # encode message #
    value_bytes = value.encode("utf-8")

    encrypted = box.encrypt(value_bytes)

    return encrypted


def decrypt_value(value: bytes, master_key: str) -> str:
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(master_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # Convert the bytes object to a string
    plaintext = box.decrypt(value)

    plaintext = plaintext.decode("utf-8")  # type: ignore
    return plaintext  # type: ignore
