from typing import TypedDict


class SecretFields(TypedDict):
    """
    Stored in data["secret_fields"]

    these fields are not logged, but used for internal purposes.
    """

    raw_headers: dict
