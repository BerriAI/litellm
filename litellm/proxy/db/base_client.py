from typing import Any, Literal, List


class CustomDB:
    """
    Implements a base class that we expect any custom db implementation (e.g. DynamoDB) to follow
    """

    def __init__(self) -> None:
        pass

    def get_data(self, key: str, table_name: Literal["user", "key", "config"]):
        """
        Check if key valid
        """
        pass

    def insert_data(self, value: Any, table_name: Literal["user", "key", "config"]):
        """
        For new key / user logic
        """
        pass

    def update_data(
        self, key: str, value: Any, table_name: Literal["user", "key", "config"]
    ):
        """
        For cost tracking logic
        """
        pass

    def delete_data(
        self, keys: List[str], table_name: Literal["user", "key", "config"]
    ):
        """
        For /key/delete endpoint s
        """

    def connect(
        self,
    ):
        """
        For connecting to db and creating / updating any tables
        """
        pass

    def disconnect(
        self,
    ):
        """
        For closing connection on server shutdown
        """
        pass
