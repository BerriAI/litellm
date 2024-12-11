from collections.abc import Iterable
from typing import List


def is_tokens_or_list_of_tokens(value: List):
    # Check if it's a list of integers (tokens)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return True
    # Check if it's a list of lists of integers (list of tokens)
    if isinstance(value, list) and all(
        isinstance(item, list) and all(isinstance(i, int) for i in item)
        for item in value
    ):
        return True
    return False
