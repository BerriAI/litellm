"""
Utilities for embedding input normalization.
"""

from typing import List, Union


def flatten_double_wrapped_embedding_input(
    input_data: Union[str, List],
) -> Union[str, List]:
    """
    Normalize [[str, ...]] → [str, ...] to prevent providers
    (e.g. Bedrock Titan) from receiving a list instead of a string.

    Only flattens when ALL sublists contain exclusively strings;
    integer token arrays like [[1, 2, 3]] are left unchanged.
    """
    if (
        isinstance(input_data, list)
        and len(input_data) > 0
        and isinstance(input_data[0], list)
        and all(
            isinstance(sublist, list) and all(isinstance(s, str) for s in sublist)
            for sublist in input_data
        )
    ):
        return [
            item
            for sublist in input_data
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
    return input_data
