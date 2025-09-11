"""Helper functions to generate state param."""

import base64
import os


def generate_random_state(length: int = 64) -> str:
    """Generate a url-safe string to use as a state."""
    bytes_length = int(length * 3 / 4)
    return base64.urlsafe_b64encode(os.urandom(bytes_length)).decode("utf-8")
