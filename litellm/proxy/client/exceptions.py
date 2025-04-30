from typing import Optional
import requests

class UnauthorizedError(Exception):
    """Exception raised when the API returns a 401 Unauthorized response."""
    
    def __init__(self, orig_exception: requests.exceptions.HTTPError):
        self.orig_exception = orig_exception
        super().__init__(str(orig_exception)) 