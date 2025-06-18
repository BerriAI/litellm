from fastapi import HTTPException


class ScimUserAlreadyExists(HTTPException):
    """
    Exception raised when a user already exists in the database.
    """
    def __init__(self, message: str, scim_type: str = "uniqueness"):
        self.message = message
        self.scim_type = scim_type
        self.status_code = 409
        self.schemas = ["urn:ietf:params:scim:api:messages:2.0:Error"]
    