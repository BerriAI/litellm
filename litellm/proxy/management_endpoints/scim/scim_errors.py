from fastapi import HTTPException


class ScimUserAlreadyExists(HTTPException):
    """
    Exception raised when a user already exists in the database.
    """
    def __init__(self, message: str, scim_type: str = "uniqueness"):
        super().__init__(status_code=409, detail=message)
        self.message = message
        self.scim_type = scim_type
        self.schemas = ["urn:ietf:params:scim:api:messages:2.0:Error"]
    