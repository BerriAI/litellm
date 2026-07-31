from pydantic import BaseModel, Field


class ForgotPasswordRequest(BaseModel):
    email: str


class ValidateResetPasswordTokenRequest(BaseModel):
    token: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=1)
