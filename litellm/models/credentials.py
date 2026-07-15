"""
Credential table models.

These are the canonical credential types for the proxy. They live in the model
layer; ``litellm.types.utils`` re-exports them for backwards compatibility.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CredentialBase(BaseModel):
    credential_name: str
    credential_info: dict


class CredentialItem(CredentialBase):
    credential_values: dict


class CreateCredentialItem(CredentialBase):
    credential_values: dict | None = None
    model_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_credential_params(cls, values):
        if not values.get("credential_values") and not values.get("model_id"):
            raise ValueError("Either credential_values or model_id must be set")
        return values


class UpdateCredentialItem(BaseModel):
    """PATCH body for ``/credentials/{name}``.

    Both ``credential_values`` and ``credential_info`` are optional so a caller
    can patch one without sending the other (team-admins patching access without
    knowing the upstream secrets; proxy admins rotating values without touching
    access). ``credential_name`` is optional because most patches don't rename.
    """

    credential_name: str | None = None
    credential_values: dict | None = None
    credential_info: dict | None = None


class CredentialAccess(BaseModel):
    """Destination-side access list on a logging credential.

    ``global`` is exposed via the JSON name "global" through a field alias since
    that's a Python keyword. ``populate_by_name`` keeps internal Python code
    using ``global_`` working while JSON in/out uses "global".
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    global_: bool = Field(default=False, alias="global")
    teams: tuple[str, ...] = ()
    orgs: tuple[str, ...] = ()


class CredentialInfo(BaseModel):
    """Typed shape of ``credential_info`` for the access-control decider.

    Existing stored credentials carry arbitrary extra fields (e.g.
    ``custom_llm_provider``); ``extra="allow"`` preserves them. The decider
    inspects ``model_fields_set`` to learn which fields the caller actually
    patched, which is what Pydantic gives us natively without dict-key spelunking.
    """

    model_config = ConfigDict(extra="allow")

    credential_type: str | None = None
    description: str | None = None
    host: str | None = None
    endpoint: str | None = None
    access: CredentialAccess | None = None
    auto_enable: bool = False
