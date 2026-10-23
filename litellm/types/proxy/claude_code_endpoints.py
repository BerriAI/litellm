"""
Claude Code Marketplace endpoint types for LiteLLM Proxy
"""

from typing import Final, Literal, TypeAlias

from pydantic import AliasChoices, ConfigDict, Field, TypeAdapter, model_validator

from litellm.types.llms.base import LiteLLMBaseModel

InstallationPreference: TypeAlias = Literal["available", "auto_install", "required"]
_BOTH_PREFERENCE_SPELLINGS: Final = frozenset({"installation_preference", "installationPreference"})
_REQUEST_KEYS: Final = TypeAdapter(dict[str, object])


class PluginAuthor(LiteLLMBaseModel):
    """Plugin author information."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(None, description="Author email")


class PluginOwner(LiteLLMBaseModel):
    """Marketplace owner information."""

    name: str = Field(..., description="Owner name")
    email: str | None = Field(None, description="Owner email")


class PluginSpec(LiteLLMBaseModel):
    """Mutable fields shared by plugin create and update requests."""

    source: dict[str, str] = Field(
        ...,
        description=(
            "Plugin source reference. Supported formats:\n"
            "- GitHub: {'source': 'github', 'repo': 'org/repo'}\n"
            "- Git URL: {'source': 'url', 'url': 'https://github.com/org/repo.git'}\n"
            "- Git Subdir: {'source': 'git-subdir', 'url': 'https://github.com/org/repo.git', 'path': 'plugins/plugin-name'}\n"
            "- Zip archive on any https host (e.g. S3): "
            "{'source': 'archive', 'url': 'https://bucket.s3.amazonaws.com/plugin.zip', "
            "'sha256': '<optional hex digest>'}\n"
            "- Gateway-hosted archive: {'source': 'hosted'}; upload the zip to LiteLLM after registering the plugin"
        ),
    )
    version: str | None = Field("1.0.0", description="Semantic version")
    description: str | None = Field(None, description="Plugin description")
    author: PluginAuthor | None = Field(None, description="Plugin author")
    homepage: str | None = Field(None, description="Plugin homepage URL")
    keywords: list[str] | None = Field(None, description="Search keywords")
    category: str | None = Field(None, description="Plugin category")
    domain: str | None = Field(None, description="Skill domain (e.g., 'Productivity')")
    namespace: str | None = Field(None, description="Skill namespace within domain (e.g., 'workflows')")
    installation_preference: InstallationPreference | None = Field(
        None,
        validation_alias=AliasChoices("installation_preference", "installationPreference"),
        description=(
            "Emitted as installationPreference on this plugin's marketplace.json entry. Needs an archive source, "
            "the only plugins Claude Desktop lists from a marketplace.json URL. LiteLLM supplies the sha256 for "
            "gateway-hosted archives. Claude Desktop honors the mark only when the archive and marketplace.json "
            "share its inference gateway origin"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def reject_both_preference_spellings(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        fields: Final = _REQUEST_KEYS.validate_python(data)
        if not _BOTH_PREFERENCE_SPELLINGS.issubset(fields):
            return fields
        if all(fields[key] is not None for key in _BOTH_PREFERENCE_SPELLINGS):
            raise ValueError("send installation_preference or installationPreference, not both")
        return {
            key: value for key, value in fields.items() if key not in _BOTH_PREFERENCE_SPELLINGS or value is not None
        }


class RegisterPluginRequest(PluginSpec):
    """
    Request body for registering a plugin in the marketplace.

    LiteLLM acts as a registry/discovery layer. Plugins are hosted on
    GitHub/GitLab/Bitbucket or as a zip archive on any https host and referenced by their source.
    """

    name: str = Field(
        ...,
        description="Plugin name (kebab-case, e.g., 'my-plugin')",
        pattern=r"^[a-z0-9-]+$",
    )


class UpdatePluginRequest(PluginSpec):
    """
    Request body for replacing an existing plugin.

    The plugin name is the resource identity and is supplied as the path
    parameter, so it cannot be changed here. This is a full replace: omitted
    fields reset to their defaults, so version is cleared rather than
    defaulting to the create-time "1.0.0".
    """

    version: str | None = Field(None, description="Semantic version; cleared if omitted")


class PluginResponse(LiteLLMBaseModel):
    """Plugin information in API responses."""

    id: str = Field(..., description="Plugin unique ID")
    name: str = Field(..., description="Plugin name")
    version: str | None = Field(None, description="Plugin version")
    description: str | None = Field(None, description="Plugin description")
    source: dict[str, str] = Field(..., description="Plugin source reference")
    installation_preference: InstallationPreference | None = Field(None, description="Plugin installation preference")
    enabled: bool = Field(..., description="Whether plugin is enabled")


class UploadPluginArchiveResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    plugin: PluginResponse = Field(..., description="Updated plugin information")
    sha256: str = Field(..., description="SHA-256 digest of the uploaded archive")
    size_bytes: int = Field(..., description="Size of the uploaded archive in bytes")


class RegisterPluginResponse(LiteLLMBaseModel):
    """Response from plugin registration."""

    status: str = Field(..., description="Operation status")
    action: str = Field(..., description="Action taken (created/updated)")
    plugin: PluginResponse = Field(..., description="Plugin information")


class PluginListItem(LiteLLMBaseModel):
    """Plugin item in list responses."""

    id: str
    name: str
    version: str | None
    description: str | None
    source: dict[str, str]
    author: PluginAuthor | None = None
    homepage: str | None = None
    keywords: list[str] | None = None
    category: str | None = None
    domain: str | None = None
    namespace: str | None = None
    installation_preference: InstallationPreference | None = None
    archive_sha256: str | None = None
    enabled: bool
    created_at: str | None
    updated_at: str | None


class ListPluginsResponse(LiteLLMBaseModel):
    """Response from listing plugins."""

    plugins: list[PluginListItem]
    count: int


class MarketplacePluginEntry(LiteLLMBaseModel):
    """Plugin entry in marketplace.json."""

    name: str
    source: dict[str, str]
    version: str | None = None
    description: str | None = None
    author: PluginAuthor | None = None
    homepage: str | None = None
    keywords: list[str] | None = None
    category: str | None = None
    installationPreference: InstallationPreference | None = None


class MarketplaceResponse(LiteLLMBaseModel):
    """
    Marketplace catalog response.

    This format is consumed by Claude Code CLI.
    See: https://docs.anthropic.com/en/docs/claude-code/plugins
    """

    name: str = Field(..., description="Marketplace identifier")
    owner: PluginOwner = Field(..., description="Marketplace owner")
    plugins: list[MarketplacePluginEntry] = Field(default_factory=list, description="Available plugins")
