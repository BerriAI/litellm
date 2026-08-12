"""
Claude Code Marketplace endpoint types for LiteLLM Proxy
"""

from typing import Final, Literal

from pydantic import BaseModel, Field

SkillApprovalStatus = Literal["pending_review", "active", "rejected"]

SKILL_PENDING_REVIEW: Final[SkillApprovalStatus] = "pending_review"
SKILL_ACTIVE: Final[SkillApprovalStatus] = "active"
SKILL_REJECTED: Final[SkillApprovalStatus] = "rejected"


class PluginAuthor(BaseModel):
    """Plugin author information."""

    name: str = Field(..., description="Author name")
    email: str | None = Field(None, description="Author email")


class PluginOwner(BaseModel):
    """Marketplace owner information."""

    name: str = Field(..., description="Owner name")
    email: str | None = Field(None, description="Owner email")


class PluginSpec(BaseModel):
    """Mutable fields shared by plugin create and update requests."""

    source: dict[str, str] = Field(
        ...,
        description=(
            "Git source reference. Supported formats:\n"
            "- GitHub: {'source': 'github', 'repo': 'org/repo'}\n"
            "- Git URL: {'source': 'url', 'url': 'https://github.com/org/repo.git'}\n"
            "- Git Subdir: {'source': 'git-subdir', 'url': 'https://github.com/org/repo.git', 'path': 'plugins/plugin-name'}"
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


class RegisterPluginRequest(PluginSpec):
    """
    Request body for registering a plugin in the marketplace.

    LiteLLM acts as a registry/discovery layer. Plugins are hosted on
    GitHub/GitLab/Bitbucket and referenced by their git source.
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


class PluginResponse(BaseModel):
    """Plugin information in API responses."""

    id: str = Field(..., description="Plugin unique ID")
    name: str = Field(..., description="Plugin name")
    version: str | None = Field(None, description="Plugin version")
    description: str | None = Field(None, description="Plugin description")
    source: dict[str, str] = Field(..., description="Git source reference")
    enabled: bool = Field(..., description="Whether plugin is enabled")
    approval_status: SkillApprovalStatus = Field("active", description="Administrator approval state")


class RegisterPluginResponse(BaseModel):
    """Response from plugin registration."""

    status: str = Field(..., description="Operation status")
    action: str = Field(..., description="Action taken (created/submitted_for_review/updated)")
    plugin: PluginResponse = Field(..., description="Plugin information")


class ApprovePluginRequest(BaseModel):
    """
    Administrator approval of a submitted skill.

    The fingerprint binds the approval to the content the administrator read.
    A submitter who edits their submission after it was read, but before it is
    approved, changes the fingerprint, so the approval is refused rather than
    publishing a manifest nobody reviewed.
    """

    reviewed_fingerprint: str = Field(
        ...,
        description="manifest_fingerprint of the submission that was reviewed, from GET /claude-code/plugins",
    )
    review_notes: str | None = Field(None, description="Reviewer feedback shown to the submitter")


class RejectPluginRequest(BaseModel):
    """Administrator rejection of a submitted skill."""

    review_notes: str | None = Field(None, description="Reviewer feedback shown to the submitter")


class ReviewPluginResponse(BaseModel):
    """Response from approving or rejecting a submitted skill."""

    status: str = Field(..., description="Operation status")
    name: str = Field(..., description="Skill name")
    approval_status: SkillApprovalStatus = Field(..., description="Resulting approval state")
    enabled: bool = Field(..., description="Whether the skill is now served to users")
    reviewed_by: str | None = Field(None, description="User id of the reviewing administrator")
    reviewed_at: str | None = Field(None, description="ISO timestamp of the review")
    review_notes: str | None = Field(None, description="Reviewer feedback")


class PluginListItem(BaseModel):
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
    enabled: bool
    approval_status: SkillApprovalStatus = "active"
    manifest_fingerprint: str = ""
    review_notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    created_by: str | None = None
    created_at: str | None
    updated_at: str | None


class ListPluginsResponse(BaseModel):
    """Response from listing plugins."""

    plugins: list[PluginListItem]
    count: int


class MarketplacePluginEntry(BaseModel):
    """Plugin entry in marketplace.json."""

    name: str
    source: dict[str, str]
    version: str | None = None
    description: str | None = None
    author: PluginAuthor | None = None
    homepage: str | None = None
    keywords: list[str] | None = None
    category: str | None = None


class MarketplaceResponse(BaseModel):
    """
    Marketplace catalog response.

    This format is consumed by Claude Code CLI.
    See: https://docs.anthropic.com/en/docs/claude-code/plugins
    """

    name: str = Field(..., description="Marketplace identifier")
    owner: PluginOwner = Field(..., description="Marketplace owner")
    plugins: list[MarketplacePluginEntry] = Field(default_factory=list, description="Available plugins")
