"""
Claude Code Marketplace endpoint types for LiteLLM Proxy
"""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class PluginAuthor(BaseModel):
    """Plugin author information."""

    name: str = Field(..., description="Author name")
    email: Optional[str] = Field(None, description="Author email")


class PluginOwner(BaseModel):
    """Marketplace owner information."""

    name: str = Field(..., description="Owner name")
    email: Optional[str] = Field(None, description="Owner email")


class RegisterPluginRequest(BaseModel):
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
    source: Dict[str, str] = Field(
        ...,
        description=(
            "Git source reference. Supported formats:\n"
            "- GitHub: {'source': 'github', 'repo': 'org/repo'}\n"
            "- Git URL: {'source': 'url', 'url': 'https://github.com/org/repo.git'}\n"
            "- Git Subdir: {'source': 'git-subdir', 'url': 'https://github.com/org/repo.git', 'path': 'plugins/plugin-name'}"
        ),
    )
    version: Optional[str] = Field("1.0.0", description="Semantic version")
    description: Optional[str] = Field(None, description="Plugin description")
    author: Optional[PluginAuthor] = Field(None, description="Plugin author")
    homepage: Optional[str] = Field(None, description="Plugin homepage URL")
    keywords: Optional[List[str]] = Field(None, description="Search keywords")
    category: Optional[str] = Field(None, description="Plugin category")
    domain: Optional[str] = Field(None, description="Skill domain (e.g., 'Productivity')")
    namespace: Optional[str] = Field(None, description="Skill namespace within domain (e.g., 'workflows')")


class PluginResponse(BaseModel):
    """Plugin information in API responses."""

    id: str = Field(..., description="Plugin unique ID")
    name: str = Field(..., description="Plugin name")
    version: Optional[str] = Field(None, description="Plugin version")
    description: Optional[str] = Field(None, description="Plugin description")
    source: Dict[str, str] = Field(..., description="Git source reference")
    enabled: bool = Field(..., description="Whether plugin is enabled")


class RegisterPluginResponse(BaseModel):
    """Response from plugin registration."""

    status: str = Field(..., description="Operation status")
    action: str = Field(..., description="Action taken (created/updated)")
    plugin: PluginResponse = Field(..., description="Plugin information")


class PluginListItem(BaseModel):
    """Plugin item in list responses."""

    id: str
    name: str
    version: Optional[str]
    description: Optional[str]
    source: Dict[str, str]
    author: Optional[PluginAuthor] = None
    homepage: Optional[str] = None
    keywords: Optional[List[str]] = None
    category: Optional[str] = None
    domain: Optional[str] = None
    namespace: Optional[str] = None
    enabled: bool
    created_at: Optional[str]
    updated_at: Optional[str]


class ListPluginsResponse(BaseModel):
    """Response from listing plugins."""

    plugins: List[PluginListItem]
    count: int


class MarketplacePluginEntry(BaseModel):
    """Plugin entry in marketplace.json."""

    name: str
    source: Dict[str, str]
    version: Optional[str] = None
    description: Optional[str] = None
    author: Optional[PluginAuthor] = None
    homepage: Optional[str] = None
    keywords: Optional[List[str]] = None
    category: Optional[str] = None


class MarketplaceResponse(BaseModel):
    """
    Marketplace catalog response.

    This format is consumed by Claude Code CLI.
    See: https://docs.anthropic.com/en/docs/claude-code/plugins
    """

    name: str = Field(..., description="Marketplace identifier")
    owner: PluginOwner = Field(..., description="Marketplace owner")
    plugins: List[MarketplacePluginEntry] = Field(default_factory=list, description="Available plugins")


# --- Multi-marketplace import (LiteLLM_SkillMarketplaceTable) ---

MarketplaceSourceType = Literal["claude_marketplace_json", "claude_plugin_json", "skills_dir", "managed"]


class GithubSource(BaseModel):
    source: Literal["github"] = "github"
    repo: str = Field(..., description="'org/repo'")


class UrlSource(BaseModel):
    source: Literal["url"] = "url"
    url: str


class GitSubdirSource(BaseModel):
    source: Literal["git-subdir"] = "git-subdir"
    url: str
    path: str


PluginSourceConfig = Annotated[
    Union[GithubSource, UrlSource, GitSubdirSource],
    Field(discriminator="source"),
]


class RegisterMarketplaceRequest(BaseModel):
    """Request body for importing an external Claude Code marketplace."""

    source: str = Field(
        ...,
        description=("'org/repo' shorthand, a github/gitlab/bitbucket URL, or a direct URL to a marketplace.json file"),
    )
    name: Optional[str] = Field(
        None, description="Marketplace slug to register under. Defaults to a slug derived from the source."
    )


class MarketplaceSourceResponse(BaseModel):
    """A registered marketplace source and its current sync state."""

    id: str
    name: str
    display_name: Optional[str] = None
    source_type: MarketplaceSourceType
    source_ref: Optional[str] = None
    branch: Optional[str] = None
    enabled: bool
    sync_status: str
    sync_error: Optional[str] = None
    last_synced_at: Optional[str] = None
    plugin_count: Optional[int] = None
    skipped_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ListMarketplacesResponse(BaseModel):
    marketplaces: List[MarketplaceSourceResponse]
    count: int


class SyncMarketplaceResponse(BaseModel):
    status: str
    marketplace: MarketplaceSourceResponse
