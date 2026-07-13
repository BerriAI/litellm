/**
 * TypeScript types for Claude Code Marketplace
 * API request/response shapes are synced from the generated OpenAPI types in @/lib/http/schema.
 */

import type { components } from "@/lib/http/schema";

// Kept hand-written: the backend types `source` as Dict[str, str], so the generated type is a
// loose string map; this discriminant union is what the parser and display helpers rely on.
export interface PluginSource {
  source: "github" | "url" | "git-subdir";
  repo?: string; // Format: "org/repo" for GitHub
  url?: string; // Full URL for other sources
  path?: string; // Subdirectory path for git-subdir
}

export type PluginAuthor = components["schemas"]["PluginAuthor"];

export interface Plugin {
  id: string;
  name: string; // kebab-case
  version?: string; // semantic version
  description?: string;
  source: PluginSource;
  author?: PluginAuthor;
  homepage?: string;
  keywords?: string[];
  category?: string;
  domain?: string;
  namespace?: string;
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
}

export interface PluginListItem {
  id: string;
  name: string;
  version?: string;
  description?: string;
  source: PluginSource;
  author?: PluginAuthor;
  homepage?: string;
  keywords?: string[];
  category?: string;
  domain?: string;
  namespace?: string;
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
}

export interface ListPluginsResponse {
  plugins: PluginListItem[];
  count: number;
}

// Request envelope synced from the OpenAPI spec, with `source` narrowed to our PluginSource
// union and `version` kept optional (the backend supplies its default).
export type SkillRegisterRequest = Omit<components["schemas"]["RegisterPluginRequest"], "source" | "version"> & {
  source: PluginSource;
  version?: string;
};

// Public marketplace types
export interface MarketplacePluginEntry {
  name: string;
  source: PluginSource;
  version?: string;
  description?: string;
  author?: PluginAuthor;
  homepage?: string;
  keywords?: string[];
  category?: string;
}

export interface MarketplaceOwner {
  name: string;
  email?: string;
}

export interface MarketplaceResponse {
  name: string; // Marketplace name (e.g., "litellm")
  owner: MarketplaceOwner;
  plugins: MarketplacePluginEntry[];
}

// Marketplace management types
// Hand-written (not generated from schema.d.ts): the /claude-code/marketplaces admin endpoints
// are not yet reflected in the OpenAPI spec this dashboard was built against.
export interface MarketplaceSource {
  id: string;
  name: string;
  display_name?: string;
  source_type: string;
  source_ref: string;
  branch?: string;
  enabled: boolean;
  sync_status: string;
  sync_error?: string;
  last_synced_at?: string;
  plugin_count: number;
  skipped_count: number;
  created_at: string;
  updated_at: string;
}

export interface RegisterMarketplaceRequest {
  source: string;
  name?: string;
}

export interface RegisterMarketplaceResponse {
  status: string;
  marketplace: MarketplaceSource;
}

export interface ListMarketplacesResponse {
  marketplaces: MarketplaceSource[];
  count: number;
}

export interface DeleteMarketplaceResponse {
  status: string;
  message: string;
}

// UI-specific types
export interface CategoryTab {
  key: string;
  label: string;
  count: number;
}
