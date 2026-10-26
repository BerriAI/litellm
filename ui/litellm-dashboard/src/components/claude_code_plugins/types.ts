/**
 * TypeScript types for Claude Code Marketplace
 * API request/response shapes are synced from the generated OpenAPI types in @/lib/http/schema.
 */

import type { components } from "@/lib/http/schema";

export interface PluginSource {
  source: "github" | "url" | "git-subdir" | "archive";
  repo?: string; // Format: "org/repo" for GitHub
  url?: string; // Full URL for other sources
  path?: string; // Subdirectory path for git-subdir
  sha256?: string;
}

export type PluginAuthor = components["schemas"]["PluginAuthor"];

export type Plugin = components["schemas"]["PluginListItem"];
export type PluginListItem = Plugin;
export type ListPluginsResponse = components["schemas"]["ListPluginsResponse"];

// Request envelope synced from the OpenAPI spec, with `source` narrowed to our PluginSource
// union and `version` kept optional (the backend supplies its default).
export type SkillRegisterRequest = Omit<components["schemas"]["RegisterPluginRequest"], "source" | "version"> & {
  source: PluginSource;
  version?: string;
};

// Public marketplace types
export type MarketplacePluginEntry = Pick<Plugin, "name" | "source"> &
  Partial<Pick<Plugin, "version" | "description" | "author" | "homepage" | "keywords" | "category">>;

export interface MarketplaceOwner {
  name: string;
  email?: string;
}

export interface MarketplaceResponse {
  name: string;
  owner: MarketplaceOwner;
  plugins: MarketplacePluginEntry[];
}

// UI-specific types
export interface CategoryTab {
  key: string;
  label: string;
  count: number;
}
