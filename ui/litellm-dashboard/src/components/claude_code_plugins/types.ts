/**
 * TypeScript types for Claude Code Marketplace
 * Matches backend API types from /litellm/types/proxy/claude_code_endpoints.py
 */

export interface PluginSource {
  source: "github" | "url";
  repo?: string;  // Format: "org/repo" for GitHub
  url?: string;   // Full URL for other sources
}

export interface PluginAuthor {
  name: string;
  email?: string;
}

export interface Plugin {
  id: string;
  name: string;  // kebab-case
  version?: string;  // semantic version
  description?: string;
  source: PluginSource;
  author?: PluginAuthor;
  homepage?: string;
  keywords?: string[];
  category?: string;
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
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
}

export interface ListPluginsResponse {
  plugins: PluginListItem[];
  count: number;
}

export interface RegisterPluginRequest {
  name: string;
  source: PluginSource;
  version?: string;
  description?: string;
  author?: PluginAuthor;
  homepage?: string;
  keywords?: string[];
  category?: string;
}

export interface RegisterPluginResponse {
  plugin: Plugin;
  action: "created" | "updated";
  message: string;
}

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
  name: string;  // Marketplace name (e.g., "litellm")
  owner: MarketplaceOwner;
  plugins: MarketplacePluginEntry[];
}

// UI-specific types
export interface CategoryTab {
  key: string;
  label: string;
  count: number;
}

export interface PluginFormData {
  name: string;
  sourceType: "github" | "url";
  repo: string;
  url: string;
  version: string;
  description: string;
  authorName: string;
  authorEmail: string;
  homepage: string;
  category: string;
  keywords: string;  // Comma-separated string, will be split into array
}
