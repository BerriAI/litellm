/**
 * Helper utilities for Claude Code Marketplace
 */

import { PluginSource, MarketplacePluginEntry } from "./types";

export interface SkillSourcePreview {
  parsed: PluginSource;
  label: string;
  suggestedName: string;
}

export const SUBDIR_PATH_REGEX = /^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/;

export const normalizeSubPath = (subPath: string): string => subPath.trim().replace(/\/+$/, "");

export const isValidSubPath = (subPath: string): boolean => {
  const normalized = normalizeSubPath(subPath);
  return normalized !== "" && SUBDIR_PATH_REGEX.test(normalized);
};

const GITHUB_HOST = "github.com";

const SKILL_FILE_EXTENSION_REGEX = /\.(md|markdown|txt|json|ya?ml|toml)$/i;

// WHATWG normalizes obfuscated IPv4 (e.g. 2130706433, 0x7f.0.0.1) to dotted-decimal, so this
// catches every IPv4 form; bracketed IPv6 is rejected separately.
const IPV4_HOST_REGEX = /^\d{1,3}(\.\d{1,3}){3}$/;

const GITHUB_ORG_REGEX = /^[A-Za-z0-9-]+$/;
const GITHUB_REPO_REGEX = /^[A-Za-z0-9._-]+$/;

const buildRepoUrl = (url: URL): string => `${url.protocol}//${url.host}${url.pathname.replace(/\/+$/, "")}`;

const pathSegments = (url: URL): string[] => url.pathname.split("/").filter((seg) => seg !== "");

/**
 * Validate and normalize a repository URL into a parsed URL, or null. Enforces https (rejects
 * http/ssh/git/etc.), rejects embedded credentials, and requires a dotted host, so the public
 * skill feeds never serve an insecure or credentialed clone URL. Everything downstream parses
 * this normalized object rather than the raw string.
 */
const parseRepoUrl = (raw: string): URL | null => {
  const trimmed = raw.trim();
  if (trimmed === "" || trimmed.startsWith("//")) {
    return null;
  }
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    return null;
  }
  if (
    url.protocol !== "https:" ||
    url.username !== "" ||
    url.password !== "" ||
    !url.hostname.includes(".") ||
    url.hostname.startsWith("[") ||
    IPV4_HOST_REGEX.test(url.hostname)
  ) {
    return null;
  }
  return url;
};

const lastSegment = (path: string): string => {
  const segments = path.split("/").filter((seg) => seg !== "");
  return segments[segments.length - 1] ?? "";
};

const toKebabCase = (value: string): string =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");

const parseGitHubSource = (url: URL, subPath?: string): SkillSourcePreview | null => {
  const parts = pathSegments(url);
  if (parts.length < 2) {
    return null;
  }

  const org = parts[0];
  const repoBase = parts[1].replace(/\.git$/, "");
  if (!GITHUB_ORG_REGEX.test(org) || !GITHUB_REPO_REGEX.test(repoBase)) {
    return null;
  }
  const repoFull = `${org}/${repoBase}`;
  const repoUrl = `https://github.com/${repoFull}`;
  const repoPreview: SkillSourcePreview = {
    parsed: { source: "github", repo: repoFull },
    label: `GitHub repo — ${repoFull}`,
    suggestedName: toKebabCase(repoBase),
  };

  const isTreeOrBlob = parts.length >= 4 && (parts[2] === "tree" || parts[2] === "blob");
  if (isTreeOrBlob) {
    const pathParts = parts.slice(4);
    const last = lastSegment(pathParts.join("/"));
    const effective = SKILL_FILE_EXTENSION_REGEX.test(last) ? pathParts.slice(0, -1) : pathParts;
    if (effective.length === 0) {
      return repoPreview;
    }
    const path = normalizeSubPath(effective.join("/"));
    if (!SUBDIR_PATH_REGEX.test(path)) {
      return null;
    }
    return {
      parsed: { source: "git-subdir", url: repoUrl, path },
      label: `GitHub subdir — ${repoFull} @ ${path}`,
      suggestedName: toKebabCase(lastSegment(path)),
    };
  }

  if (parts.length !== 2) {
    return null;
  }

  const normalized = normalizeSubPath(subPath ?? "");
  if (normalized !== "") {
    if (!SUBDIR_PATH_REGEX.test(normalized)) {
      return null;
    }
    return {
      parsed: { source: "git-subdir", url: repoUrl, path: normalized },
      label: `GitHub subdir — ${repoFull} @ ${normalized}`,
      suggestedName: toKebabCase(lastSegment(normalized)),
    };
  }

  return repoPreview;
};

const parseRawGitSource = (url: URL, subPath?: string): SkillSourcePreview | null => {
  if (pathSegments(url).length < 2) {
    return null;
  }

  const repoUrl = buildRepoUrl(url);

  const normalized = normalizeSubPath(subPath ?? "");
  if (normalized !== "") {
    if (!SUBDIR_PATH_REGEX.test(normalized)) {
      return null;
    }
    return {
      parsed: { source: "git-subdir", url: repoUrl, path: normalized },
      label: `Git subdir — ${repoUrl} @ ${normalized}`,
      suggestedName: toKebabCase(lastSegment(normalized)),
    };
  }

  return {
    parsed: { source: "url", url: repoUrl },
    label: `Git repo — ${repoUrl}`,
    suggestedName: toKebabCase(lastSegment(url.pathname).replace(/\.git$/, "")),
  };
};

/**
 * Parse any git-accessible repository URL into a registerable skill source.
 * GitHub URLs keep their `github`/`git-subdir` shorthand; every other host is
 * treated as a raw repo URL, with an optional subfolder turning it into git-subdir.
 */
export const parseSkillSource = (rawUrl: string, subPath?: string): SkillSourcePreview | null => {
  const url = parseRepoUrl(rawUrl);
  if (!url) {
    return null;
  }
  if (url.hostname.replace(/^www\./, "") === GITHUB_HOST) {
    return parseGitHubSource(url, subPath);
  }
  return parseRawGitSource(url, subPath);
};

/**
 * Marketplace name the proxy publishes in marketplace.json. Claude Code keys its
 * `extraKnownMarketplaces` config and `@<marketplace>` install suffix off this value.
 */
export const MARKETPLACE_NAME = "litellm";

/**
 * Path the proxy serves the Claude Code marketplace manifest from
 */
export const MARKETPLACE_MANIFEST_PATH = "/claude-code/marketplace.json";

/**
 * Generate the Claude Code install command for a registered skill
 * Format: /plugin install <name>@litellm
 */
export const formatInstallCommand = (plugin: { name: string }): string =>
  `/plugin install ${plugin.name}@${MARKETPLACE_NAME}`;

export interface MarketplaceSettings {
  extraKnownMarketplaces: Record<string, { source: { source: "url"; url: string } }>;
}

/**
 * Build the `~/.claude/settings.json` snippet that registers the proxy as a Claude Code
 * marketplace. Claude Code requires `source` to be a nested object, and the marketplace key
 * must match the name the proxy returns in marketplace.json.
 */
export const buildMarketplaceSettings = (marketplaceUrl: string): MarketplaceSettings => ({
  extraKnownMarketplaces: {
    [MARKETPLACE_NAME]: {
      source: { source: "url", url: marketplaceUrl },
    },
  },
});

/**
 * Extract unique categories from plugins list
 * Returns array with "All" first, then sorted categories, then "Other"
 */
export const extractCategories = (plugins: Array<{ category?: string }>): string[] => {
  const categories = new Set<string>();

  plugins.forEach((p) => {
    if (p.category && p.category.trim() !== "") {
      categories.add(p.category);
    }
  });

  const sortedCategories = Array.from(categories).sort();

  // Return: All, sorted categories, Other
  return ["All", ...sortedCategories, "Other"];
};

/**
 * Validate plugin name format (kebab-case)
 * Must be lowercase letters, numbers, and hyphens only
 */
export const validatePluginName = (name: string): boolean => {
  if (!name || name.trim() === "") {
    return false;
  }
  // Regex: lowercase letters, numbers, hyphens
  return /^[a-z0-9-]+$/.test(name);
};

/**
 * Get human-readable source display text
 */
export const getSourceDisplayText = (source: PluginSource): string => {
  if (source.source === "github" && source.repo) {
    return `GitHub: ${source.repo}`;
  }
  if (source.source === "git-subdir" && source.url && source.path) {
    return `${source.url} @ ${source.path}`;
  }
  if (source.source === "url" && source.url) {
    return source.url;
  }
  return "Unknown source";
};

/**
 * Get clickable link for plugin source
 */
export const getSourceLink = (source: PluginSource): string | null => {
  if (source.source === "github" && source.repo) {
    return `https://github.com/${source.repo}`;
  }
  if ((source.source === "url" || source.source === "git-subdir") && source.url) {
    return source.url;
  }
  return null;
};

/**
 * Get badge color based on category
 */
export const getCategoryBadgeColor = (
  category?: string,
): "blue" | "green" | "purple" | "red" | "orange" | "yellow" | "gray" => {
  if (!category) {
    return "gray";
  }

  const categoryLower = category.toLowerCase();

  if (categoryLower.includes("development") || categoryLower.includes("dev")) {
    return "blue";
  } else if (categoryLower.includes("productivity") || categoryLower.includes("workflow")) {
    return "green";
  } else if (categoryLower.includes("learning") || categoryLower.includes("education")) {
    return "purple";
  } else if (categoryLower.includes("security") || categoryLower.includes("safety")) {
    return "red";
  } else if (categoryLower.includes("data") || categoryLower.includes("analytics")) {
    return "orange";
  } else if (categoryLower.includes("integration") || categoryLower.includes("api")) {
    return "yellow";
  }

  return "gray";
};

/**
 * Format date to readable string
 */
export const formatDateString = (dateString?: string): string => {
  if (!dateString) {
    return "N/A";
  }

  try {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch (error) {
    return "Invalid date";
  }
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (!text || text.length <= maxLength) {
    return text;
  }
  return text.substring(0, maxLength) + "...";
};

/**
 * Filter plugins by search term
 * Searches in: name, description, keywords
 */
export const filterPluginsBySearch = (
  plugins: MarketplacePluginEntry[],
  searchTerm: string,
): MarketplacePluginEntry[] => {
  if (!searchTerm || searchTerm.trim() === "") {
    return plugins;
  }

  const term = searchTerm.toLowerCase().trim();

  return plugins.filter((plugin) => {
    const nameMatch = plugin.name.toLowerCase().includes(term);
    const descriptionMatch = plugin.description?.toLowerCase().includes(term) || false;
    const keywordsMatch = plugin.keywords?.some((keyword) => keyword.toLowerCase().includes(term)) || false;

    return nameMatch || descriptionMatch || keywordsMatch;
  });
};

/**
 * Filter plugins by category
 */
export const filterPluginsByCategory = (
  plugins: MarketplacePluginEntry[],
  category: string,
): MarketplacePluginEntry[] => {
  if (category === "All") {
    return plugins;
  }

  if (category === "Other") {
    return plugins.filter((p) => !p.category || p.category.trim() === "");
  }

  return plugins.filter((p) => p.category === category);
};

/**
 * Validate semantic version format (basic check)
 */
export const isValidSemanticVersion = (version?: string): boolean => {
  if (!version) {
    return true; // Version is optional
  }

  // Basic semver check: X.Y.Z
  const semverRegex = /^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/;
  return semverRegex.test(version);
};

/**
 * Validate email format
 */
export const isValidEmail = (email?: string): boolean => {
  if (!email) {
    return true; // Email is optional
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

/**
 * Validate URL format
 */
export const isValidUrl = (url?: string): boolean => {
  if (!url) {
    return true; // URL is optional
  }

  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

/**
 * Parse keywords from comma-separated string
 */
export const parseKeywords = (keywordsString: string): string[] => {
  if (!keywordsString || keywordsString.trim() === "") {
    return [];
  }

  return keywordsString
    .split(",")
    .map((kw) => kw.trim())
    .filter((kw) => kw !== "");
};

/**
 * Format keywords array to comma-separated string
 */
export const formatKeywords = (keywords?: string[]): string => {
  if (!keywords || keywords.length === 0) {
    return "";
  }

  return keywords.join(", ");
};
