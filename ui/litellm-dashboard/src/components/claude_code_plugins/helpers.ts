/**
 * Helper utilities for Claude Code Marketplace
 */

import { PluginSource, MarketplacePluginEntry } from "./types";

/**
 * Generate install command for Claude Code CLI
 * Format: /plugin marketplace add org/repo OR /plugin marketplace add url
 */
export const formatInstallCommand = (plugin: {
  name: string;
  source: PluginSource;
}): string => {
  if (plugin.source.source === "github" && plugin.source.repo) {
    return `/plugin marketplace add ${plugin.source.repo}`;
  } else if (plugin.source.source === "url" && plugin.source.url) {
    return `/plugin marketplace add ${plugin.source.url}`;
  }
  // Fallback to plugin name
  return `/plugin marketplace add ${plugin.name}`;
};

/**
 * Extract unique categories from plugins list
 * Returns array with "All" first, then sorted categories, then "Other"
 */
export const extractCategories = (
  plugins: Array<{ category?: string }>
): string[] => {
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
  } else if (source.source === "url" && source.url) {
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
  } else if (source.source === "url" && source.url) {
    return source.url;
  }
  return null;
};

/**
 * Get badge color based on category
 */
export const getCategoryBadgeColor = (
  category?: string
): "blue" | "green" | "purple" | "red" | "orange" | "yellow" | "gray" => {
  if (!category) {
    return "gray";
  }

  const categoryLower = category.toLowerCase();

  if (categoryLower.includes("development") || categoryLower.includes("dev")) {
    return "blue";
  } else if (
    categoryLower.includes("productivity") ||
    categoryLower.includes("workflow")
  ) {
    return "green";
  } else if (
    categoryLower.includes("learning") ||
    categoryLower.includes("education")
  ) {
    return "purple";
  } else if (
    categoryLower.includes("security") ||
    categoryLower.includes("safety")
  ) {
    return "red";
  } else if (
    categoryLower.includes("data") ||
    categoryLower.includes("analytics")
  ) {
    return "orange";
  } else if (
    categoryLower.includes("integration") ||
    categoryLower.includes("api")
  ) {
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
  searchTerm: string
): MarketplacePluginEntry[] => {
  if (!searchTerm || searchTerm.trim() === "") {
    return plugins;
  }

  const term = searchTerm.toLowerCase().trim();

  return plugins.filter((plugin) => {
    const nameMatch = plugin.name.toLowerCase().includes(term);
    const descriptionMatch =
      plugin.description?.toLowerCase().includes(term) || false;
    const keywordsMatch =
      plugin.keywords?.some((keyword) =>
        keyword.toLowerCase().includes(term)
      ) || false;

    return nameMatch || descriptionMatch || keywordsMatch;
  });
};

/**
 * Filter plugins by category
 */
export const filterPluginsByCategory = (
  plugins: MarketplacePluginEntry[],
  category: string
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
