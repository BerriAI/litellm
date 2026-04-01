import { describe, expect, it } from "vitest";
import {
  formatInstallCommand,
  extractCategories,
  validatePluginName,
  getSourceDisplayText,
  getSourceLink,
  getCategoryBadgeColor,
  formatDateString,
  truncateText,
  filterPluginsBySearch,
  filterPluginsByCategory,
  isValidSemanticVersion,
  isValidEmail,
  isValidUrl,
  parseKeywords,
  formatKeywords,
} from "./helpers";
import { MarketplacePluginEntry, PluginSource } from "./types";

describe("formatInstallCommand", () => {
  it("formats github source with repo", () => {
    const plugin = { name: "my-plugin", source: { source: "github" as const, repo: "org/repo" } };
    expect(formatInstallCommand(plugin)).toBe("/plugin marketplace add org/repo");
  });

  it("formats url source", () => {
    const plugin = { name: "my-plugin", source: { source: "url" as const, url: "https://example.com/plugin" } };
    expect(formatInstallCommand(plugin)).toBe("/plugin marketplace add https://example.com/plugin");
  });

  it("falls back to plugin name when no repo or url", () => {
    const plugin = { name: "my-plugin", source: { source: "github" as const } };
    expect(formatInstallCommand(plugin)).toBe("/plugin marketplace add my-plugin");
  });
});

describe("extractCategories", () => {
  it("returns All and Other for empty list", () => {
    expect(extractCategories([])).toEqual(["All", "Other"]);
  });

  it("extracts and sorts unique categories", () => {
    const plugins = [
      { category: "Development" },
      { category: "Analytics" },
      { category: "Development" },
    ];
    expect(extractCategories(plugins)).toEqual(["All", "Analytics", "Development", "Other"]);
  });

  it("ignores empty/whitespace categories", () => {
    const plugins = [{ category: "" }, { category: "  " }, { category: "Tools" }];
    expect(extractCategories(plugins)).toEqual(["All", "Tools", "Other"]);
  });

  it("handles undefined category", () => {
    const plugins = [{ category: undefined }, { category: "Security" }];
    expect(extractCategories(plugins)).toEqual(["All", "Security", "Other"]);
  });
});

describe("validatePluginName", () => {
  it("accepts valid kebab-case names", () => {
    expect(validatePluginName("my-plugin")).toBe(true);
    expect(validatePluginName("plugin123")).toBe(true);
    expect(validatePluginName("a-b-c")).toBe(true);
  });

  it("rejects names with uppercase", () => {
    expect(validatePluginName("MyPlugin")).toBe(false);
  });

  it("rejects names with spaces", () => {
    expect(validatePluginName("my plugin")).toBe(false);
  });

  it("rejects empty/whitespace names", () => {
    expect(validatePluginName("")).toBe(false);
    expect(validatePluginName("  ")).toBe(false);
  });

  it("rejects names with special characters", () => {
    expect(validatePluginName("my_plugin")).toBe(false);
    expect(validatePluginName("my.plugin")).toBe(false);
  });
});

describe("getSourceDisplayText", () => {
  it("shows github repo", () => {
    expect(getSourceDisplayText({ source: "github", repo: "org/repo" })).toBe("GitHub: org/repo");
  });

  it("shows url", () => {
    expect(getSourceDisplayText({ source: "url", url: "https://example.com" })).toBe("https://example.com");
  });

  it("returns unknown for missing data", () => {
    expect(getSourceDisplayText({ source: "github" })).toBe("Unknown source");
  });
});

describe("getSourceLink", () => {
  it("returns github link for github source", () => {
    expect(getSourceLink({ source: "github", repo: "org/repo" })).toBe("https://github.com/org/repo");
  });

  it("returns url for url source", () => {
    expect(getSourceLink({ source: "url", url: "https://example.com" })).toBe("https://example.com");
  });

  it("returns null when no repo or url", () => {
    expect(getSourceLink({ source: "github" })).toBeNull();
  });
});

describe("getCategoryBadgeColor", () => {
  it("returns blue for development categories", () => {
    expect(getCategoryBadgeColor("Development")).toBe("blue");
    expect(getCategoryBadgeColor("dev-tools")).toBe("blue");
  });

  it("returns green for productivity categories", () => {
    expect(getCategoryBadgeColor("Productivity")).toBe("green");
    expect(getCategoryBadgeColor("Workflow")).toBe("green");
  });

  it("returns purple for learning categories", () => {
    expect(getCategoryBadgeColor("Learning")).toBe("purple");
    expect(getCategoryBadgeColor("Education")).toBe("purple");
  });

  it("returns red for security categories", () => {
    expect(getCategoryBadgeColor("Security")).toBe("red");
    expect(getCategoryBadgeColor("Safety")).toBe("red");
  });

  it("returns orange for data categories", () => {
    expect(getCategoryBadgeColor("Data")).toBe("orange");
    expect(getCategoryBadgeColor("Analytics")).toBe("orange");
  });

  it("returns yellow for integration categories", () => {
    expect(getCategoryBadgeColor("Integration")).toBe("yellow");
    expect(getCategoryBadgeColor("API")).toBe("yellow");
  });

  it("returns gray for unknown or undefined categories", () => {
    expect(getCategoryBadgeColor("Unknown")).toBe("gray");
    expect(getCategoryBadgeColor(undefined)).toBe("gray");
  });
});

describe("formatDateString", () => {
  it("formats valid date strings", () => {
    const result = formatDateString("2024-01-15T12:00:00Z");
    expect(result).toContain("2024");
    expect(result).toContain("Jan");
    expect(result).toContain("15");
  });

  it("returns N/A for undefined", () => {
    expect(formatDateString(undefined)).toBe("N/A");
  });

  it("returns N/A for empty string", () => {
    expect(formatDateString("")).toBe("N/A");
  });
});

describe("truncateText", () => {
  it("returns text unchanged if shorter than max", () => {
    expect(truncateText("hello", 10)).toBe("hello");
  });

  it("truncates and adds ellipsis", () => {
    expect(truncateText("hello world", 5)).toBe("hello...");
  });

  it("handles exact length", () => {
    expect(truncateText("hello", 5)).toBe("hello");
  });

  it("handles empty text", () => {
    expect(truncateText("", 5)).toBe("");
  });
});

describe("filterPluginsBySearch", () => {
  const plugins: MarketplacePluginEntry[] = [
    {
      name: "code-formatter",
      source: { source: "github", repo: "org/formatter" },
      description: "Formats code nicely",
      keywords: ["format", "lint"],
    },
    {
      name: "data-viewer",
      source: { source: "github", repo: "org/viewer" },
      description: "View data",
      keywords: ["analytics"],
    },
  ];

  it("returns all plugins for empty search", () => {
    expect(filterPluginsBySearch(plugins, "")).toEqual(plugins);
    expect(filterPluginsBySearch(plugins, "  ")).toEqual(plugins);
  });

  it("matches by name", () => {
    expect(filterPluginsBySearch(plugins, "formatter")).toHaveLength(1);
    expect(filterPluginsBySearch(plugins, "formatter")[0].name).toBe("code-formatter");
  });

  it("matches by description", () => {
    expect(filterPluginsBySearch(plugins, "nicely")).toHaveLength(1);
  });

  it("matches by keyword", () => {
    expect(filterPluginsBySearch(plugins, "analytics")).toHaveLength(1);
    expect(filterPluginsBySearch(plugins, "analytics")[0].name).toBe("data-viewer");
  });

  it("is case insensitive", () => {
    expect(filterPluginsBySearch(plugins, "FORMATTER")).toHaveLength(1);
  });
});

describe("filterPluginsByCategory", () => {
  const plugins: MarketplacePluginEntry[] = [
    { name: "a", source: { source: "github" }, category: "Dev" },
    { name: "b", source: { source: "github" }, category: "Security" },
    { name: "c", source: { source: "github" }, category: "" },
    { name: "d", source: { source: "github" } },
  ];

  it("returns all plugins for 'All'", () => {
    expect(filterPluginsByCategory(plugins, "All")).toEqual(plugins);
  });

  it("returns uncategorized plugins for 'Other'", () => {
    const result = filterPluginsByCategory(plugins, "Other");
    expect(result).toHaveLength(2);
    expect(result.map((p) => p.name)).toEqual(["c", "d"]);
  });

  it("filters by specific category", () => {
    const result = filterPluginsByCategory(plugins, "Dev");
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe("a");
  });
});

describe("isValidSemanticVersion", () => {
  it("accepts valid semver", () => {
    expect(isValidSemanticVersion("1.0.0")).toBe(true);
    expect(isValidSemanticVersion("0.1.0-alpha")).toBe(true);
    expect(isValidSemanticVersion("2.3.4+build.1")).toBe(true);
  });

  it("rejects invalid semver", () => {
    expect(isValidSemanticVersion("1.0")).toBe(false);
    expect(isValidSemanticVersion("abc")).toBe(false);
  });

  it("returns true for undefined (optional)", () => {
    expect(isValidSemanticVersion(undefined)).toBe(true);
  });
});

describe("isValidEmail", () => {
  it("accepts valid emails", () => {
    expect(isValidEmail("user@example.com")).toBe(true);
  });

  it("rejects invalid emails", () => {
    expect(isValidEmail("not-an-email")).toBe(false);
    expect(isValidEmail("@example.com")).toBe(false);
  });

  it("returns true for undefined (optional)", () => {
    expect(isValidEmail(undefined)).toBe(true);
  });
});

describe("isValidUrl", () => {
  it("accepts valid urls", () => {
    expect(isValidUrl("https://example.com")).toBe(true);
    expect(isValidUrl("http://localhost:3000")).toBe(true);
  });

  it("rejects invalid urls", () => {
    expect(isValidUrl("not a url")).toBe(false);
  });

  it("returns true for undefined (optional)", () => {
    expect(isValidUrl(undefined)).toBe(true);
  });
});

describe("parseKeywords", () => {
  it("splits comma-separated keywords", () => {
    expect(parseKeywords("a, b, c")).toEqual(["a", "b", "c"]);
  });

  it("trims whitespace", () => {
    expect(parseKeywords("  foo ,  bar  ")).toEqual(["foo", "bar"]);
  });

  it("filters empty entries", () => {
    expect(parseKeywords("a,,b,")).toEqual(["a", "b"]);
  });

  it("returns empty array for empty string", () => {
    expect(parseKeywords("")).toEqual([]);
    expect(parseKeywords("  ")).toEqual([]);
  });
});

describe("formatKeywords", () => {
  it("joins keywords with comma and space", () => {
    expect(formatKeywords(["a", "b", "c"])).toBe("a, b, c");
  });

  it("returns empty string for empty/undefined array", () => {
    expect(formatKeywords([])).toBe("");
    expect(formatKeywords(undefined)).toBe("");
  });
});
