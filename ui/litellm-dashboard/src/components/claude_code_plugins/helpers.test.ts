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
  parseSkillSource,
  isValidSubPath,
} from "./helpers";
import { MarketplacePluginEntry } from "./types";

describe("formatInstallCommand", () => {
  it("produces a /plugin install command scoped to the litellm marketplace", () => {
    expect(formatInstallCommand({ name: "my-plugin" })).toBe("/plugin install my-plugin@litellm");
  });

  it("uses the plugin name as the identifier", () => {
    expect(formatInstallCommand({ name: "code-review" })).toBe("/plugin install code-review@litellm");
  });
});

describe("extractCategories", () => {
  it("returns All and Other for empty list", () => {
    expect(extractCategories([])).toEqual(["All", "Other"]);
  });

  it("extracts and sorts unique categories", () => {
    const plugins = [{ category: "Development" }, { category: "Analytics" }, { category: "Development" }];
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

  it("shows git-subdir as url @ path for a github subdir", () => {
    expect(getSourceDisplayText({ source: "git-subdir", url: "https://github.com/org/repo", path: "plugins/x" })).toBe(
      "https://github.com/org/repo @ plugins/x",
    );
  });

  it("shows git-subdir as url @ path for a gitlab subdir", () => {
    expect(getSourceDisplayText({ source: "git-subdir", url: "https://gitlab.com/org/repo", path: "sub/dir" })).toBe(
      "https://gitlab.com/org/repo @ sub/dir",
    );
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

  it("returns the repo url for a github git-subdir source", () => {
    expect(getSourceLink({ source: "git-subdir", url: "https://github.com/org/repo", path: "plugins/x" })).toBe(
      "https://github.com/org/repo",
    );
  });

  it("returns the repo url for a gitlab git-subdir source", () => {
    expect(getSourceLink({ source: "git-subdir", url: "https://gitlab.com/org/repo", path: "sub/dir" })).toBe(
      "https://gitlab.com/org/repo",
    );
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

describe("parseSkillSource", () => {
  it("parses a plain github repo", () => {
    expect(parseSkillSource("github.com/org/repo")?.parsed).toEqual({ source: "github", repo: "org/repo" });
  });

  it("strips a .git suffix from the github repo shorthand", () => {
    expect(parseSkillSource("https://github.com/org/repo.git")?.parsed).toEqual({
      source: "github",
      repo: "org/repo",
    });
  });

  it("parses a github tree URL into a git-subdir", () => {
    expect(parseSkillSource("github.com/org/repo/tree/main/plugins/x")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://github.com/org/repo",
      path: "plugins/x",
    });
  });

  it("drops a trailing file segment from a github blob URL", () => {
    expect(parseSkillSource("github.com/org/repo/blob/main/x/SKILL.md")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://github.com/org/repo",
      path: "x",
    });
  });

  it("combines a github repo with an explicit subfolder", () => {
    expect(parseSkillSource("github.com/org/repo", "plugins/x")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://github.com/org/repo",
      path: "plugins/x",
    });
  });

  it("treats a gitlab repo as a raw url source", () => {
    expect(parseSkillSource("gitlab.com/org/repo")?.parsed).toEqual({
      source: "url",
      url: "https://gitlab.com/org/repo",
    });
  });

  it("keeps the .git suffix on raw urls", () => {
    expect(parseSkillSource("https://gitlab.com/org/repo.git")?.parsed).toEqual({
      source: "url",
      url: "https://gitlab.com/org/repo.git",
    });
  });

  it("combines a gitlab repo with an explicit subfolder", () => {
    expect(parseSkillSource("gitlab.com/org/repo", "plugins/x")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://gitlab.com/org/repo",
      path: "plugins/x",
    });
  });

  it("combines a self-hosted host with an explicit subfolder", () => {
    expect(parseSkillSource("https://git.acme.com/team/repo", "sub/dir")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://git.acme.com/team/repo",
      path: "sub/dir",
    });
  });

  it("lets a github URL-encoded subdir win over an also-provided subfolder", () => {
    expect(parseSkillSource("github.com/org/repo/tree/main/plugins/x", "ignored/path")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://github.com/org/repo",
      path: "plugins/x",
    });
  });

  it("rejects traversal, absolute, and double-slash subfolders", () => {
    expect(parseSkillSource("gitlab.com/org/repo", "../etc")).toBeNull();
    expect(parseSkillSource("gitlab.com/org/repo", "/abs")).toBeNull();
    expect(parseSkillSource("gitlab.com/org/repo", "a//b")).toBeNull();
  });

  it("returns null for empty and garbage input", () => {
    expect(parseSkillSource("")).toBeNull();
    expect(parseSkillSource("   ")).toBeNull();
    expect(parseSkillSource("not a url")).toBeNull();
  });

  it("suggests a kebab-friendly name from the last path segment", () => {
    expect(parseSkillSource("github.com/org/my-awesome-skill")?.suggestedName).toBe("my-awesome-skill");
    expect(parseSkillSource("github.com/org/repo/tree/main/plugins/cool-skill")?.suggestedName).toBe("cool-skill");
    expect(parseSkillSource("gitlab.com/org/repo", "plugins/x")?.suggestedName).toBe("x");
  });

  it("rejects a bad explicit subfolder for a github repo", () => {
    expect(parseSkillSource("github.com/org/repo", "../etc")).toBeNull();
    expect(parseSkillSource("github.com/org/repo", "/abs")).toBeNull();
    expect(parseSkillSource("github.com/org/repo", "a//b")).toBeNull();
  });

  it("treats a blob URL pointing at a root file as the plain repo", () => {
    expect(parseSkillSource("github.com/org/repo/blob/main/SKILL.md")?.parsed).toEqual({
      source: "github",
      repo: "org/repo",
    });
  });

  it("strips query strings and fragments before parsing", () => {
    expect(parseSkillSource("github.com/org/repo?tab=readme")?.parsed).toEqual({ source: "github", repo: "org/repo" });
    expect(parseSkillSource("github.com/org/repo#section")?.parsed).toEqual({ source: "github", repo: "org/repo" });
  });

  it("rejects a tree URL whose folder has a space or percent-encoded segment", () => {
    expect(parseSkillSource("github.com/org/repo/tree/main/a b")).toBeNull();
    expect(parseSkillSource("github.com/org/repo/tree/main/a%20b")).toBeNull();
  });

  it("routes uppercase and www github hosts through the github shorthand", () => {
    expect(parseSkillSource("GitHub.com/org/repo/tree/main/x")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://github.com/org/repo",
      path: "x",
    });
    expect(parseSkillSource("www.github.com/org/repo")?.parsed).toEqual({ source: "github", repo: "org/repo" });
  });

  it("keeps a dotted folder name as the subdir path", () => {
    expect(parseSkillSource("github.com/org/repo/blob/main/my.skill")?.parsed).toEqual({
      source: "git-subdir",
      url: "https://github.com/org/repo",
      path: "my.skill",
    });
  });

  it("falls back to the repo for a tree URL with a branch but no folder", () => {
    expect(parseSkillSource("github.com/org/repo/tree/main")?.parsed).toEqual({ source: "github", repo: "org/repo" });
  });

  it("kebab-cases the suggested name from a mixed-case repo", () => {
    expect(parseSkillSource("github.com/Org/My_Repo")?.suggestedName).toBe("my-repo");
  });

  it("rejects a bare host or single-segment raw git url", () => {
    expect(parseSkillSource("gitlab.com")).toBeNull();
    expect(parseSkillSource("gitlab.com/org")).toBeNull();
  });
});

// Skill sources are served on the unauthenticated public feeds and cloned by clients, so the
// parser must never publish an insecure, credentialed, internal, or malformed clone URL.
describe("parseSkillSource — security boundary", () => {
  it("rejects non-https schemes", () => {
    for (const url of [
      "http://gitlab.com/org/repo",
      "HTTP://gitlab.com/org/repo",
      "ssh://gitlab.com/org/repo",
      "git://gitlab.com/org/repo",
      "ftp://gitlab.com/org/repo",
      "file:///etc/passwd",
      "javascript:alert(1)",
      "data:text/plain,hi",
      "//gitlab.com/org/repo",
    ]) {
      expect(parseSkillSource(url)).toBeNull();
    }
  });

  it("rejects URLs with embedded credentials", () => {
    expect(parseSkillSource("https://user:token@gitlab.com/org/repo")).toBeNull();
    expect(parseSkillSource("https://user@gitlab.com/org/repo")).toBeNull();
    // userinfo confusion: the real host is evil.com, not github.com
    expect(parseSkillSource("https://github.com@evil.com/org/repo")).toBeNull();
  });

  it("rejects IP-literal hosts (loopback, private, metadata, obfuscated, IPv6)", () => {
    for (const url of [
      "https://127.0.0.1/org/repo",
      "https://10.0.0.5/org/repo",
      "https://169.254.169.254/org/repo",
      "https://2130706433/org/repo",
      "https://[::ffff:127.0.0.1]/org/repo",
    ]) {
      expect(parseSkillSource(url)).toBeNull();
    }
  });

  it("does not grant GitHub shorthand to a look-alike host", () => {
    expect(parseSkillSource("https://github.com.evil.com/org/repo")?.parsed).toEqual({
      source: "url",
      url: "https://github.com.evil.com/org/repo",
    });
  });

  it("rejects GitHub org/repo segments with illegal characters", () => {
    expect(parseSkillSource("github.com/o@x/repo")).toBeNull();
    expect(parseSkillSource("github.com/org/..%2f..%2fx")).toBeNull();
  });
});

describe("isValidSubPath", () => {
  it("accepts relative segment paths", () => {
    expect(isValidSubPath("plugins/x")).toBe(true);
    expect(isValidSubPath("sub/dir")).toBe(true);
    expect(isValidSubPath("a.b-c_d")).toBe(true);
    expect(isValidSubPath("plugins/x/")).toBe(true);
  });

  it("rejects empty, traversal, absolute, and double-slash paths", () => {
    expect(isValidSubPath("")).toBe(false);
    expect(isValidSubPath("../etc")).toBe(false);
    expect(isValidSubPath("/abs")).toBe(false);
    expect(isValidSubPath("a//b")).toBe(false);
  });
});
