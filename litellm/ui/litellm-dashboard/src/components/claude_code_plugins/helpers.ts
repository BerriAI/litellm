/**
 * Helper utilities for Claude Code Marketplace
 */

import i18n from "@/i18n/i18n";
import type { Plugin, PluginSource, SkillImport } from "./types";

export interface SkillSourcePreview {
  parsed: PluginSource;
  label: string;
  suggestedName: string;
  pathFromUrl?: boolean;
  fileFromUrl?: string;
}

export const SUBDIR_PATH_REGEX = /^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/;

export const normalizeSubPath = (subPath: string): string => subPath.trim().replace(/\/+$/, "");

export const isValidSubPath = (subPath: string): boolean => {
  const normalized = normalizeSubPath(subPath);
  return normalized !== "" && SUBDIR_PATH_REGEX.test(normalized);
};

const GITHUB_HOST = "github.com";
const withoutWww = (host: string): string => host.replace(/^www\./, "");

const ZIP_ARCHIVE_REGEX = /\.zip$/i;

export const SHA256_REGEX = /^[0-9a-fA-F]{64}$/;

export const isValidSha256 = (digest: string): boolean => digest.trim() === "" || SHA256_REGEX.test(digest.trim());

// WHATWG normalizes obfuscated IPv4 (e.g. 2130706433, 0x7f.0.0.1) to dotted-decimal on https, so
// this catches every IPv4 form there; on a non-special scheme like ssh it catches the dotted form
// only. Bracketed IPv6 is rejected separately.
const IPV4_HOST_REGEX = /^\d{1,3}(\.\d{1,3}){3}$/;

const GITHUB_ORG_REGEX = /^[A-Za-z0-9-]+$/;
const GITHUB_REPO_REGEX = /^[A-Za-z0-9._-]+$/;

const BROWSABLE_URL_REGEX = /^https?:\/\//i;
const SSH_SCHEME = "ssh://";
const SSH_SCP_REGEX = /^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i;

const buildRepoUrl = (url: URL): string => `${url.protocol}//${url.host}${url.pathname.replace(/\/+$/, "")}`;

const pathSegments = (url: URL): string[] => url.pathname.split("/").filter((seg) => seg !== "");

const toUrl = (candidate: string): URL | null => {
  try {
    return new URL(candidate);
  } catch {
    return null;
  }
};

/** One host rule for every scheme, so an ssh remote is neither more nor less trusted than its https twin. */
const isSafeHost = (url: URL): boolean =>
  url.hostname.includes(".") && !url.hostname.startsWith("[") && !IPV4_HOST_REGEX.test(url.hostname);

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
  const url = toUrl(withScheme);
  if (!url || url.protocol !== "https:" || url.username !== "" || url.password !== "" || !isSafeHost(url)) {
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

const withGitRef = (preview: SkillSourcePreview | null, ref?: string): SkillSourcePreview | null => {
  const target = ref?.trim();
  if (!preview || !target) return preview;
  return {
    ...preview,
    parsed: {
      ...preview.parsed,
      ...(/^[a-fA-F0-9]{40}$/.test(target) ? { sha: target.toLowerCase() } : { ref: target }),
    },
    label: `${preview.label} (${target})`,
  };
};

interface BrowserLocation {
  ref: string;
  path: string;
  pathFromUrl: boolean;
  fileFromUrl?: string;
}

const browserLocation = (
  parts: string[],
  kind: "tree" | "blob",
  explicitRef?: string,
  subPath?: string,
): BrowserLocation | null => {
  if (!parts.length) return null;
  try {
    const ref = explicitRef?.trim() || decodeURIComponent(parts[0]);
    const refLength = explicitRef?.trim()
      ? parts.findIndex((_, index) => decodeURIComponent(parts.slice(0, index + 1).join("/")) === ref) + 1
      : 1;
    if (!refLength || (kind === "blob" && parts.length === refLength)) return null;
    const path = parts
      .slice(refLength, kind === "blob" ? -1 : undefined)
      .map(decodeURIComponent)
      .join("/");
    const fileFromUrl = kind === "blob" ? decodeURIComponent(parts[parts.length - 1]) : undefined;
    const pathFromUrl = Boolean(path || fileFromUrl);
    return { ref, path: pathFromUrl ? path : normalizeSubPath(subPath ?? ""), pathFromUrl, fileFromUrl };
  } catch {
    return null;
  }
};

const parseGitHubSource = (
  url: URL,
  subPath?: string,
  ref?: string,
  pathContract: "reference" | "portable" = "reference",
): SkillSourcePreview | null => {
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
    label: i18n.t("claudeCodePlugins.sources.githubRepo", { repo: repoFull }),
    suggestedName: toKebabCase(repoBase),
  };

  const kind = parts[2];
  const location = kind === "tree" || kind === "blob" ? browserLocation(parts.slice(3), kind, ref, subPath) : undefined;
  if (location === null || (location === undefined && parts.length !== 2)) {
    return null;
  }
  const path = location?.path ?? normalizeSubPath(subPath ?? "");
  if (pathContract === "reference" && path && !SUBDIR_PATH_REGEX.test(path)) {
    return null;
  }
  const preview: SkillSourcePreview = path
    ? {
        parsed: { source: "git-subdir", url: repoUrl, path },
        label: i18n.t("claudeCodePlugins.sources.githubSubdir", { repo: repoFull, path }),
        suggestedName: toKebabCase(lastSegment(path)),
      }
    : repoPreview;
  return withGitRef(
    {
      ...preview,
      ...(location ? { pathFromUrl: location.pathFromUrl, fileFromUrl: location.fileFromUrl } : {}),
    },
    location?.ref ?? ref,
  );
};

const buildGitSourcePreview = (
  kind: "Git" | "SSH",
  repoUrl: string,
  repoName: string,
  subPath?: string,
): SkillSourcePreview | null => {
  const normalized = normalizeSubPath(subPath ?? "");
  if (normalized !== "") {
    if (!SUBDIR_PATH_REGEX.test(normalized)) {
      return null;
    }
    return {
      parsed: { source: "git-subdir", url: repoUrl, path: normalized },
      label: i18n.t("claudeCodePlugins.sources.gitSubdir", { kind, url: repoUrl, path: normalized }),
      suggestedName: toKebabCase(lastSegment(normalized)),
    };
  }

  return {
    parsed: { source: "url", url: repoUrl },
    label: i18n.t("claudeCodePlugins.sources.gitRepo", { kind, url: repoUrl }),
    suggestedName: toKebabCase(repoName),
  };
};

const parseRawGitSource = (url: URL, subPath?: string): SkillSourcePreview | null => {
  if (pathSegments(url).length < 2) {
    return null;
  }
  const repoName = lastSegment(url.pathname).replace(/\.git$/, "");
  return buildGitSourcePreview("Git", buildRepoUrl(url), repoName, subPath);
};

const parseGitLabBrowserSource = (url: URL, subPath?: string, ref?: string): SkillSourcePreview | null => {
  const parts = pathSegments(url);
  const separator = parts.findIndex((part, index) => part === "-" && ["tree", "blob"].includes(parts[index + 1]));
  const kind = parts[separator + 1];
  if (separator < 2 || (kind !== "tree" && kind !== "blob")) return null;
  const location = browserLocation(parts.slice(separator + 2), kind, ref, subPath);
  if (!location) return null;
  const repoUrl = `${url.protocol}//${withoutWww(url.host)}/${parts.slice(0, separator).join("/")}`;
  const preview = buildGitSourcePreview("Git", repoUrl, parts[separator - 1].replace(/\.git$/, ""), location.path);
  const selected = preview
    ? { ...preview, pathFromUrl: location.pathFromUrl, fileFromUrl: location.fileFromUrl }
    : null;
  return withGitRef(selected, location.ref);
};

/**
 * Parse an scp-style `git@host:org/repo` or `ssh://git@host/org/repo` clone URL, registering it
 * exactly as typed: git treats the `.git` suffix as optional, and forcing one on breaks hosts whose
 * paths are not `org/repo`, like Azure DevOps `v3/...` and CodeCommit `v1/repos/...`. The scp form is
 * rewritten to `ssh://` only to reuse the https host and credential rules, and only a URL that
 * survives that round trip unchanged is accepted, which keeps traversal segments off the feed.
 */
const parseSshSource = (raw: string, subPath?: string): SkillSourcePreview | null => {
  const trimmed = raw.trim();
  const scp = SSH_SCP_REGEX.exec(trimmed);
  const candidate = scp ? `${SSH_SCHEME}${scp[1]}@${scp[2]}/${scp[3]}` : trimmed;
  if (!candidate.toLowerCase().startsWith(SSH_SCHEME)) {
    return null;
  }
  const url = toUrl(candidate);
  if (!url || url.username === "" || url.password !== "" || !isSafeHost(url)) {
    return null;
  }
  const pathStart = candidate.indexOf("/", SSH_SCHEME.length);
  if (pathStart === -1 || url.pathname !== candidate.slice(pathStart) || pathSegments(url).length < 2) {
    return null;
  }
  return buildGitSourcePreview("SSH", trimmed, lastSegment(url.pathname).replace(/\.git$/i, ""), subPath);
};

const parseArchiveSource = (url: URL): SkillSourcePreview => ({
  parsed: { source: "archive", url: url.href },
  label: i18n.t("claudeCodePlugins.sources.zipArchive", { location: `${url.host}${url.pathname}` }),
  suggestedName: toKebabCase(lastSegment(url.pathname).replace(ZIP_ARCHIVE_REGEX, "")),
});

const parseSkillImportUrl = (rawUrl: string): URL | null => {
  const url = parseRepoUrl(rawUrl);
  if (!url) return null;
  const host = withoutWww(url.host);
  return [GITHUB_HOST, "skills.sh"].includes(host)
    ? toUrl(`${url.protocol}//${host}${url.pathname}${url.search}${url.hash}`)
    : null;
};

export const requiresBrowserRefConfirmation = (preview: SkillSourcePreview | null | undefined, ref: string): boolean =>
  Boolean(
    preview?.pathFromUrl &&
      preview.parsed.path &&
      !(ref.trim() || preview.parsed.sha || preview.parsed.ref?.includes("/")),
  );

export const availableSkillUrlModes = (rawUrl: string) => {
  const url = parseSkillImportUrl(rawUrl);
  if (url?.host === "skills.sh") return ["import"] as const;
  return url ? (["register", "import"] as const) : (["register"] as const);
};

interface SkillImportSource {
  selection: Pick<SkillImport, "url" | "skill" | "ref" | "path">;
  preview: SkillSourcePreview | null;
  skillFromUrl?: string;
}

const parseDirectorySkillSource = (url: URL, path?: string, ref?: string): SkillImportSource | null => {
  try {
    const parts = url.pathname
      .replace(/^\/+|\/+$/g, "")
      .split("/")
      .map(decodeURIComponent);
    if (parts.length !== 3 || parts.slice(0, 2).some((part) => !GITHUB_REPO_REGEX.test(part))) return null;
    const skillFromUrl = parts[2];
    if (!skillFromUrl) return null;
    return {
      selection: {
        url: url.href,
        ...(ref?.trim() ? { ref: ref.trim() } : {}),
        ...(path?.trim() ? { path: path.trim() } : {}),
      },
      preview: null,
      skillFromUrl,
    };
  } catch {
    return null;
  }
};

export const parseSkillImportSource = (
  rawUrl: string,
  path?: string,
  ref?: string,
  skill?: string,
): SkillImportSource | null => {
  const url = parseSkillImportUrl(rawUrl);
  if (!url || url.search || url.hash) return null;
  if (url.host === "skills.sh") return parseDirectorySkillSource(url, path, ref);
  const preview = parseGitHubSource(url, path, ref, "portable");
  if (!preview || (preview.fileFromUrl && preview.fileFromUrl !== "SKILL.md")) return null;
  const source = preview.parsed;
  const revision = source.sha ?? source.ref;
  return {
    selection: {
      url: source.source === "github" ? `https://github.com/${source.repo}` : source.url ?? url.href,
      ...(skill?.trim() ? { skill: skill.trim() } : {}),
      ...(revision ? { ref: revision } : {}),
      ...(source.path || preview.fileFromUrl ? { path: source.path ?? "" } : {}),
    },
    preview,
  };
};

/**
 * Parse any git-accessible repository URL or https zip archive URL into a registerable skill
 * source. A `.zip` path is an `archive` source (S3, Artifactory, any static host). GitHub URLs
 * keep their `github`/`git-subdir` shorthand; ssh clone URLs stay ssh so a private host
 * authenticates with the user's own key; every other host is treated as a raw repo URL,
 * with an optional subfolder turning it into git-subdir.
 */
export const parseSkillSource = (rawUrl: string, subPath?: string, ref?: string): SkillSourcePreview | null => {
  const ssh = parseSshSource(rawUrl, subPath);
  if (ssh) {
    return withGitRef(ssh, ref);
  }
  const url = parseRepoUrl(rawUrl);
  if (!url) {
    return null;
  }
  const host = withoutWww(url.hostname);
  if (host === GITHUB_HOST && ["tree", "blob"].includes(pathSegments(url)[2])) {
    return parseGitHubSource(url, subPath, ref);
  }
  if (host === "gitlab.com" && /\/-\/(?:tree|blob)(?:\/|$)/.test(url.pathname)) {
    return parseGitLabBrowserSource(url, subPath, ref);
  }
  if (ZIP_ARCHIVE_REGEX.test(url.pathname)) {
    return parseArchiveSource(url);
  }
  if (host === GITHUB_HOST) {
    return parseGitHubSource(url, subPath, ref);
  }
  return withGitRef(parseRawGitSource(url, subPath), ref);
};

/**
 * Build the `~/.claude/settings.json` snippet that registers the proxy as a marketplace.
 * Claude Code expects `extraKnownMarketplaces.<name>.source` to be a source object, not a
 * bare `"url"` string, so the url/source pair is nested one level deeper. The key must be
 * "litellm" to match the name the proxy returns in marketplace.json.
 */
export const MARKETPLACE_KEY_PLACEHOLDER = "<your-litellm-key>";

export const buildMarketplaceSettingsSnippet = (proxyOrigin: string, requiresKey: boolean): string =>
  JSON.stringify(
    {
      extraKnownMarketplaces: {
        litellm: {
          source: {
            source: "url",
            url: `${proxyOrigin}/claude-code/marketplace.json`,
            ...(requiresKey ? { headers: { Authorization: `Bearer ${MARKETPLACE_KEY_PLACEHOLDER}` } } : {}),
          },
        },
      },
    },
    null,
    2,
  );

/**
 * Generate install command for Claude Code CLI.
 * Installs the named plugin from the "litellm" marketplace registered in settings.json.
 */
export const formatInstallCommand = (plugin: { name: string }): string => `/plugin install ${plugin.name}@litellm`;

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
export const getSourceDisplayText = (source: Plugin["source"]): string => {
  if (source.source === "github" && source.repo) {
    return i18n.t("claudeCodePlugins.sources.githubDisplay", { repo: source.repo });
  }
  if (source.source === "git-subdir" && source.url && source.path) {
    return `${source.url} @ ${source.path}`;
  }
  if ((source.source === "url" || source.source === "archive") && source.url) {
    return source.url;
  }
  return i18n.t("claudeCodePlugins.sources.unknown");
};

/**
 * Get clickable link for plugin source. Ssh clone urls are not browsable, so they yield null.
 */
export const getSourceLink = (source: Plugin["source"]): string | null => {
  if (!["github", "url", "git-subdir", "archive"].includes(source.source)) return null;
  const link = source.source === "github" ? source.repo && `https://github.com/${source.repo}` : source.url;
  if (!link || !BROWSABLE_URL_REGEX.test(link)) return null;
  if (source.source === "archive") return link;
  const url = toUrl(link);
  const host = url && withoutWww(url.hostname);
  if (!url || (host !== GITHUB_HOST && host !== "gitlab.com")) return link;
  const parts = pathSegments(url);
  const isRepository = host === GITHUB_HOST ? parts.length === 2 : parts.length >= 2 && !parts.includes("-");
  if (!isRepository) return link;
  const root = buildRepoUrl(url).replace(/\.git$/, "");
  const ref = source.sha || source.ref;
  if (!ref) return root;
  const path =
    source.source === "git-subdir" && source.path ? `/${source.path.split("/").map(encodeURIComponent).join("/")}` : "";
  return `${root}/${host === GITHUB_HOST ? "tree" : "-/tree"}/${encodeURIComponent(ref)}${path}`;
};

export const getManagedSkillArchivePath = (
  name: string,
  source: Plugin["source"],
  proxyBaseUrl: string,
  pageOrigin?: string,
): string | null => {
  if (source.source !== "archive" || !validatePluginName(name)) {
    return null;
  }
  if (source.sha256 && !SHA256_REGEX.test(source.sha256)) {
    return null;
  }
  const path = source.sha256
    ? `/claude-code/plugins/${name}/archive/${source.sha256}.zip`
    : `/skill-hub/skills/${name}/archive`;
  try {
    const expectedUrl = new URL(`${proxyBaseUrl.replace(/\/+$/, "")}${path}`, pageOrigin);
    return /^https?:$/.test(expectedUrl.protocol) && source.url === expectedUrl.href ? path : null;
  } catch {
    return null;
  }
};

/**
 * Get badge color based on category
 */
export const getCategoryBadgeColor = (
  category?: string | null,
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
