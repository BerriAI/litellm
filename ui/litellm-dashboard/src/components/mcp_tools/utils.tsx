export const extractMCPToken = (url: string): { token: string | null; baseUrl: string } => {
  try {
    const mcpIndex = url.indexOf("/mcp/");
    if (mcpIndex === -1) return { token: null, baseUrl: url };

    const parts = url.split("/mcp/");
    if (parts.length !== 2) return { token: null, baseUrl: url };

    const baseUrl = parts[0] + "/mcp/";
    const afterMcp = parts[1];

    // If there's no content after /mcp/, return null token
    if (!afterMcp) return { token: null, baseUrl: url };

    return {
      token: afterMcp,
      baseUrl: baseUrl,
    };
  } catch (error) {
    console.error("Error parsing MCP URL:", error);
    return { token: null, baseUrl: url };
  }
};

export const maskUrl = (url: string): string => {
  const { token, baseUrl } = extractMCPToken(url);
  if (!token) return url;
  return baseUrl + "...";
};

export const getMaskedAndFullUrl = (url: string): { maskedUrl: string; hasToken: boolean } => {
  const { token } = extractMCPToken(url);
  return {
    maskedUrl: maskUrl(url),
    hasToken: !!token,
  };
};

// Validation utilities for MCP server forms
export const validateMCPServerUrl = (value: string) => {
  if (!value) return Promise.resolve();
  // More flexible URL validation that allows Kubernetes service names and various URL formats
  const urlPattern = /^https?:\/\/[^\s/$.?#].[^\s]*$/i;
  return urlPattern.test(value)
    ? Promise.resolve()
    : Promise.reject("Please enter a valid URL (e.g., http://service-name.domain:1234/path or https://example.com)");
};

export const validateMCPServerName = (value: string) => {
  return value && (value.includes("-") || value.includes(" "))
    ? Promise.reject("Cannot contain '-' (hyphen) or spaces. Please use '_' (underscore) instead.")
    : Promise.resolve();
};

// Local SVG asset directory used for the curated logo set. Kept as a
// constant so MCPLogoSelector and any URL-based guesser stay in sync.
export const LOGOS_DIR = "/ui/assets/logos/";

export interface WellKnownLogo {
  /** Display name shown in tooltips / pickers. */
  name: string;
  /** Local SVG asset URL (served from /ui/assets/logos/). */
  url: string;
  /**
   * Hostnames whose URL should pre-select this logo at create/edit time.
   * Each entry is matched as an exact host or a `*.suffix` wildcard. The
   * list is small and explicit on purpose — no slug arithmetic, no
   * third-party CDN lookups, no surprise 404s. Add a host here when a
   * brand reliably exposes its API on it.
   */
  hosts?: ReadonlyArray<string>;
}

/**
 * Curated logo registry. Single source of truth shared by `MCPLogoSelector`
 * (the picker grid) and `guessLogoFromUrl` (the create-form host lookup).
 */
export const WELL_KNOWN_LOGOS: ReadonlyArray<WellKnownLogo> = [
  {
    name: "GitHub",
    url: `${LOGOS_DIR}github.svg`,
    hosts: ["github.com", "*.github.com", "*.githubusercontent.com"],
  },
  {
    name: "GitLab",
    url: `${LOGOS_DIR}gitlab.svg`,
    hosts: ["gitlab.com", "*.gitlab.com"],
  },
  {
    name: "Slack",
    url: `${LOGOS_DIR}slack.svg`,
    hosts: ["slack.com", "*.slack.com"],
  },
  {
    name: "Notion",
    url: `${LOGOS_DIR}notion.svg`,
    hosts: ["notion.com", "*.notion.com", "notion.so", "*.notion.so"],
  },
  {
    name: "Linear",
    url: `${LOGOS_DIR}linear.svg`,
    hosts: ["linear.app", "*.linear.app"],
  },
  {
    name: "Jira",
    url: `${LOGOS_DIR}jira.svg`,
    hosts: ["*.atlassian.net", "*.atlassian.com", "atlassian.com"],
  },
  {
    name: "Figma",
    url: `${LOGOS_DIR}figma.svg`,
    hosts: ["figma.com", "*.figma.com"],
  },
  {
    name: "Gmail",
    url: `${LOGOS_DIR}gmail.svg`,
    hosts: ["mail.google.com"],
  },
  {
    name: "Google Drive",
    url: `${LOGOS_DIR}google_drive.svg`,
    hosts: ["drive.google.com"],
  },
  {
    name: "Google",
    url: `${LOGOS_DIR}google.svg`,
    hosts: ["google.com", "*.googleapis.com"],
  },
  {
    name: "Stripe",
    url: `${LOGOS_DIR}stripe.svg`,
    hosts: ["stripe.com", "*.stripe.com"],
  },
  {
    name: "Shopify",
    url: `${LOGOS_DIR}shopify.svg`,
    hosts: ["shopify.com", "*.shopify.com", "*.myshopify.com"],
  },
  {
    name: "Salesforce",
    url: `${LOGOS_DIR}salesforce.svg`,
    hosts: ["salesforce.com", "*.salesforce.com", "*.force.com"],
  },
  {
    name: "HubSpot",
    url: `${LOGOS_DIR}hubspot.svg`,
    hosts: ["hubspot.com", "*.hubspot.com", "*.hubapi.com"],
  },
  {
    name: "Twilio",
    url: `${LOGOS_DIR}twilio.svg`,
    hosts: ["twilio.com", "*.twilio.com"],
  },
  {
    name: "Cloudflare",
    url: `${LOGOS_DIR}cloudflare.svg`,
    hosts: ["cloudflare.com", "*.cloudflare.com"],
  },
  {
    name: "Sentry",
    url: `${LOGOS_DIR}sentry.svg`,
    hosts: ["sentry.io", "*.sentry.io"],
  },
  {
    name: "PostgreSQL",
    url: `${LOGOS_DIR}postgresql.svg`,
  },
  {
    name: "Snowflake",
    url: `${LOGOS_DIR}snowflake.svg`,
    hosts: ["*.snowflakecomputing.com"],
  },
  {
    name: "Zapier",
    url: `${LOGOS_DIR}zapier.svg`,
    hosts: ["zapier.com", "*.zapier.com"],
  },
];

const matchesHostPattern = (host: string, pattern: string): boolean => {
  if (pattern.startsWith("*.")) {
    const suffix = pattern.slice(2).toLowerCase();
    const lower = host.toLowerCase();
    return lower === suffix || lower.endsWith(`.${suffix}`);
  }
  return host.toLowerCase() === pattern.toLowerCase();
};

/**
 * Best-effort logo suggestion based on the upstream URL host.
 *
 * Used at create/edit time to pre-select a logo from the server URL the
 * admin typed (e.g. `https://api.github.com/...` → GitHub). The match
 * table is the explicit `WELL_KNOWN_LOGOS.hosts` list — no slug
 * arithmetic on the server's name, no third-party CDN lookups. Returns
 * `undefined` when the URL is unparseable or its host isn't in the
 * registry; the caller should leave the logo unset in that case.
 */
export const guessLogoFromUrl = (
  url: string | null | undefined,
): string | undefined => {
  if (!url) return undefined;
  let host: string;
  try {
    host = new URL(url).hostname;
  } catch {
    return undefined;
  }
  if (!host) return undefined;
  for (const logo of WELL_KNOWN_LOGOS) {
    if (logo.hosts?.some((p) => matchesHostPattern(host, p))) {
      return logo.url;
    }
  }
  return undefined;
};
