import type { MCPServer } from "./types";

type LogoServer = Pick<MCPServer, "mcp_info" | "url" | "transport" | "available_on_public_internet">;

const SHARED_HOSTING_SUFFIXES = [
  "amazonaws.com",
  "appspot.com",
  "azurewebsites.net",
  "cloudfront.net",
  "github.io",
  "herokuapp.com",
  "netlify.app",
  "onrender.com",
  "pages.dev",
  "vercel.app",
  "workers.dev",
] as const;

const COUNTRY_CODE_SECOND_LEVELS = new Set(["ac", "co", "com", "edu", "gov", "net", "org"]);

export function getMcpServerLogoSrc(server: LogoServer): string[] {
  const configuredLogo = server.mcp_info?.logo_url?.trim();
  if (configuredLogo) return [configuredLogo];
  if (!server.url || server.transport === "stdio" || server.available_on_public_internet === false) return [];

  const url = URL.canParse(server.url) ? new URL(server.url) : undefined;
  if (url?.protocol !== "https:") return [];

  const hostname = url.hostname.replace(/\.$/, "");
  if (!/^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z][a-z0-9-]*$/.test(hostname)) return [];
  if (
    /\.(?:arpa|corp|example|home|internal|intranet|invalid|lan|local|localdomain|localhost|onion|private|test)$/.test(
      hostname,
    )
  )
    return [];

  const origin = `${url.origin}/favicon.ico`;
  if (SHARED_HOSTING_SUFFIXES.some((suffix) => hostname === suffix || hostname.endsWith(`.${suffix}`))) {
    return [origin];
  }

  const labels = hostname.split(".");
  const [tld, secondLevel] = labels.slice(-2).reverse();
  const minLabels = tld.length === 2 && COUNTRY_CODE_SECOND_LEVELS.has(secondLevel) ? 3 : 2;
  if (labels.length < minLabels) return [];
  const parents = Array.from(
    { length: labels.length - minLabels },
    (_, index) => `https://${labels.slice(index + 1).join(".")}/favicon.ico`,
  );

  return [origin, ...parents];
}
