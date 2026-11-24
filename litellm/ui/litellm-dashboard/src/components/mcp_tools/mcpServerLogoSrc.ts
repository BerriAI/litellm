import type { MCPServer } from "./types";

type LogoServer = Pick<MCPServer, "mcp_info" | "url" | "transport" | "available_on_public_internet">;

export function getMcpServerLogoSrc(server: LogoServer): string | undefined {
  const configuredLogo = server.mcp_info?.logo_url?.trim();
  if (configuredLogo) return configuredLogo;
  if (!server.url || server.transport === "stdio" || server.available_on_public_internet === false) return;

  const url = URL.canParse(server.url) ? new URL(server.url) : undefined;
  if (url?.protocol !== "https:") return;

  const hostname = url.hostname.replace(/\.$/, "");
  if (!/^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z][a-z0-9-]*$/.test(hostname)) return;
  if (
    /\.(?:arpa|corp|example|home|internal|intranet|invalid|lan|local|localdomain|localhost|onion|private|test)$/.test(
      hostname,
    )
  )
    return;

  return `${url.origin}/favicon.ico`;
}
