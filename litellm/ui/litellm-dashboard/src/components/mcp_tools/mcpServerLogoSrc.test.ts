import { describe, expect, it } from "vitest";
import { getMcpServerLogoSrc } from "./mcpServerLogoSrc";

describe("getMcpServerLogoSrc", () => {
  it("prefers an explicitly configured logo", () => {
    expect(
      getMcpServerLogoSrc({
        url: "https://mcp.example.com/mcp",
        mcp_info: { server_name: "demo", logo_url: "/ui/assets/logos/github.svg" },
      }),
    ).toEqual(["/ui/assets/logos/github.svg"]);
  });

  it("derives the favicon from the origin without passing server URL credentials, path, or query", () => {
    expect(getMcpServerLogoSrc({ url: "https://user:pass@mcp.example.com:8443/mcp?token=example#settings" })).toEqual([
      "https://mcp.example.com:8443/favicon.ico",
      "https://example.com/favicon.ico",
    ]);
  });

  it("orders candidates from the endpoint origin through each parent", () => {
    expect(getMcpServerLogoSrc({ url: "https://qa.mcp.example.com/mcp" })).toEqual([
      "https://qa.mcp.example.com/favicon.ico",
      "https://mcp.example.com/favicon.ico",
      "https://example.com/favicon.ico",
    ]);
  });

  it.each(["cloudfront.net", "amazonaws.com", "azurewebsites.net", "vercel.app", "herokuapp.com", "github.io"])(
    "does not guess a logo from a shared host under %s",
    (suffix) => {
      expect(getMcpServerLogoSrc({ url: `https://api.tenant.${suffix}/mcp` })).toEqual([
        `https://api.tenant.${suffix}/favicon.ico`,
      ]);
    },
  );

  it.each(["co.uk", "com.au"])("does not request the public suffix %s", (suffix) => {
    expect(getMcpServerLogoSrc({ url: `https://api.example.${suffix}/mcp` })).toEqual([
      `https://api.example.${suffix}/favicon.ico`,
      `https://example.${suffix}/favicon.ico`,
    ]);
  });

  it.each([
    "https://mcp.namespace.svc.cluster.local/mcp",
    "https://my-host.localhost./mcp",
    "https://intranet/mcp",
    "https://127.0.0.1/mcp",
    "https://[::1]/mcp",
    "http://mcp.example.com/mcp",
    "not a URL",
    undefined,
  ])("does not derive a favicon for %s", (url) => {
    expect(getMcpServerLogoSrc({ url })).toEqual([]);
  });

  it("does not derive a favicon for servers marked private or using stdio", () => {
    const url = "https://mcp.example.com/mcp";
    expect(getMcpServerLogoSrc({ url, available_on_public_internet: false })).toEqual([]);
    expect(getMcpServerLogoSrc({ url, transport: "stdio" })).toEqual([]);
  });
});
