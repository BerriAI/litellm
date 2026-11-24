import { describe, expect, it } from "vitest";
import { getMcpServerLogoSrc } from "./mcpServerLogoSrc";

describe("getMcpServerLogoSrc", () => {
  it("prefers an explicitly configured logo", () => {
    expect(
      getMcpServerLogoSrc({
        url: "https://mcp.example.com/mcp",
        mcp_info: { server_name: "demo", logo_url: "/ui/assets/logos/github.svg" },
      }),
    ).toBe("/ui/assets/logos/github.svg");
  });

  it("derives the favicon from the origin without passing server URL credentials, path, or query", () => {
    expect(getMcpServerLogoSrc({ url: "https://user:pass@mcp.example.com:8443/mcp?token=example#settings" })).toBe(
      "https://mcp.example.com:8443/favicon.ico",
    );
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
    expect(getMcpServerLogoSrc({ url })).toBeUndefined();
  });

  it("does not derive a favicon for servers marked private or using stdio", () => {
    const url = "https://mcp.example.com/mcp";
    expect(getMcpServerLogoSrc({ url, available_on_public_internet: false })).toBeUndefined();
    expect(getMcpServerLogoSrc({ url, transport: "stdio" })).toBeUndefined();
  });
});
