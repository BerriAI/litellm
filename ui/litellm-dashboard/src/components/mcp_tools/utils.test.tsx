import { describe, it, expect } from "vitest";
import {
  extractMCPToken,
  maskUrl,
  getMaskedAndFullUrl,
  validateMCPServerUrl,
  validateMCPServerName,
  guessLogoFromUrl,
} from "./utils";

describe("extractMCPToken", () => {
  it("should extract token after /mcp/", () => {
    const result = extractMCPToken("https://example.com/mcp/abc123");
    expect(result).toEqual({ token: "abc123", baseUrl: "https://example.com/mcp/" });
  });

  it("should return null token when URL has no /mcp/ segment", () => {
    const result = extractMCPToken("https://example.com/api/v1");
    expect(result).toEqual({ token: null, baseUrl: "https://example.com/api/v1" });
  });

  it("should return null token when nothing follows /mcp/", () => {
    const result = extractMCPToken("https://example.com/mcp/");
    expect(result).toEqual({ token: null, baseUrl: "https://example.com/mcp/" });
  });
});

describe("maskUrl", () => {
  it("should replace the token with ellipsis", () => {
    expect(maskUrl("https://example.com/mcp/secret-token")).toBe("https://example.com/mcp/...");
  });

  it("should return the original URL when there is no token", () => {
    expect(maskUrl("https://example.com/api")).toBe("https://example.com/api");
  });
});

describe("getMaskedAndFullUrl", () => {
  it("should return hasToken true when a token exists", () => {
    const result = getMaskedAndFullUrl("https://example.com/mcp/tok");
    expect(result).toEqual({ maskedUrl: "https://example.com/mcp/...", hasToken: true });
  });

  it("should return hasToken false when no token exists", () => {
    const result = getMaskedAndFullUrl("https://example.com/api");
    expect(result).toEqual({ maskedUrl: "https://example.com/api", hasToken: false });
  });
});

describe("validateMCPServerUrl", () => {
  it("should resolve for a valid HTTP URL", async () => {
    await expect(validateMCPServerUrl("https://example.com/path")).resolves.toBeUndefined();
  });

  it("should resolve for an empty string", async () => {
    await expect(validateMCPServerUrl("")).resolves.toBeUndefined();
  });

  it("should reject for an invalid URL", async () => {
    await expect(validateMCPServerUrl("not-a-url")).rejects.toBeDefined();
  });
});

describe("validateMCPServerName", () => {
  it("should resolve for a valid underscore name", async () => {
    await expect(validateMCPServerName("my_server")).resolves.toBeUndefined();
  });

  it("should reject names containing hyphens", async () => {
    await expect(validateMCPServerName("my-server")).rejects.toBeDefined();
  });

  it("should reject names containing spaces", async () => {
    await expect(validateMCPServerName("my server")).rejects.toBeDefined();
  });
});

describe("guessLogoFromUrl", () => {
  it("should match an exact host against the registry", () => {
    expect(guessLogoFromUrl("https://github.com/org/repo")).toBe(
      "/ui/assets/logos/github.svg",
    );
    expect(guessLogoFromUrl("https://figma.com")).toBe(
      "/ui/assets/logos/figma.svg",
    );
  });

  it("should match wildcard subdomains", () => {
    expect(guessLogoFromUrl("https://api.github.com/user")).toBe(
      "/ui/assets/logos/github.svg",
    );
    expect(guessLogoFromUrl("https://acme.atlassian.net/jira/api")).toBe(
      "/ui/assets/logos/jira.svg",
    );
    expect(guessLogoFromUrl("https://api.linear.app/graphql")).toBe(
      "/ui/assets/logos/linear.svg",
    );
    expect(guessLogoFromUrl("https://shop.myshopify.com")).toBe(
      "/ui/assets/logos/shopify.svg",
    );
  });

  it("should be case-insensitive on the host", () => {
    expect(guessLogoFromUrl("https://API.GitHub.com")).toBe(
      "/ui/assets/logos/github.svg",
    );
  });

  it("should return undefined for unknown hosts", () => {
    expect(
      guessLogoFromUrl("https://internal-tools.example.com"),
    ).toBeUndefined();
    expect(guessLogoFromUrl("https://localhost:4000/mcp")).toBeUndefined();
  });

  it("should return undefined for unparseable URLs and empty input", () => {
    expect(guessLogoFromUrl(undefined)).toBeUndefined();
    expect(guessLogoFromUrl(null)).toBeUndefined();
    expect(guessLogoFromUrl("")).toBeUndefined();
    expect(guessLogoFromUrl("not a url")).toBeUndefined();
  });

  it("should not match a wildcard pattern against an unrelated host", () => {
    // `*.github.com` must not match `github.com.evil.example`.
    expect(
      guessLogoFromUrl("https://github.com.evil.example/path"),
    ).toBeUndefined();
  });
});
