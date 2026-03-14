import { describe, it, expect } from "vitest";
import {
  extractMCPToken,
  maskUrl,
  getMaskedAndFullUrl,
  validateMCPServerUrl,
  validateMCPServerName,
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
