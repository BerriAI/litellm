import { describe, it, expect } from "vitest";
import { translate } from "@/i18n/language";
import {
  extractMCPToken,
  maskUrl,
  getMaskedAndFullUrl,
  getMCPNetworkAccess,
  validateMCPServerUrl,
  validateMCPServerName,
  normalizeToolOverrideMap,
} from "./utils";

describe("getMCPNetworkAccess", () => {
  it.each([
    { publicIp: true, explicit: false, label: "All Networks" },
    { publicIp: false, explicit: true, label: "All Networks" },
    { publicIp: true, explicit: true, label: "All Networks" },
    { publicIp: false, explicit: false, label: "Internal Only" },
    { publicIp: true, explicit: undefined, label: "All Networks" },
    { publicIp: false, explicit: undefined, label: "Unknown" },
    { publicIp: undefined, explicit: false, label: "Unknown" },
  ])("reports $label for network=$publicIp and publication=$explicit", ({ publicIp, explicit, label }) => {
    expect(
      getMCPNetworkAccess({
        available_on_public_internet: publicIp,
        mcp_info: { server_name: "demo", is_public: true, is_public_explicit: explicit },
      }).label,
    ).toBe(label);
  });

  it("explains when hub publication permits public IPs", () => {
    expect(
      getMCPNetworkAccess({
        available_on_public_internet: false,
        mcp_info: { server_name: "demo", is_public_explicit: true },
      }).description,
    ).toContain("because this server is published in MCP Hub");
  });
});

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
  it("中文提示 URL 格式错误，保留英文默认错误与验证规则", async () => {
    const t = (message: string) => translate("zh-CN", message);
    await expect(validateMCPServerUrl("not-a-url", t)).rejects.toBe(
      "请输入有效的 URL，例如 http://service-name.domain:1234/path 或 https://example.com",
    );
    await expect(validateMCPServerUrl("not-a-url")).rejects.toBe(
      "Please enter a valid URL (e.g., http://service-name.domain:1234/path or https://example.com)",
    );
  });
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
  it("中文提示名称格式错误，不改变有效业务名称", async () => {
    const t = (message: string) => translate("zh-CN", message);
    await expect(validateMCPServerName("my-server", t)).rejects.toBe("不能包含连字符或空格，请改用下划线。");
    await expect(validateMCPServerName("Home", t)).resolves.toBeUndefined();
  });
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

describe("normalizeToolOverrideMap", () => {
  it("returns empty object for nullish input", () => {
    expect(normalizeToolOverrideMap(null)).toEqual({});
    expect(normalizeToolOverrideMap(undefined)).toEqual({});
  });

  it("parses JSON string maps from legacy API responses", () => {
    expect(normalizeToolOverrideMap('{"read_wiki_structure":"browse_docs"}')).toEqual({
      read_wiki_structure: "browse_docs",
    });
  });

  it("passes through object maps unchanged", () => {
    const map = { read_user: "Read User" };
    expect(normalizeToolOverrideMap(map)).toBe(map);
  });
});
