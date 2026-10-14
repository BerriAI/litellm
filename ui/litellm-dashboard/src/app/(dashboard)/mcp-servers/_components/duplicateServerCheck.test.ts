import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/http/client";
import { findDuplicateMcpServer, mcpSubmitErrorReason } from "./duplicateServerCheck";

const servers = [
  { server_id: "s1", server_name: "GitHub_MCP", alias: "github" },
  { server_id: "s2", server_name: "Email Service", alias: "email_service" },
];

describe("findDuplicateMcpServer", () => {
  it("flags an incoming server_name that matches an existing alias", () => {
    expect(findDuplicateMcpServer(servers, "github", "other")?.field).toBe("server_name");
  });

  it("flags an incoming alias that matches an existing server_name", () => {
    expect(findDuplicateMcpServer(servers, "new", "GitHub_MCP")?.serverId).toBe("s1");
  });

  it("matches case-insensitively", () => {
    expect(findDuplicateMcpServer(servers, "GITHUB", "new")?.serverId).toBe("s1");
  });

  it("normalizes spaces to underscores like the backend does", () => {
    expect(findDuplicateMcpServer(servers, "new", "email service")?.serverId).toBe("s2");
  });

  it("does not flag the server's own identifiers while editing", () => {
    expect(findDuplicateMcpServer(servers, "GitHub_MCP", "github", "s1")).toBeNull();
  });

  it("flags the same alias on a different server while editing", () => {
    expect(findDuplicateMcpServer(servers, "other", "github", "s2")?.serverId).toBe("s1");
  });

  it("returns null when nothing matches", () => {
    expect(findDuplicateMcpServer(servers, "brand_new", "brand_new")).toBeNull();
  });
});

describe("mcpSubmitErrorReason", () => {
  it("unwraps the FastAPI detail.error envelope into readable toast text", () => {
    const error = new ApiError("boom", 400, { detail: { error: "An MCP server with alias 'x' already exists" } });
    expect(mcpSubmitErrorReason(error)).toContain("An MCP server with alias 'x' already exists");
  });

  it("never produces [object Object] for a non-Error rejection", () => {
    expect(mcpSubmitErrorReason({ detail: { error: "structured 400" } })).toBe("structured 400");
  });
});
