import { describe, it, expect } from "vitest";
import { buildMcpToolBlocks } from "./mcp_tool_blocks";
import { MCPServer, MCPToolset } from "@/components/mcp_tools/types";

const server = (over: Partial<MCPServer>): MCPServer =>
  ({
    server_id: "id-1",
    server_name: "deepwiki",
    alias: "wiki",
    url: "",
    transport: "http",
    auth_type: "none",
    ...over,
  }) as any;

describe("buildMcpToolBlocks", () => {
  it("returns no blocks when nothing is selected", () => {
    expect(buildMcpToolBlocks({ selectedMCPServers: [] })).toEqual([]);
    expect(buildMcpToolBlocks({ selectedMCPServers: undefined })).toEqual([]);
  });

  it("routes by server_name, not alias, so colliding aliases cannot cross-route", () => {
    const [block] = buildMcpToolBlocks({
      selectedMCPServers: ["id-1"],
      mcpServers: [server({})],
    });
    expect(block.server_url).toBe("litellm_proxy/mcp/deepwiki");
    expect(block.server_label).toBe("deepwiki");
  });

  it("does not percent-encode the name; the gateway splits the raw path and never decodes", () => {
    const [block] = buildMcpToolBlocks({
      selectedMCPServers: ["id-1"],
      mcpServers: [server({ server_name: "my server" }) as any],
    });
    expect(block.server_url).toBe("litellm_proxy/mcp/my server");
    expect(block.server_url).not.toContain("%20");
  });

  it("passes per-server tool restrictions through as allowed_tools", () => {
    const [block] = buildMcpToolBlocks({
      selectedMCPServers: ["id-1"],
      mcpServers: [server({})],
      mcpServerToolRestrictions: { "id-1": ["read_wiki_structure"] },
    });
    expect(block.allowed_tools).toEqual(["read_wiki_structure"]);
  });

  it("collapses the all-servers sentinel to a single proxy-wide block", () => {
    expect(buildMcpToolBlocks({ selectedMCPServers: ["__all__", "id-1"] })).toEqual([
      { type: "mcp", server_label: "litellm", server_url: "litellm_proxy/mcp", require_approval: "never" },
    ]);
  });

  it("routes a toolset by its name", () => {
    const toolset = { toolset_id: "ts-1", toolset_name: "docs" } as MCPToolset;
    const [block] = buildMcpToolBlocks({
      selectedMCPServers: ["toolset:ts-1"],
      mcpToolsets: [toolset],
    });
    expect(block.server_url).toBe("litellm_proxy/mcp/docs");
  });
});
