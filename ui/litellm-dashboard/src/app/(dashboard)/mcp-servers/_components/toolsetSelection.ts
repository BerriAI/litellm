import type { MCPToolsetTool } from "@/components/mcp_tools/types";

export interface SearchableTool {
  name: string;
  description: string;
}

export type SelectionState = "all" | "some" | "none";

const toolKey = (tool: MCPToolsetTool): string => `${tool.server_id}\u0000${tool.tool_name}`;

const normalize = (query: string): string => query.trim().toLowerCase();

/** Whether a non-empty search query matches any of a server's names (alias, server name). */
export const matchesServerName = (query: string, serverNames: readonly string[]): boolean => {
  const q = normalize(query);
  return q !== "" && serverNames.some((name) => name.toLowerCase().includes(q));
};

/**
 * The tools of one server that a search query shows. A query matching the server's
 * name shows every tool on it; otherwise a tool shows when its name or description matches.
 */
export const filterServerTools = <T extends SearchableTool>(
  query: string,
  serverNames: readonly string[],
  tools: readonly T[],
): readonly T[] => {
  const q = normalize(query);
  if (!q || matchesServerName(query, serverNames)) {
    return tools;
  }
  return tools.filter((tool) => tool.name.toLowerCase().includes(q) || tool.description.toLowerCase().includes(q));
};

/** Adds `tools` to the selection, keeping existing order and skipping tools already selected. */
export const selectTools = (
  selected: readonly MCPToolsetTool[],
  tools: readonly MCPToolsetTool[],
): MCPToolsetTool[] => {
  const seen = new Set(selected.map(toolKey));
  const added = tools.filter((tool) => {
    const key = toolKey(tool);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return [...selected, ...added];
};

/** Removes exactly `tools` from the selection; every other selected tool is kept. */
export const deselectTools = (
  selected: readonly MCPToolsetTool[],
  tools: readonly MCPToolsetTool[],
): MCPToolsetTool[] => {
  const removed = new Set(tools.map(toolKey));
  return selected.filter((tool) => !removed.has(toolKey(tool)));
};

export const toggleTool = (selected: readonly MCPToolsetTool[], tool: MCPToolsetTool): MCPToolsetTool[] =>
  isToolSelected(selected, tool) ? deselectTools(selected, [tool]) : selectTools(selected, [tool]);

export const isToolSelected = (selected: readonly MCPToolsetTool[], tool: MCPToolsetTool): boolean =>
  selected.some((t) => toolKey(t) === toolKey(tool));

/** How much of `tools` is selected, for a tri-state "select all" checkbox. */
export const selectionState = (
  selected: readonly MCPToolsetTool[],
  tools: readonly MCPToolsetTool[],
): SelectionState => {
  if (tools.length === 0) return "none";
  const selectedKeys = new Set(selected.map(toolKey));
  const count = tools.filter((tool) => selectedKeys.has(toolKey(tool))).length;
  if (count === 0) return "none";
  return count === tools.length ? "all" : "some";
};
