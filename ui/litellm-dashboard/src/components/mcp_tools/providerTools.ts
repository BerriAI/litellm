import type { MCPServer } from "./types";

export const providerToolChoices = (server: MCPServer): { name: string; description: string }[] =>
  (server.allowed_tools ?? []).map((name) => ({ name, description: server.tool_name_to_description?.[name] ?? "" }));
