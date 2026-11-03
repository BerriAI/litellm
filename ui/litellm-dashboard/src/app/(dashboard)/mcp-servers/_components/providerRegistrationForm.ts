import { z } from "zod";
import type { MCPServer } from "@/components/mcp_tools/types";

const toolNames = (value: string): string[] => [
  ...new Set(
    value
      .split(/\r?\n/)
      .map((name) => name.trim())
      .filter(Boolean),
  ),
];

const isProviderUrl = (value: string): boolean => {
  if (/[\s\\#]/.test(value) || [...value].some((char) => char.charCodeAt(0) < 32)) return false;
  if (!/^https:\/\/[^/?#@]+(?:[/?]|$)/i.test(value)) return false;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password;
  } catch {
    return false;
  }
};

const identifier = z
  .string()
  .trim()
  .refine((value) => !/[\s-]/.test(value), "Use underscores instead of spaces or hyphens");

export const providerRegistrationSchema = z.object({
  server_name: identifier.refine(Boolean, "Enter a server name"),
  alias: identifier,
  description: z.string(),
  url: z.string().trim().refine(isProviderUrl, "Enter an HTTPS URL without credentials or a fragment"),
  tools_text: z
    .string()
    .refine(
      (value) => toolNames(value).every((name) => !/[*?\[\]]/.test(name)),
      "Enter exact tool names without wildcards",
    ),
  mcp_access_groups: z.array(z.string()),
  allow_all_keys: z.boolean(),
});

export type ProviderRegistrationValues = z.infer<typeof providerRegistrationSchema>;

export const providerRegistrationDefaults = (server?: MCPServer): ProviderRegistrationValues => ({
  server_name: server?.server_name ?? "",
  alias: server?.alias ?? "",
  description: server?.description ?? "",
  url: server?.url ?? "",
  tools_text: (server?.allowed_tools ?? []).join("\n"),
  mcp_access_groups: server?.mcp_access_groups ?? [],
  allow_all_keys: server?.allow_all_keys ?? false,
});

export const buildProviderRegistrationPayload = (values: ProviderRegistrationValues, server?: MCPServer) => {
  const allowedTools = toolNames(values.tools_text);
  const retainedOverrides = (overrides: Readonly<Record<string, string>> | undefined) =>
    Object.fromEntries(Object.entries(overrides ?? {}).filter(([name]) => allowedTools.includes(name)));
  return {
    ...(server ? { server_id: server.server_id } : { execution_mode: "provider" as const }),
    server_name: values.server_name,
    alias: values.alias || null,
    description: values.description || null,
    url: values.url,
    allowed_tools: allowedTools,
    mcp_access_groups: values.mcp_access_groups,
    allow_all_keys: values.allow_all_keys,
    mcp_info: { ...server?.mcp_info, tool_allowlist_enforced: true },
    ...(server
      ? {
          tool_name_to_description: retainedOverrides(server.tool_name_to_description),
          tool_name_to_display_name: retainedOverrides(server.tool_name_to_display_name),
        }
      : {}),
  };
};
