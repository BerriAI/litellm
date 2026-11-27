import { oauth2FlowToFormValue, TRANSPORT, type MCPServer } from "@/components/mcp_tools/types";
import type { MountedFormValues } from "@/components/common_components/MountedFormField";

export const CREATE_DEFAULTS: MountedFormValues = {
  allow_all_keys: false,
  available_on_public_internet: true,
  delegate_auth_to_upstream: false,
  oauth_passthrough: false,
};

export const mapMcpEditDefaults = (server: MCPServer): MountedFormValues => ({
  ...server,
  transport: server.spec_path && server.transport !== "stdio" ? TRANSPORT.OPENAPI : server.transport,
  static_headers: Object.entries(server.static_headers ?? {}).map(([header, value]) => ({
    header,
    value: value != null ? String(value) : "",
  })),
  env_vars: Array.isArray(server.env_vars)
    ? server.env_vars.map((entry) => ({
        name: entry.name,
        value: entry.value ?? "",
        scope: entry.scope === "user" ? "user" : "global",
        description: entry.description ?? "",
      }))
    : [],
  extra_headers: server.extra_headers || [],
  oauth_flow_type: oauth2FlowToFormValue(server.oauth2_flow),
  dcr_bridge: Boolean(server.dcr_bridge),
  token_validation_json: server.token_validation ? JSON.stringify(server.token_validation, null, 2) : undefined,
});
