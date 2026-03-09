"use client";

import { MCPServer } from "./types";

export interface RequiredFieldDef {
  key: string;
  label: string;
  description: string;
  check: (server: MCPServer) => boolean;
}

export interface FieldGroup {
  label: string;
  fields: RequiredFieldDef[];
}

export const FIELD_GROUPS: FieldGroup[] = [
  {
    label: "Documentation",
    fields: [
      {
        key: "description",
        label: "Description",
        description: "Must have a non-empty description",
        check: (s) => !!s.description?.trim(),
      },
      {
        key: "alias",
        label: "Alias",
        description: "Must have a display alias",
        check: (s) => !!s.alias?.trim(),
      },
    ],
  },
  {
    label: "Source",
    fields: [
      {
        key: "source_url",
        label: "GitHub / Source URL",
        description: "Must link to a source repository",
        check: (s) => !!s.source_url?.trim(),
      },
    ],
  },
  {
    label: "Connection",
    fields: [
      {
        key: "url",
        label: "Server URL",
        description: "Must have a URL configured",
        check: (s) => !!s.url?.trim(),
      },
    ],
  },
  {
    label: "Security",
    fields: [
      {
        key: "auth_type",
        label: "Auth configured",
        description: "Must use authentication (not 'none')",
        check: (s) => !!s.auth_type && s.auth_type !== "none",
      },
    ],
  },
];

export const MCP_REQUIRED_FIELD_DEFS: RequiredFieldDef[] = FIELD_GROUPS.flatMap((g) => g.fields);

export const SETTINGS_KEY = "mcp_required_fields";
