"use client";

import { MCPServer } from "@/components/mcp_tools/types";
import i18n from "@/i18n/i18n";

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
    get label() {
      return i18n.t("mcpServers.standardsSettings.groups.documentation");
    },
    fields: [
      {
        key: "description",
        get label() {
          return i18n.t("mcpServers.standardsSettings.fields.description.label");
        },
        get description() {
          return i18n.t("mcpServers.standardsSettings.fields.description.description");
        },
        check: (s) => !!s.description?.trim(),
      },
      {
        key: "alias",
        get label() {
          return i18n.t("mcpServers.standardsSettings.fields.alias.label");
        },
        get description() {
          return i18n.t("mcpServers.standardsSettings.fields.alias.description");
        },
        check: (s) => !!s.alias?.trim(),
      },
    ],
  },
  {
    get label() {
      return i18n.t("mcpServers.standardsSettings.groups.source");
    },
    fields: [
      {
        key: "source_url",
        get label() {
          return i18n.t("mcpServers.standardsSettings.fields.sourceUrl.label");
        },
        get description() {
          return i18n.t("mcpServers.standardsSettings.fields.sourceUrl.description");
        },
        check: (s) => !!s.source_url?.trim(),
      },
    ],
  },
  {
    get label() {
      return i18n.t("mcpServers.standardsSettings.groups.connection");
    },
    fields: [
      {
        key: "url",
        get label() {
          return i18n.t("mcpServers.standardsSettings.fields.url.label");
        },
        get description() {
          return i18n.t("mcpServers.standardsSettings.fields.url.description");
        },
        check: (s) => !!s.url?.trim(),
      },
    ],
  },
  {
    get label() {
      return i18n.t("mcpServers.standardsSettings.groups.security");
    },
    fields: [
      {
        key: "auth_type",
        get label() {
          return i18n.t("mcpServers.standardsSettings.fields.authType.label");
        },
        get description() {
          return i18n.t("mcpServers.standardsSettings.fields.authType.description");
        },
        check: (s) => !!s.auth_type && s.auth_type !== "none",
      },
    ],
  },
];

export const MCP_REQUIRED_FIELD_DEFS: RequiredFieldDef[] = FIELD_GROUPS.flatMap((g) => g.fields);

export const SETTINGS_KEY = "mcp_required_fields";
