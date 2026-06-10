"use client";

import type { TFunction } from "i18next";
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

export const getFieldGroups = (t: TFunction): FieldGroup[] => [
  {
    label: t("mcpTools.mcpStandardsSettings.groupDocumentation"),
    fields: [
      {
        key: "description",
        label: t("mcpTools.mcpStandardsSettings.fieldDescriptionLabel"),
        description: t("mcpTools.mcpStandardsSettings.fieldDescriptionDesc"),
        check: (s) => !!s.description?.trim(),
      },
      {
        key: "alias",
        label: t("mcpTools.mcpStandardsSettings.fieldAliasLabel"),
        description: t("mcpTools.mcpStandardsSettings.fieldAliasDesc"),
        check: (s) => !!s.alias?.trim(),
      },
    ],
  },
  {
    label: t("mcpTools.mcpStandardsSettings.groupSource"),
    fields: [
      {
        key: "source_url",
        label: t("mcpTools.mcpStandardsSettings.fieldSourceUrlLabel"),
        description: t("mcpTools.mcpStandardsSettings.fieldSourceUrlDesc"),
        check: (s) => !!s.source_url?.trim(),
      },
    ],
  },
  {
    label: t("mcpTools.mcpStandardsSettings.groupConnection"),
    fields: [
      {
        key: "url",
        label: t("mcpTools.mcpStandardsSettings.fieldServerUrlLabel"),
        description: t("mcpTools.mcpStandardsSettings.fieldServerUrlDesc"),
        check: (s) => !!s.url?.trim(),
      },
    ],
  },
  {
    label: t("mcpTools.mcpStandardsSettings.groupSecurity"),
    fields: [
      {
        key: "auth_type",
        label: t("mcpTools.mcpStandardsSettings.fieldAuthLabel"),
        description: t("mcpTools.mcpStandardsSettings.fieldAuthDesc"),
        check: (s) => !!s.auth_type && s.auth_type !== "none",
      },
    ],
  },
];

export const getMcpRequiredFieldDefs = (t: TFunction): RequiredFieldDef[] => getFieldGroups(t).flatMap((g) => g.fields);

export const SETTINGS_KEY = "mcp_required_fields";
