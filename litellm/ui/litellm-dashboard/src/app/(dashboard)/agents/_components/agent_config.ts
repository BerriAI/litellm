import type { ParseKeys } from "i18next";
import { parseIdentityForForm } from "./agent_identity";
/**
 * Shared configuration for agent form fields
 * Used across create, view, and update operations
 */

import {
  EMPTY_KILL_SWITCH_FORM,
  buildKillSwitchFromForm,
  parseKillSwitchForForm,
  type KillSwitchConfig,
  type KillSwitchFormValue,
} from "./kill_switch_config";

export interface FieldConfig {
  name: string;
  labelKey: ParseKeys;
  type: "text" | "textarea" | "url" | "switch" | "list" | "select";
  required?: boolean;
  tooltipKey?: ParseKeys;
  placeholder?: string;
  placeholderKey?: ParseKeys;
  defaultValue?: any;
  rows?: number;
  validation?: any[];
  options?: string[];
  helpTextKey?: ParseKeys;
}

export interface SectionConfig {
  key: string;
  titleKey: ParseKeys;
  fields: FieldConfig[];
  defaultExpanded?: boolean;
}

export const AGENT_FORM_CONFIG: {
  basic: SectionConfig;
  skills: SectionConfig;
  capabilities: SectionConfig;
  optional: SectionConfig;
  litellm: SectionConfig;
  cost: SectionConfig;
  tracing: SectionConfig;
} = {
  basic: {
    key: "basic",
    titleKey: "agents.formConfig.basic.title",
    defaultExpanded: true,
    fields: [
      {
        name: "name",
        labelKey: "agents.formConfig.fields.name.label",
        type: "text",
        required: true,
        placeholderKey: "agents.formConfig.fields.name.placeholder",
      },
      {
        name: "description",
        labelKey: "agents.formConfig.fields.description.label",
        type: "textarea",
        required: false,
        placeholderKey: "agents.formConfig.fields.description.placeholder",
        rows: 3,
      },
      {
        name: "url",
        labelKey: "agents.formConfig.fields.url.label",
        type: "url",
        required: false,
        placeholder: "http://localhost:9999/",
        tooltipKey: "agents.formConfig.fields.url.tooltip",
      },
      {
        name: "version",
        labelKey: "agents.formConfig.fields.version.label",
        type: "text",
        placeholder: "1.0.0",
        defaultValue: "1.0.0",
      },
      {
        name: "protocolVersion",
        labelKey: "agents.formConfig.fields.protocolVersion.label",
        type: "select",
        options: ["1.0", "0.3"],
        defaultValue: "1.0",
        tooltipKey: "agents.formConfig.fields.protocolVersion.tooltip",
        helpTextKey: "agents.formConfig.basic.protocolVersionHelp",
      },
    ],
  },
  skills: {
    key: "skills",
    titleKey: "agents.formConfig.skills.title",
    fields: [
      {
        name: "skills",
        labelKey: "agents.formConfig.fields.skills.label",
        type: "list",
        defaultValue: [],
      },
    ],
  },
  capabilities: {
    key: "capabilities",
    titleKey: "agents.formConfig.capabilities.title",
    fields: [
      {
        name: "streaming",
        labelKey: "agents.formConfig.fields.streaming.label",
        type: "switch",
        defaultValue: false,
      },
      {
        name: "pushNotifications",
        labelKey: "agents.formConfig.fields.pushNotifications.label",
        type: "switch",
      },
      {
        name: "stateTransitionHistory",
        labelKey: "agents.formConfig.fields.stateTransitionHistory.label",
        type: "switch",
      },
    ],
  },
  optional: {
    key: "optional",
    titleKey: "agents.formConfig.optional.title",
    fields: [
      {
        name: "iconUrl",
        labelKey: "agents.formConfig.fields.iconUrl.label",
        type: "url",
        placeholder: "https://example.com/icon.png",
      },
      {
        name: "documentationUrl",
        labelKey: "agents.formConfig.fields.documentationUrl.label",
        type: "url",
        placeholder: "https://docs.example.com",
      },
      {
        name: "supportsAuthenticatedExtendedCard",
        labelKey: "agents.formConfig.fields.supportsAuthenticatedExtendedCard.label",
        type: "switch",
      },
    ],
  },
  litellm: {
    key: "litellm",
    titleKey: "agents.formConfig.litellm.title",
    fields: [
      {
        name: "model",
        labelKey: "agents.formConfig.fields.model.label",
        type: "text",
      },
      {
        name: "make_public",
        labelKey: "agents.formConfig.fields.makePublic.label",
        type: "switch",
      },
    ],
  },
  cost: {
    key: "cost",
    titleKey: "agents.formConfig.cost.title",
    fields: [
      {
        name: "cost_per_query",
        labelKey: "agents.formConfig.fields.costPerQuery.label",
        type: "text",
        placeholder: "0.0",
        tooltipKey: "agents.formConfig.fields.costPerQuery.tooltip",
      },
      {
        name: "input_cost_per_token",
        labelKey: "agents.formConfig.fields.inputCostPerToken.label",
        type: "text",
        placeholder: "0.000001",
        tooltipKey: "agents.formConfig.fields.inputCostPerToken.tooltip",
      },
      {
        name: "output_cost_per_token",
        labelKey: "agents.formConfig.fields.outputCostPerToken.label",
        type: "text",
        placeholder: "0.000002",
        tooltipKey: "agents.formConfig.fields.outputCostPerToken.tooltip",
      },
    ],
  },
  tracing: {
    key: "tracing",
    titleKey: "agents.formConfig.tracing.title",
    fields: [
      {
        name: "enable_tracing",
        labelKey: "agents.formConfig.fields.enableTracing.label",
        type: "switch",
        defaultValue: false,
        tooltipKey: "agents.formConfig.fields.enableTracing.tooltip",
      },
    ],
  },
};

export const SKILL_FIELD_CONFIG = {
  id: {
    name: "id",
    labelKey: "agents.formConfig.skillFields.id" as ParseKeys,
    required: true,
    placeholderKey: "agents.formConfig.skillFields.idPlaceholder" as ParseKeys,
  },
  name: {
    name: "name",
    labelKey: "agents.formConfig.skillFields.name" as ParseKeys,
    required: true,
    placeholderKey: "agents.formConfig.skillFields.namePlaceholder" as ParseKeys,
  },
  description: {
    name: "description",
    labelKey: "agents.formConfig.skillFields.description" as ParseKeys,
    required: true,
    placeholderKey: "agents.formConfig.skillFields.descriptionPlaceholder" as ParseKeys,
    rows: 2,
  },
  tags: {
    name: "tags",
    labelKey: "agents.formConfig.skillFields.tags" as ParseKeys,
    required: true,
    placeholderKey: "agents.formConfig.skillFields.tagsPlaceholder" as ParseKeys,
  },
  examples: {
    name: "examples",
    labelKey: "agents.formConfig.skillFields.examples" as ParseKeys,
    placeholderKey: "agents.formConfig.skillFields.examplesPlaceholder" as ParseKeys,
  },
};

/**
 * Get default form values from configuration
 */
export const getDefaultFormValues = () => {
  const defaults: any = {
    defaultInputModes: ["text"],
    defaultOutputModes: ["text"],
    kill_switch: { ...EMPTY_KILL_SWITCH_FORM },
  };

  Object.values(AGENT_FORM_CONFIG).forEach((section) => {
    section.fields.forEach((field) => {
      if (field.defaultValue !== undefined) {
        defaults[field.name] = field.defaultValue;
      }
    });
  });

  return defaults;
};

/**
 * Build agent data from form values according to AgentConfig spec
 */
export const buildAgentDataFromForm = (values: any, existingAgent?: any) => {
  const agentData: any = {
    agent_name: values.agent_name,
    agent_card_params: {
      protocolVersion: values.protocolVersion || "1.0",
      name: values.name || values.agent_name,
      description: values.description || "",
      url: values.url || "",
      version: values.version || "1.0.0",
      defaultInputModes: existingAgent?.agent_card_params?.defaultInputModes || ["text"],
      defaultOutputModes: existingAgent?.agent_card_params?.defaultOutputModes || ["text"],
      capabilities: {
        streaming: values.streaming === true,
        ...(values.pushNotifications !== undefined && { pushNotifications: values.pushNotifications }),
        ...(values.stateTransitionHistory !== undefined && { stateTransitionHistory: values.stateTransitionHistory }),
      },
      skills: values.skills || [],
      ...(values.iconUrl && { iconUrl: values.iconUrl }),
      ...(values.documentationUrl && { documentationUrl: values.documentationUrl }),
      ...(values.supportsAuthenticatedExtendedCard !== undefined && {
        supportsAuthenticatedExtendedCard: values.supportsAuthenticatedExtendedCard,
      }),
    },
  };

  const params: Record<string, any> = {};

  if (values.model) params.model = values.model;
  if (values.make_public !== undefined) params.make_public = values.make_public;
  if (values.cost_per_query) params.cost_per_query = parseFloat(values.cost_per_query);
  if (values.input_cost_per_token) params.input_cost_per_token = parseFloat(values.input_cost_per_token);
  if (values.output_cost_per_token) params.output_cost_per_token = parseFloat(values.output_cost_per_token);

  if (Object.keys(params).length > 0) {
    agentData.litellm_params = params;
  }

  if (values.tpm_limit != null) agentData.tpm_limit = values.tpm_limit;
  if (values.rpm_limit != null) agentData.rpm_limit = values.rpm_limit;
  if (values.session_tpm_limit != null) agentData.session_tpm_limit = values.session_tpm_limit;
  if (values.session_rpm_limit != null) agentData.session_rpm_limit = values.session_rpm_limit;
  // static_headers: convert [{header, value}, ...] → {header: value, ...}
  if (Array.isArray(values.static_headers) && values.static_headers.length > 0) {
    const staticHeaders: Record<string, string> = {};
    values.static_headers.forEach((entry: { header?: string; value?: string }) => {
      const key = entry?.header?.trim();
      if (key) staticHeaders[key] = entry?.value ?? "";
    });
    if (Object.keys(staticHeaders).length > 0) {
      agentData.static_headers = staticHeaders;
    }
  }

  // extra_headers: already an array of strings from Select tags
  if (Array.isArray(values.extra_headers) && values.extra_headers.length > 0) {
    agentData.extra_headers = values.extra_headers;
  }

  applyKillSwitchToPayload(agentData, values.kill_switch, existingAgent);

  return agentData;
};

export const applyKillSwitchToPayload = (
  agentData: { kill_switch?: KillSwitchConfig | null },
  form: KillSwitchFormValue | undefined,
  existingAgent?: { kill_switch?: KillSwitchConfig | null },
) => {
  const killSwitch = buildKillSwitchFromForm(form);
  if (killSwitch !== undefined && (killSwitch !== null || existingAgent?.kill_switch)) {
    agentData.kill_switch = killSwitch;
  }
};

export const parseAccessGroupIdsForForm = (agent: { access_group_ids?: string[] | null }) => ({
  access_group_ids: agent.access_group_ids ?? [],
});

export const parseMcpPermissionsForForm = (agent: any) => ({
  ...parseIdentityForForm(agent),
  allowed_mcp_servers_and_groups: {
    servers: agent.object_permission?.mcp_servers ?? [],
    accessGroups: agent.object_permission?.mcp_access_groups ?? [],
    toolsets: agent.object_permission?.mcp_toolsets ?? [],
  },
  mcp_tool_permissions: agent.object_permission?.mcp_tool_permissions ?? {},
});

/**
 * Always includes every MCP key (empty when cleared) so removals persist;
 * the proxy merges object_permission per key, leaving non-MCP grants untouched.
 */
export const buildMcpObjectPermission = (values: any) => ({
  mcp_servers: values.allowed_mcp_servers_and_groups?.servers ?? [],
  mcp_access_groups: values.allowed_mcp_servers_and_groups?.accessGroups ?? [],
  mcp_toolsets: values.allowed_mcp_servers_and_groups?.toolsets ?? [],
  mcp_tool_permissions: values.mcp_tool_permissions ?? {},
});

/**
 * Parse agent data for form fields
 */
export const parseAgentForForm = (agent: any) => {
  const card = agent.agent_card_params ?? {};
  const skills =
    card.skills?.map((skill: any) => ({
      ...skill,
      tags: skill.tags,
      examples: skill.examples || [],
    })) || [];

  return {
    agent_name: agent.agent_name,
    name: card.name || agent.agent_name,
    description: card.description,
    url: card.url,
    version: card.version,
    protocolVersion: card.protocolVersion,
    streaming: card.capabilities?.streaming,
    pushNotifications: card.capabilities?.pushNotifications,
    stateTransitionHistory: card.capabilities?.stateTransitionHistory,
    skills: skills,
    iconUrl: card.iconUrl,
    documentationUrl: card.documentationUrl,
    supportsAuthenticatedExtendedCard: card.supportsAuthenticatedExtendedCard,
    model: agent.litellm_params?.model,
    make_public: agent.litellm_params?.make_public,
    cost_per_query: agent.litellm_params?.cost_per_query,
    input_cost_per_token: agent.litellm_params?.input_cost_per_token,
    output_cost_per_token: agent.litellm_params?.output_cost_per_token,
    tpm_limit: agent.tpm_limit,
    rpm_limit: agent.rpm_limit,
    session_tpm_limit: agent.session_tpm_limit,
    session_rpm_limit: agent.session_rpm_limit,
    // static_headers: {key: value} → [{header, value}, ...]
    static_headers: agent.static_headers
      ? Object.entries(agent.static_headers as Record<string, string>).map(([header, value]) => ({
          header,
          value,
        }))
      : [],
    // extra_headers: already an array of strings
    extra_headers: agent.extra_headers ?? [],
    kill_switch: parseKillSwitchForForm(agent.kill_switch),
    ...parseMcpPermissionsForForm(agent),
    ...parseAccessGroupIdsForForm(agent),
  };
};
