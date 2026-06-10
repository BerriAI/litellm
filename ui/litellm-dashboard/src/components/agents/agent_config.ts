/**
 * Shared configuration for agent form fields
 * Used across create, view, and update operations
 */
import type { TFunction } from "i18next";

export interface FieldConfig {
  name: string;
  label: string;
  type: "text" | "textarea" | "url" | "switch" | "list";
  required?: boolean;
  tooltip?: string;
  placeholder?: string;
  defaultValue?: any;
  rows?: number;
  validation?: any[];
}

export interface SectionConfig {
  key: string;
  title: string;
  fields: FieldConfig[];
  defaultExpanded?: boolean;
}

export const getAgentFormConfig = (
  t: TFunction,
): {
  basic: SectionConfig;
  skills: SectionConfig;
  capabilities: SectionConfig;
  optional: SectionConfig;
  litellm: SectionConfig;
  cost: SectionConfig;
  tracing: SectionConfig;
} => ({
  basic: {
    key: "basic",
    title: t("agentsPage.agentConfig.basicTitle"),
    defaultExpanded: true,
    fields: [
      {
        name: "name",
        label: t("agentsPage.agentConfig.displayNameLabel"),
        type: "text",
        required: true,
        placeholder: t("agentsPage.agentConfig.displayNamePlaceholder"),
      },
      {
        name: "description",
        label: t("common.description"),
        type: "textarea",
        required: true,
        placeholder: t("agentsPage.agentConfig.descriptionPlaceholder"),
        rows: 3,
      },
      {
        name: "url",
        label: t("agentsPage.agentConfig.urlLabel"),
        type: "url",
        required: false,
        placeholder: t("agentsPage.agentConfig.urlPlaceholder"),
        tooltip: t("agentsPage.agentConfig.urlTooltip"),
      },
      {
        name: "version",
        label: t("agentsPage.agentConfig.versionLabel"),
        type: "text",
        placeholder: "1.0.0",
        defaultValue: "1.0.0",
      },
      {
        name: "protocolVersion",
        label: t("agentsPage.agentConfig.protocolVersionLabel"),
        type: "text",
        placeholder: "1.0",
        defaultValue: "1.0",
      },
    ],
  },
  skills: {
    key: "skills",
    title: t("agentsPage.agentConfig.skillsTitle"),
    fields: [
      {
        name: "skills",
        label: t("agentsPage.agentConfig.skillsTitle"),
        type: "list",
        defaultValue: [],
      },
    ],
  },
  capabilities: {
    key: "capabilities",
    title: t("agentsPage.agentConfig.capabilitiesTitle"),
    fields: [
      {
        name: "streaming",
        label: t("agentsPage.agentConfig.streamingLabel"),
        type: "switch",
        defaultValue: false,
      },
      {
        name: "pushNotifications",
        label: t("agentsPage.agentConfig.pushNotificationsLabel"),
        type: "switch",
      },
      {
        name: "stateTransitionHistory",
        label: t("agentsPage.agentConfig.stateTransitionHistoryLabel"),
        type: "switch",
      },
    ],
  },
  optional: {
    key: "optional",
    title: t("agentsPage.agentConfig.optionalTitle"),
    fields: [
      {
        name: "iconUrl",
        label: t("agentsPage.agentConfig.iconUrlLabel"),
        type: "url",
        placeholder: "https://example.com/icon.png",
      },
      {
        name: "documentationUrl",
        label: t("agentsPage.agentConfig.documentationUrlLabel"),
        type: "url",
        placeholder: "https://docs.example.com",
      },
      {
        name: "supportsAuthenticatedExtendedCard",
        label: t("agentsPage.agentConfig.supportsAuthCardLabel"),
        type: "switch",
      },
    ],
  },
  litellm: {
    key: "litellm",
    title: t("agentsPage.agentConfig.litellmTitle"),
    fields: [
      {
        name: "model",
        label: t("agentsPage.agentConfig.modelOptionalLabel"),
        type: "text",
      },
      {
        name: "make_public",
        label: t("agentsPage.agentConfig.makePublicLabel"),
        type: "switch",
      },
    ],
  },
  cost: {
    key: "cost",
    title: t("agentsPage.agentConfig.costTitle"),
    fields: [
      {
        name: "cost_per_query",
        label: t("agentsPage.agentConfig.costPerQueryLabel"),
        type: "text",
        placeholder: "0.0",
        tooltip: t("agentsPage.agentConfig.costPerQueryTooltip"),
      },
      {
        name: "input_cost_per_token",
        label: t("agentsPage.agentConfig.inputCostPerTokenLabel"),
        type: "text",
        placeholder: "0.000001",
        tooltip: t("agentsPage.agentConfig.inputCostPerTokenTooltip"),
      },
      {
        name: "output_cost_per_token",
        label: t("agentsPage.agentConfig.outputCostPerTokenLabel"),
        type: "text",
        placeholder: "0.000002",
        tooltip: t("agentsPage.agentConfig.outputCostPerTokenTooltip"),
      },
    ],
  },
  tracing: {
    key: "tracing",
    title: t("agentsPage.agentConfig.tracingTitle"),
    fields: [
      {
        name: "enable_tracing",
        label: t("agentsPage.agentConfig.enableTracingLabel"),
        type: "switch",
        defaultValue: false,
        tooltip: t("agentsPage.agentConfig.enableTracingTooltip"),
      },
    ],
  },
});

export const getSkillFieldConfig = (t: TFunction) => ({
  id: {
    name: "id",
    label: t("agentsPage.agentConfig.skillIdLabel"),
    required: true,
    placeholder: t("agentsPage.agentConfig.skillIdPlaceholder"),
  },
  name: {
    name: "name",
    label: t("agentsPage.agentConfig.skillNameLabel"),
    required: true,
    placeholder: t("agentsPage.agentConfig.skillNamePlaceholder"),
  },
  description: {
    name: "description",
    label: t("common.description"),
    required: true,
    placeholder: t("agentsPage.agentConfig.skillDescriptionPlaceholder"),
    rows: 2,
  },
  tags: {
    name: "tags",
    label: t("agentsPage.agentConfig.skillTagsLabel"),
    required: true,
    placeholder: t("agentsPage.agentConfig.skillTagsPlaceholder"),
  },
  examples: {
    name: "examples",
    label: t("agentsPage.agentConfig.skillExamplesLabel"),
    placeholder: t("agentsPage.agentConfig.skillExamplesPlaceholder"),
  },
});

/**
 * Get default form values from configuration
 */
export const getDefaultFormValues = () => ({
  defaultInputModes: ["text"],
  defaultOutputModes: ["text"],
  version: "1.0.0",
  protocolVersion: "1.0",
  streaming: false,
  skills: [],
  enable_tracing: false,
});

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

  return agentData;
};

/**
 * Parse agent data for form fields
 */
export const parseAgentForForm = (agent: any) => {
  const skills =
    agent.agent_card_params?.skills?.map((skill: any) => ({
      ...skill,
      tags: skill.tags,
      examples: skill.examples || [],
    })) || [];

  return {
    agent_name: agent.agent_name,
    name: agent.agent_card_params?.name,
    description: agent.agent_card_params?.description,
    url: agent.agent_card_params?.url,
    version: agent.agent_card_params?.version,
    protocolVersion: agent.agent_card_params?.protocolVersion,
    streaming: agent.agent_card_params?.capabilities?.streaming,
    pushNotifications: agent.agent_card_params?.capabilities?.pushNotifications,
    stateTransitionHistory: agent.agent_card_params?.capabilities?.stateTransitionHistory,
    skills: skills,
    iconUrl: agent.agent_card_params?.iconUrl,
    documentationUrl: agent.agent_card_params?.documentationUrl,
    supportsAuthenticatedExtendedCard: agent.agent_card_params?.supportsAuthenticatedExtendedCard,
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
  };
};
