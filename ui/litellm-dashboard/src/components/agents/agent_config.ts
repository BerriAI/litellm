/**
 * Shared configuration for agent form fields
 * Used across create, view, and update operations
 */

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

export const AGENT_FORM_CONFIG: {
  basic: SectionConfig;
  skills: SectionConfig;
  capabilities: SectionConfig;
  optional: SectionConfig;
  litellm: SectionConfig;
} = {
  basic: {
    key: "basic",
    title: "Basic Information",
    defaultExpanded: true,
    fields: [
      {
        name: "name",
        label: "Display Name",
        type: "text",
        required: true,
        placeholder: "e.g., Customer Support Agent",
      },
      {
        name: "description",
        label: "Description",
        type: "textarea",
        required: true,
        placeholder: "Describe what this agent does...",
        rows: 3,
      },
      {
        name: "url",
        label: "URL",
        type: "url",
        required: true,
        placeholder: "http://localhost:9999/",
        tooltip: "Base URL where the agent is hosted",
      },
      {
        name: "version",
        label: "Version",
        type: "text",
        placeholder: "1.0.0",
        defaultValue: "1.0.0",
      },
      {
        name: "protocolVersion",
        label: "Protocol Version",
        type: "text",
        placeholder: "1.0",
        defaultValue: "1.0",
      },
    ],
  },
  skills: {
    key: "skills",
    title: "Skills",
    fields: [
      {
        name: "skills",
        label: "Skills",
        type: "list",
        defaultValue: [],
      },
    ],
  },
  capabilities: {
    key: "capabilities",
    title: "Capabilities",
    fields: [
      {
        name: "streaming",
        label: "Streaming",
        type: "switch",
        defaultValue: false,
      },
      {
        name: "pushNotifications",
        label: "Push Notifications",
        type: "switch",
      },
      {
        name: "stateTransitionHistory",
        label: "State Transition History",
        type: "switch",
      },
    ],
  },
  optional: {
    key: "optional",
    title: "Optional Settings",
    fields: [
      {
        name: "iconUrl",
        label: "Icon URL",
        type: "url",
        placeholder: "https://example.com/icon.png",
      },
      {
        name: "documentationUrl",
        label: "Documentation URL",
        type: "url",
        placeholder: "https://docs.example.com",
      },
      {
        name: "supportsAuthenticatedExtendedCard",
        label: "Supports Authenticated Extended Card",
        type: "switch",
      },
    ],
  },
  litellm: {
    key: "litellm",
    title: "LiteLLM Parameters",
    fields: [
      {
        name: "model",
        label: "Model (Optional)",
        type: "text",
      },
      {
        name: "make_public",
        label: "Make Public",
        type: "switch",
      },
    ],
  },
};

export const SKILL_FIELD_CONFIG = {
  id: {
    name: "id",
    label: "Skill ID",
    required: true,
    placeholder: "e.g., hello_world",
  },
  name: {
    name: "name",
    label: "Skill Name",
    required: true,
    placeholder: "e.g., Returns hello world",
  },
  description: {
    name: "description",
    label: "Description",
    required: true,
    placeholder: "What this skill does",
    rows: 2,
  },
  tags: {
    name: "tags",
    label: "Tags (comma-separated)",
    required: true,
    placeholder: "e.g., hello world, greeting",
  },
  examples: {
    name: "examples",
    label: "Examples (comma-separated)",
    placeholder: "e.g., hi, hello world",
  },
};

/**
 * Get default form values from configuration
 */
export const getDefaultFormValues = () => {
  const defaults: any = {
    defaultInputModes: ["text"],
    defaultOutputModes: ["text"],
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
      name: values.name,
      description: values.description,
      url: values.url,
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

  // Only add litellm_params if there are values
  if (values.model || values.make_public !== undefined) {
    agentData.litellm_params = {
      ...(values.model && { model: values.model }),
      ...(values.make_public !== undefined && { make_public: values.make_public }),
    };
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
  };
};
