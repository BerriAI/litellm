export interface AzureTextModerationCategory {
  name: string;
  description: string;
  enabled: boolean;
  severityThreshold?: number;
}

export interface AzureTextModerationConfigurationProps {
  selectedCategories: string[];
  globalSeverityThreshold: number;
  categorySpecificThresholds: { [key: string]: number };
  onCategorySelect: (category: string) => void;
  onGlobalSeverityChange: (threshold: number) => void;
  onCategorySeverityChange: (category: string, threshold: number) => void;
}

export interface AzureTextModerationGuardrail {
  guardrail_id: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
    categories?: string[];
    severity_threshold?: number;
    severity_threshold_by_category?: { [key: string]: number };
    [key: string]: any;
  };
  guardrail_info: Record<string, any> | null;
  created_at?: string;
  updated_at?: string;
}

export const AZURE_TEXT_MODERATION_CATEGORIES = [
  {
    name: "Hate",
    description: "Content that attacks or uses discriminatory language based on protected characteristics",
  },
  {
    name: "Sexual",
    description: "Content that describes sexual activity or other sexual content",
  },
  {
    name: "SelfHarm",
    description: "Content that promotes, encourages, or depicts acts of self-harm",
  },
  {
    name: "Violence",
    description: "Content that depicts death, violence, or physical injury",
  },
];

export const SEVERITY_LEVELS = [
  {
    value: 0,
    label: "Level 0 - Safe",
    description: "Content is appropriate and safe",
  },
  {
    value: 2,
    label: "Level 2 - Low",
    description: "Content may be inappropriate in some contexts",
  },
  {
    value: 4,
    label: "Level 4 - Medium",
    description: "Content is inappropriate and should be filtered",
  },
  {
    value: 6,
    label: "Level 6 - High",
    description: "Content is harmful and should be blocked",
  },
];
