const asset_logos_folder = '/ui/assets/logos/';

interface CallbackConfig {
  id: string;                    // Internal callback name (e.g., "arize", "custom_callback_api")
  displayName: string;           // User-facing name (e.g., "Arize", "Custom Callback API")
  logo: string;                  // Logo path
  supports_key_team_logging: boolean;
  dynamic_params: Record<string, "text" | "password" | "select" | "upload" | "number">;
  description: string;
}

// Single source of truth for ALL callback configurations
export const CALLBACK_CONFIGS: CallbackConfig[] = [
  {
    id: "arize",
    displayName: "Arize",
    logo: `${asset_logos_folder}arize.png`,
    supports_key_team_logging: true,
    dynamic_params: {
      "arize_api_key": "password",
      "arize_space_key": "password",
    },
    description: "Arize Logging Integration"
  },
  {
    id: "braintrust",
    displayName: "Braintrust",
    logo: `${asset_logos_folder}braintrust.png`,
    supports_key_team_logging: false,
    dynamic_params: {
      "braintrust_api_key": "password",
      "braintrust_project_name": "text"
    },
    description: "Braintrust Logging Integration"
  },
  {
    id: "custom_callback_api",
    displayName: "Custom Callback API",
    logo: `${asset_logos_folder}custom.svg`,
    supports_key_team_logging: true,
    dynamic_params: {
      "custom_callback_api_url": "text",
      "custom_callback_api_headers": "text"
    },
    description: "Custom Callback API Logging Integration"
  },
  {
    id: "datadog",
    displayName: "Datadog",
    logo: `${asset_logos_folder}datadog.png`,
    supports_key_team_logging: false,
    dynamic_params: {
      "dd_api_key": "password",
      "dd_site": "text"
    },
    description: "Datadog Logging Integration"
  },
  {
    id: "lago",
    displayName: "Lago",
    logo: `${asset_logos_folder}lago.svg`,
    supports_key_team_logging: false,
    dynamic_params: {
      "lago_api_url": "text",
      "lago_api_key": "password"
    },
    description: "Lago Billing Logging Integration"
  },
  {
    id: "langfuse",
    displayName: "Langfuse",
    logo: `${asset_logos_folder}langfuse.png`,
    supports_key_team_logging: true,
    dynamic_params: {
      "langfuse_public_key": "text",
      "langfuse_secret_key": "password",
      "langfuse_host": "text"
    },
    description: "Langfuse v2 Logging Integration"
  },
  {
    id: "langfuse_otel",
    displayName: "Langfuse OTEL",
    logo: `${asset_logos_folder}langfuse.png`,
    supports_key_team_logging: true,
    dynamic_params: {
      "langfuse_public_key": "text",
      "langfuse_secret_key": "password",
      "langfuse_host": "text"
    },
    description: "Langfuse v3 OTEL Logging Integration"
  },
  {
    id: "langsmith",
    displayName: "LangSmith",
    logo: `${asset_logos_folder}langsmith.png`,
    supports_key_team_logging: true,
    dynamic_params: {
      "langsmith_api_key": "password",
      "langsmith_project": "text",
      "langsmith_base_url": "text",
      "langsmith_sampling_rate": "number"
    },
    description: "Langsmith Logging Integration"
  },
  {
    id: "openmeter",
    displayName: "OpenMeter",
    logo: `${asset_logos_folder}openmeter.png`,
    supports_key_team_logging: false,
    dynamic_params: {
      "openmeter_api_key": "password",
      "openmeter_base_url": "text"
    },
    description: "OpenMeter Logging Integration"
  },
  {
    id: "otel",
    displayName: "Open Telemetry",
    logo: `${asset_logos_folder}otel.png`,
    supports_key_team_logging: false,
    dynamic_params: {
      "otel_endpoint": "text",
      "otel_headers": "text"
    },
    description: "OpenTelemetry Logging Integration"
  },
  {
    id: "s3",
    displayName: "S3",
    logo: `${asset_logos_folder}aws.svg`,
    supports_key_team_logging: false,
    dynamic_params: {
      "s3_bucket_name": "text",
      "aws_access_key_id": "password",
      "aws_secret_access_key": "password",
      "aws_region": "text"
    },
    description: "S3 Bucket (AWS) Logging Integration"
  }
];

// Utility functions for easy access
export const getCallbackById = (id: string): CallbackConfig | undefined => {
  return CALLBACK_CONFIGS.find(callback => callback.id === id);
};

export const getCallbackByDisplayName = (displayName: string): CallbackConfig | undefined => {
  return CALLBACK_CONFIGS.find(callback => callback.displayName === displayName);
};
