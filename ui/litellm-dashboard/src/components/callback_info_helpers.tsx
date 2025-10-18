interface CallbackConfig {
  id: string;
  displayName: string;
  logo: string;
  supports_key_team_logging: boolean;
  dynamic_params: Record<string, "text" | "password" | "select" | "upload" | "number">;
  description: string;
}

const asset_logos_folder = '/ui/assets/logos/';

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

// Create callbackInfo object mapping display names to config objects
export const callbackInfo: Record<string, CallbackConfig> = CALLBACK_CONFIGS.reduce((acc, config) => {
  acc[config.displayName] = config;
  return acc;
}, {} as Record<string, CallbackConfig>);

// Create callback_map mapping display names to internal IDs
export const callback_map: Record<string, string> = CALLBACK_CONFIGS.reduce((acc, config) => {
  acc[config.displayName] = config.id;
  return acc;
}, {} as Record<string, string>);

// create reverse_callback_map to map internal IDs to display names
export const reverse_callback_map: Record<string, string> = CALLBACK_CONFIGS.reduce((acc, config) => {
  acc[config.id] = config.displayName;
  return acc;
}, {} as Record<string, string>);

// Function to map display names to internal names
export const mapDisplayToInternalNames = (displayNames: string[]): string[] => {
  return displayNames.map(name => callback_map[name] || name);
};

// Function to map internal names to display names
export const mapInternalToDisplayNames = (internalNames: string[]): string[] => {
  return internalNames.map(name => reverse_callback_map[name] || name);
};

// Utility functions for easy access
export const getCallbackById = (id: string): CallbackConfig | undefined => {
  return CALLBACK_CONFIGS.find(callback => callback.id === id);
};

export const getCallbackByDisplayName = (displayName: string): CallbackConfig | undefined => {
  return CALLBACK_CONFIGS.find(callback => callback.displayName === displayName);
};
