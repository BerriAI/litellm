export enum Callbacks {
  Braintrust = "Braintrust",
  CustomCallbackAPI = "Custom Callback API",
  Datadog = "Datadog",
  Langfuse = "Langfuse",
  LangfuseOtel = "LangfuseOtel",
  LangSmith = "LangSmith",
  Lago = "Lago",
  OpenMeter = "OpenMeter",
  OTel = "Open Telemetry",
  S3 = "S3",
  Arize = "Arize",
}

export const callback_map: Record<string, string> = {
  Braintrust: "braintrust",
  CustomCallbackAPI: "custom_callback_api",
  Datadog: "datadog",
  Langfuse: "langfuse",
  LangfuseOtel: "langfuse_otel",
  LangSmith: "langsmith",
  Lago: "lago",
  OpenMeter: "openmeter",
  OTel: "otel",
  S3: "s3",
  Arize: "arize",
}

// Reverse mapping from internal values to display names
export const reverse_callback_map: Record<string, string> = Object.fromEntries(
  Object.entries(callback_map).map(([key, value]) => [value, key])
);

// Utility function to convert internal callback values to display names
export const mapInternalToDisplayNames = (internalValues: string[]): string[] => {
  return internalValues.map(value => reverse_callback_map[value] || value);
};

// Utility function to convert display names to internal callback values
export const mapDisplayToInternalNames = (displayValues: string[]): string[] => {
  return displayValues.map(value => callback_map[value] || value);
};

const asset_logos_folder = '/ui/assets/logos/';

interface CallbackInfo {
  logo: string;
  supports_key_team_logging: boolean;
  dynamic_params: Record<string, "text" | "password" | "select" | "upload" | "number">;
  description: string | null;
}

export const callbackInfo: Record<string, CallbackInfo> = {
  [Callbacks.Langfuse]: {
    logo: `${asset_logos_folder}langfuse.png`,
    supports_key_team_logging: true,
    dynamic_params: {
        "langfuse_public_key": "text",
        "langfuse_secret_key": "password",
        "langfuse_host": "text"
    },
    description: "Langfuse v2 Logging Integration"
    },
    [Callbacks.LangfuseOtel]: {
      logo: `${asset_logos_folder}langfuse.png`,
      supports_key_team_logging: true,
      dynamic_params: {
        "langfuse_public_key": "text",
        "langfuse_secret_key": "password",
        "langfuse_host": "text"
      },
      description: "Langfuse v3 OTEL Logging Integration"
    },
    [Callbacks.Arize]: {
      logo: `${asset_logos_folder}arize.png`,
      supports_key_team_logging: true,
      dynamic_params: {
        "arize_api_key": "password",
        "arize_space_id": "text",
      },
      description: "Arize Logging Integration"
    },
    [Callbacks.LangSmith]: {
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
    [Callbacks.Braintrust]: {
        logo: `${asset_logos_folder}braintrust.png`,
        supports_key_team_logging: false,
        dynamic_params: {},
        description: "Braintrust Logging Integration"
    },
    [Callbacks.CustomCallbackAPI]: {
        logo: `${asset_logos_folder}custom.svg`,
        supports_key_team_logging: true,
        dynamic_params: {},
        description: "Custom Callback API Logging Integration"
    },
    [Callbacks.Datadog]: {
        logo: `${asset_logos_folder}datadog.png`,
        supports_key_team_logging: false,
        dynamic_params: {},
        description: "Datadog Logging Integration"
    },
    [Callbacks.Lago]: {
        logo: `${asset_logos_folder}lago.svg`,
        supports_key_team_logging: false,
        dynamic_params: {},
        description: "Lago Billing Logging Integration"
    },
    [Callbacks.OpenMeter]: {
        logo: `${asset_logos_folder}openmeter.png`,
        supports_key_team_logging: false,
        dynamic_params: {},
        description: "OpenMeter Logging Integration"
    },
    [Callbacks.OTel]: {
        logo: `${asset_logos_folder}otel.png`,
        supports_key_team_logging: false,
        dynamic_params: {},
        description: "OpenTelemetry Logging Integration"
    },
    [Callbacks.S3]: {
        logo: `${asset_logos_folder}aws.svg`,
        supports_key_team_logging: false,
        dynamic_params: {},
        description: "S3 Bucket (AWS) Logging Integration"
    }
};
  