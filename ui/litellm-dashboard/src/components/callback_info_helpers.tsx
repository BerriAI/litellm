export enum Callbacks {
  Braintrust = "Braintrust",
  CustomCallbackAPI = "Custom Callback API",
  Datadog = "Datagog",
  Langfuse = "Langfuse",
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
  LangSmith: "langsmith",
  Lago: "lago",
  OpenMeter: "openmeter",
  OTel: "otel",
  S3: "s3",
  Arize: "arize",
}

const asset_logos_folder = '/ui/assets/logos/';

interface CallbackInfo {
  logo: string;
  supports_key_team_logging: boolean;
  dynamic_params: Record<string, string>;
}

export const callbackInfo: Record<string, CallbackInfo> = {
    [Callbacks.Braintrust]: {
        logo: `${asset_logos_folder}braintrust.png`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.CustomCallbackAPI]: {
        logo: `${asset_logos_folder}custom.svg`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.Datadog]: {
        logo: `${asset_logos_folder}datadog.png`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.Langfuse]: {
        logo: `${asset_logos_folder}langfuse.png`,
        supports_key_team_logging: true,
        dynamic_params: {
            "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY", // [RECOMMENDED] reference key in proxy environment
            "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY", // [RECOMMENDED] reference key in proxy environment
            "langfuse_host": "https://cloud.langfuse.com"
        }
    },
    [Callbacks.LangSmith]: {
        logo: `${asset_logos_folder}langsmith.png`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.Lago]: {
        logo: `${asset_logos_folder}lago.svg`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.OpenMeter]: {
        logo: `${asset_logos_folder}openmeter.png`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.OTel]: {
        logo: `${asset_logos_folder}otel.png`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.S3]: {
        logo: `${asset_logos_folder}aws.svg`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
    [Callbacks.Arize]: {
        logo: `${asset_logos_folder}arize.png`,
        supports_key_team_logging: false,
        dynamic_params: {}
    },
};
  