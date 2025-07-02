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

export const callbackLogoMap: Record<string, string> = {
    [Callbacks.Braintrust]: `${asset_logos_folder}braintrust.png`,
    [Callbacks.Datadog]: `${asset_logos_folder}datadog.png`,
    [Callbacks.Langfuse]: `${asset_logos_folder}langfuse.png`,
    [Callbacks.LangSmith]: `${asset_logos_folder}langsmith.png`,
    [Callbacks.Lago]: `${asset_logos_folder}lago.svg`,
    [Callbacks.OpenMeter]: `${asset_logos_folder}openmeter.png`,
    [Callbacks.OTel]: `${asset_logos_folder}otel.png`,
    [Callbacks.S3]: `${asset_logos_folder}aws.svg`,
    [Callbacks.Arize]: `${asset_logos_folder}arize.png`,
    [Callbacks.CustomCallbackAPI]: `${asset_logos_folder}custom.svg`,
};
