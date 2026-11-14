import type { ParseKeys } from "i18next";
import arizeLogo from "../../public/assets/logos/arize.png";
import awsLogo from "../../public/assets/logos/aws.svg";
import braintrustLogo from "../../public/assets/logos/braintrust.png";
import datadogLogo from "../../public/assets/logos/datadog.png";
import galileoLogo from "../../public/assets/logos/galileo.ico";
import lagoLogo from "../../public/assets/logos/lago.svg";
import langfuseLogo from "../../public/assets/logos/langfuse.png";
import langsmithLogo from "../../public/assets/logos/langsmith.png";
import newrelicLogo from "../../public/assets/logos/newrelic.png";
import openmeterLogo from "../../public/assets/logos/openmeter.png";
import otelLogo from "../../public/assets/logos/otel.png";
import pointfiveLogo from "../../public/assets/logos/pointfive.png";
import signozLogo from "../../public/assets/logos/signoz.svg";
import databricksLogo from "../../public/assets/logos/databricks.svg";

interface CallbackConfig {
  id: string;
  displayName: string;
  logo?: string;
  supports_key_team_logging: boolean;
  dynamic_params: Record<string, "text" | "password" | "select" | "upload" | "number">;
  dynamic_param_options?: Record<string, readonly string[]>;
  success_event_params?: string[];
  descriptionKey: ParseKeys;
}

export const CALLBACK_CONFIGS: CallbackConfig[] = [
  {
    id: "arize",
    displayName: "Arize",
    logo: arizeLogo.src,
    supports_key_team_logging: true,
    dynamic_params: {
      arize_api_key: "password",
      arize_space_id: "password",
      arize_success_sampling_rate: "number",
      arize_error_sampling_rate: "number",
      arize_otlp_protocol: "select",
    },
    dynamic_param_options: {
      arize_otlp_protocol: ["grpc", "http/protobuf"],
    },
    success_event_params: ["arize_otlp_protocol"],
    descriptionKey: "callbackInfo.descriptions.arize",
  },
  {
    id: "braintrust",
    displayName: "Braintrust",
    logo: braintrustLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      braintrust_api_key: "password",
      braintrust_project_name: "text",
    },
    descriptionKey: "callbackInfo.descriptions.braintrust",
  },
  {
    id: "custom_callback_api",
    displayName: "Custom Callback API",
    supports_key_team_logging: true,
    dynamic_params: {
      custom_callback_api_url: "text",
      custom_callback_api_headers: "text",
    },
    descriptionKey: "callbackInfo.descriptions.customCallbackApi",
  },
  {
    id: "galileo",
    displayName: "Galileo",
    logo: galileoLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      GALILEO_API_KEY: "password",
      GALILEO_PROJECT_ID: "text",
      GALILEO_LOG_STREAM_ID: "text",
      GALILEO_BASE_URL: "text",
      GALILEO_USERNAME: "text",
      GALILEO_PASSWORD: "password",
    },
    descriptionKey: "callbackInfo.descriptions.galileo",
  },
  {
    id: "datadog",
    displayName: "Datadog",
    logo: datadogLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      dd_api_key: "password",
      dd_site: "text",
    },
    descriptionKey: "callbackInfo.descriptions.datadog",
  },
  {
    id: "newrelic",
    displayName: "New Relic",
    logo: newrelicLogo.src,
    supports_key_team_logging: true,
    dynamic_params: {
      newrelic_api_key: "password",
      newrelic_region: "text",
    },
    descriptionKey: "callbackInfo.descriptions.newrelic",
  },
  {
    id: "lago",
    displayName: "Lago",
    logo: lagoLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      lago_api_url: "text",
      lago_api_key: "password",
    },
    descriptionKey: "callbackInfo.descriptions.lago",
  },
  {
    id: "langfuse",
    displayName: "Langfuse",
    logo: langfuseLogo.src,
    supports_key_team_logging: true,
    dynamic_params: {
      langfuse_public_key: "text",
      langfuse_secret_key: "password",
      langfuse_host: "text",
      langfuse_environment: "text",
    },
    descriptionKey: "callbackInfo.descriptions.langfuse",
  },
  {
    id: "langfuse_otel",
    displayName: "Langfuse OTEL",
    logo: langfuseLogo.src,
    supports_key_team_logging: true,
    dynamic_params: {
      langfuse_public_key: "text",
      langfuse_secret_key: "password",
      langfuse_host: "text",
      langfuse_environment: "text",
      langfuse_span_scope: "select",
    },
    dynamic_param_options: {
      langfuse_span_scope: ["full", "llm_only"],
    },
    descriptionKey: "callbackInfo.descriptions.langfuseOtel",
  },
  {
    id: "langsmith",
    displayName: "LangSmith",
    logo: langsmithLogo.src,
    supports_key_team_logging: true,
    dynamic_params: {
      langsmith_api_key: "password",
      langsmith_project: "text",
      langsmith_base_url: "text",
      langsmith_sampling_rate: "number",
    },
    descriptionKey: "callbackInfo.descriptions.langsmith",
  },
  {
    id: "openmeter",
    displayName: "OpenMeter",
    logo: openmeterLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      openmeter_api_key: "password",
      openmeter_base_url: "text",
    },
    descriptionKey: "callbackInfo.descriptions.openmeter",
  },
  {
    id: "otel",
    displayName: "Open Telemetry",
    logo: otelLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      otel_endpoint: "text",
      otel_headers: "text",
      otel_exporter_otlp_protocol: "select",
    },
    descriptionKey: "callbackInfo.descriptions.otel",
  },
  {
    id: "pointfive",
    displayName: "PointFive",
    logo: pointfiveLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      POINTFIVE_API_KEY: "password",
      POINTFIVE_API_URL: "text",
    },
    descriptionKey: "callbackInfo.descriptions.pointfive",
  },
  {
    id: "zerobus",
    displayName: "Databricks Zerobus",
    logo: databricksLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      ZEROBUS_WORKSPACE_URL: "text",
      ZEROBUS_SERVER_ENDPOINT: "text",
      ZEROBUS_CLIENT_ID: "text",
      ZEROBUS_CLIENT_SECRET: "password",
      ZEROBUS_TABLE_NAME: "text",
    },
    descriptionKey: "callbackInfo.descriptions.zerobus",
  },
  {
    id: "s3",
    displayName: "S3",
    logo: awsLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      s3_bucket_name: "text",
      aws_access_key_id: "password",
      aws_secret_access_key: "password",
      aws_region: "text",
    },
    descriptionKey: "callbackInfo.descriptions.s3",
  },
  {
    id: "signoz",
    displayName: "SigNoz",
    logo: signozLogo.src,
    supports_key_team_logging: true,
    dynamic_params: {
      signoz_ingestion_endpoint: "text",
      signoz_ingestion_key: "password",
    },
    descriptionKey: "callbackInfo.descriptions.signoz",
  },
  {
    id: "SQS",
    displayName: "SQS",
    logo: awsLogo.src,
    supports_key_team_logging: false,
    dynamic_params: {
      sqs_queue_url: "text",
      aws_access_key_id: "password",
      aws_secret_access_key: "password",
      aws_region: "text",
    },
    descriptionKey: "callbackInfo.descriptions.sqs",
  },
];

// Create callbackInfo object mapping display names to config objects
export const callbackInfo: Record<string, CallbackConfig> = CALLBACK_CONFIGS.reduce(
  (acc, config) => {
    acc[config.displayName] = config;
    return acc;
  },
  {} as Record<string, CallbackConfig>,
);

// Create callback_map mapping display names to internal IDs
export const callback_map: Record<string, string> = CALLBACK_CONFIGS.reduce(
  (acc, config) => {
    acc[config.displayName] = config.id;
    return acc;
  },
  {} as Record<string, string>,
);

// create reverse_callback_map to map internal IDs to display names
export const reverse_callback_map: Record<string, string> = CALLBACK_CONFIGS.reduce(
  (acc, config) => {
    acc[config.id] = config.displayName;
    return acc;
  },
  {} as Record<string, string>,
);

// Function to map display names to internal names
export const mapDisplayToInternalNames = (displayNames: string[]): string[] => {
  return displayNames.map((name) => callback_map[name] || name);
};

// Function to map internal names to display names
export const mapInternalToDisplayNames = (internalNames: string[]): string[] => {
  return internalNames.map((name) => reverse_callback_map[name] || name);
};

// Utility functions for easy access
export const getCallbackById = (id: string): CallbackConfig | undefined => {
  return CALLBACK_CONFIGS.find((callback) => callback.id === id);
};

export const getCallbackByDisplayName = (displayName: string): CallbackConfig | undefined => {
  return CALLBACK_CONFIGS.find((callback) => callback.displayName === displayName);
};
