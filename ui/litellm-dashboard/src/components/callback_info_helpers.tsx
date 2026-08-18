import arizeLogo from "../../public/assets/logos/arize.png";
import awsLogo from "../../public/assets/logos/aws.svg";
import azureLogo from "../../public/assets/logos/microsoft_azure.svg";
import braintrustLogo from "../../public/assets/logos/braintrust.png";
import datadogLogo from "../../public/assets/logos/datadog.png";
import galileoLogo from "../../public/assets/logos/galileo.ico";
import googleLogo from "../../public/assets/logos/google.svg";
import lagoLogo from "../../public/assets/logos/lago.svg";
import langfuseLogo from "../../public/assets/logos/langfuse.png";
import langsmithLogo from "../../public/assets/logos/langsmith.png";
import openmeterLogo from "../../public/assets/logos/openmeter.png";
import otelLogo from "../../public/assets/logos/otel.png";

interface CallbackConfig {
  id: string;
  displayName: string;
  logo?: string;
  dynamic_params: Record<string, "text" | "password" | "select" | "upload" | "number">;
  description: string;
}

export const CALLBACK_CONFIGS: CallbackConfig[] = [
  {
    id: "arize",
    displayName: "Arize",
    logo: arizeLogo.src,
    dynamic_params: {
      arize_api_key: "password",
      arize_space_id: "password",
    },
    description: "Arize Logging Integration",
  },
  {
    id: "arize_phoenix",
    displayName: "Arize Phoenix",
    logo: arizeLogo.src,
    dynamic_params: {},
    description: "Arize Phoenix Logging Integration",
  },
  {
    id: "azure_storage",
    displayName: "Azure Blob Storage",
    logo: azureLogo.src,
    dynamic_params: {},
    description: "Azure Blob Storage Logging Integration",
  },
  {
    id: "braintrust",
    displayName: "Braintrust",
    logo: braintrustLogo.src,
    dynamic_params: {},
    description: "Braintrust Logging Integration",
  },
  {
    id: "custom_callback_api",
    displayName: "Custom Callback API",
    dynamic_params: {},
    description: "Custom Callback API Logging Integration",
  },
  {
    id: "datadog",
    displayName: "Datadog",
    logo: datadogLogo.src,
    dynamic_params: {
      dd_api_key: "password",
      dd_site: "text",
    },
    description: "Datadog Logging Integration",
  },
  {
    id: "datadog_llm_observability",
    displayName: "Datadog LLM Observability",
    logo: datadogLogo.src,
    dynamic_params: {},
    description: "Datadog LLM Observability Logging Integration",
  },
  {
    id: "galileo",
    displayName: "Galileo",
    logo: galileoLogo.src,
    dynamic_params: {},
    description: "Galileo AI Observability Integration",
  },
  {
    id: "gcs_bucket",
    displayName: "GCS Bucket",
    logo: googleLogo.src,
    dynamic_params: {
      gcs_bucket_name: "text",
      gcs_path_service_account: "text",
    },
    description: "Google Cloud Storage Bucket Logging Integration",
  },
  {
    id: "gcs_pubsub",
    displayName: "GCS Pub/Sub",
    logo: googleLogo.src,
    dynamic_params: {},
    description: "Google Cloud Pub/Sub Logging Integration",
  },
  {
    id: "lago",
    displayName: "Lago",
    logo: lagoLogo.src,
    dynamic_params: {},
    description: "Lago Billing Logging Integration",
  },
  {
    id: "langfuse",
    displayName: "Langfuse",
    logo: langfuseLogo.src,
    dynamic_params: {
      langfuse_public_key: "text",
      langfuse_secret_key: "password",
      langfuse_host: "text",
    },
    description: "Langfuse v2 Logging Integration",
  },
  {
    id: "langfuse_otel",
    displayName: "Langfuse OTEL",
    logo: langfuseLogo.src,
    dynamic_params: {
      langfuse_public_key: "text",
      langfuse_secret_key: "password",
      langfuse_host: "text",
    },
    description: "Langfuse v3 OTEL Logging Integration",
  },
  {
    id: "langsmith",
    displayName: "LangSmith",
    logo: langsmithLogo.src,
    dynamic_params: {
      langsmith_api_key: "password",
      langsmith_project: "text",
      langsmith_base_url: "text",
      langsmith_sampling_rate: "number",
    },
    description: "Langsmith Logging Integration",
  },
  {
    id: "mlflow",
    displayName: "MLflow",
    dynamic_params: {},
    description: "MLflow Logging Integration",
  },
  {
    id: "openmeter",
    displayName: "OpenMeter",
    logo: openmeterLogo.src,
    dynamic_params: {},
    description: "OpenMeter Logging Integration",
  },
  {
    id: "opik",
    displayName: "Opik",
    dynamic_params: {},
    description: "Comet Opik Logging Integration",
  },
  {
    id: "otel",
    displayName: "Open Telemetry",
    logo: otelLogo.src,
    dynamic_params: {},
    description: "OpenTelemetry Logging Integration",
  },
  {
    id: "posthog",
    displayName: "PostHog",
    dynamic_params: {
      posthog_api_key: "password",
      posthog_api_url: "text",
    },
    description: "PostHog Logging Integration",
  },
  {
    id: "s3",
    displayName: "S3",
    logo: awsLogo.src,
    dynamic_params: {},
    description: "S3 Bucket (AWS) Logging Integration",
  },
  {
    id: "aws_sqs",
    displayName: "SQS",
    logo: awsLogo.src,
    dynamic_params: {},
    description: "SQS Queue (AWS) Logging Integration",
  },
  {
    id: "weave_otel",
    displayName: "Weave",
    dynamic_params: {
      wandb_api_key: "password",
      weave_project_id: "text",
    },
    description: "Weights & Biases Weave Logging Integration",
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
