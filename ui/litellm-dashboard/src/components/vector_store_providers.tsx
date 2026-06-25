import type { TFunction } from "i18next";
import { resolveLogoSrc } from "@/lib/assetPaths";

export enum VectorStoreProviders {
  Bedrock = "Amazon Bedrock",
  S3Vectors = "Amazon S3 Vectors",
  PgVector = "PostgreSQL pgvector (LiteLLM Connector)",
  VertexRagEngine = "Vertex AI RAG Engine",
  VertexAiSearch = "Vertex AI Search",
  OpenAI = "OpenAI",
  Azure = "Azure OpenAI",
  Milvus = "Milvus",
}

export const vectorStoreProviderMap: Record<string, string> = {
  Bedrock: "bedrock",
  PgVector: "pg_vector",
  VertexRagEngine: "vertex_ai",
  VertexAiSearch: "vertex_ai/search_api",
  OpenAI: "openai",
  Azure: "azure",
  Milvus: "milvus",
  S3Vectors: "s3_vectors",
};

const asset_logos_folder = "/ui/assets/logos/";

export const vectorStoreProviderLogoMap: Record<string, string> = {
  [VectorStoreProviders.Bedrock]: `${asset_logos_folder}bedrock.svg`,
  [VectorStoreProviders.PgVector]: `${asset_logos_folder}postgresql.svg`, // Fallback to a generic database icon if needed
  [VectorStoreProviders.VertexRagEngine]: `${asset_logos_folder}google.svg`,
  [VectorStoreProviders.VertexAiSearch]: `${asset_logos_folder}google.svg`,
  [VectorStoreProviders.OpenAI]: `${asset_logos_folder}openai_small.svg`,
  [VectorStoreProviders.Azure]: `${asset_logos_folder}microsoft_azure.svg`,
  [VectorStoreProviders.Milvus]: `${asset_logos_folder}milvus.svg`,
  [VectorStoreProviders.S3Vectors]: `${asset_logos_folder}s3_vector.png`,
};

// Define field types for provider-specific configurations
export interface VectorStoreFieldConfig {
  name: string;
  label: string;
  tooltip: string;
  placeholder?: string;
  required: boolean;
  type?: "text" | "password" | "select";
  options?: { value: string; label: string }[];
  initialValue?: string;
}

export const getVectorStoreProviderFields = (t: TFunction): Record<string, VectorStoreFieldConfig[]> => ({
  bedrock: [],
  pg_vector: [
    {
      name: "api_base",
      label: t("vectorStoreProviders.pgVector.apiBaseLabel"),
      tooltip: t("vectorStoreProviders.pgVector.apiBaseTooltip"),
      placeholder: "http://your-deployed-server:8000",
      required: true,
      type: "text",
    },
    {
      name: "api_key",
      label: t("vectorStoreProviders.pgVector.apiKeyLabel"),
      tooltip: t("vectorStoreProviders.pgVector.apiKeyTooltip"),
      placeholder: "your-deployed-api-key",
      required: true,
      type: "password",
    },
  ],
  vertex_rag_engine: [],
  "vertex_ai/search_api": [
    {
      name: "vertex_project",
      label: t("vectorStoreProviders.vertexAiSearch.vertexProjectLabel"),
      tooltip: t("vectorStoreProviders.vertexAiSearch.vertexProjectTooltip"),
      placeholder: "my-gcp-project-id",
      required: true,
      type: "text",
    },
    {
      name: "vertex_location",
      label: t("vectorStoreProviders.vertexAiSearch.vertexLocationLabel"),
      tooltip: t("vectorStoreProviders.vertexAiSearch.vertexLocationTooltip"),
      required: true,
      type: "select",
      options: [
        { value: "global", label: "global" },
        { value: "us", label: "us" },
        { value: "eu", label: "eu" },
      ],
      initialValue: "global",
    },
    {
      name: "vertex_collection_id",
      label: t("vectorStoreProviders.vertexAiSearch.vertexCollectionIdLabel"),
      tooltip: t("vectorStoreProviders.vertexAiSearch.vertexCollectionIdTooltip"),
      placeholder: t("vectorStoreProviders.vertexAiSearch.vertexCollectionIdPlaceholder"),
      required: false,
      type: "text",
    },
    {
      name: "vertex_engine_id",
      label: t("vectorStoreProviders.vertexAiSearch.vertexEngineIdLabel"),
      tooltip: t("vectorStoreProviders.vertexAiSearch.vertexEngineIdTooltip"),
      placeholder: t("vectorStoreProviders.vertexAiSearch.vertexEngineIdPlaceholder"),
      required: false,
      type: "text",
    },
  ],
  openai: [
    {
      name: "api_key",
      label: t("vectorStoreProviders.openai.apiKeyLabel"),
      tooltip: t("vectorStoreProviders.openai.apiKeyTooltip"),
      placeholder: "sk-...",
      required: true,
      type: "password",
    },
  ],
  azure: [
    {
      name: "api_key",
      label: t("vectorStoreProviders.azure.apiKeyLabel"),
      tooltip: t("vectorStoreProviders.azure.apiKeyTooltip"),
      placeholder: "your-azure-api-key",
      required: true,
      type: "password",
    },
    {
      name: "api_base",
      label: t("vectorStoreProviders.azure.apiBaseLabel"),
      tooltip: t("vectorStoreProviders.azure.apiBaseTooltip"),
      placeholder: "https://your-resource.openai.azure.com/",
      required: true,
      type: "text",
    },
  ],
  milvus: [
    {
      name: "api_key",
      label: t("vectorStoreProviders.milvus.apiKeyLabel"),
      tooltip: t("vectorStoreProviders.milvus.apiKeyTooltip"),
      placeholder: t("vectorStoreProviders.milvus.apiKeyPlaceholder"),
      required: true,
      type: "password",
    },
    {
      name: "api_base",
      label: t("vectorStoreProviders.milvus.apiBaseLabel"),
      tooltip: t("vectorStoreProviders.milvus.apiBaseTooltip"),
      placeholder: "https://your-milvus-endpoint.com/",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: t("vectorStoreProviders.milvus.embeddingModelLabel"),
      tooltip: t("vectorStoreProviders.milvus.embeddingModelTooltip"),
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
  ],
  s3_vectors: [
    {
      name: "vector_bucket_name",
      label: t("vectorStoreProviders.s3Vectors.vectorBucketNameLabel"),
      tooltip: t("vectorStoreProviders.s3Vectors.vectorBucketNameTooltip"),
      placeholder: "my-vector-bucket",
      required: true,
      type: "text",
    },
    {
      name: "index_name",
      label: t("vectorStoreProviders.s3Vectors.indexNameLabel"),
      tooltip: t("vectorStoreProviders.s3Vectors.indexNameTooltip"),
      placeholder: "my-vector-index",
      required: false,
      type: "text",
    },
    {
      name: "aws_region_name",
      label: t("vectorStoreProviders.s3Vectors.awsRegionLabel"),
      tooltip: t("vectorStoreProviders.s3Vectors.awsRegionTooltip"),
      placeholder: "us-west-2",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: t("vectorStoreProviders.s3Vectors.embeddingModelLabel"),
      tooltip: t("vectorStoreProviders.s3Vectors.embeddingModelTooltip"),
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
  ],
});

export const getVectorStoreProviderLogoAndName = (providerValue: string): { logo: string; displayName: string } => {
  if (!providerValue) {
    return { logo: "", displayName: "-" };
  }

  // Find the enum key by matching vectorStoreProviderMap values
  const enumKey = Object.keys(vectorStoreProviderMap).find(
    (key) => vectorStoreProviderMap[key].toLowerCase() === providerValue.toLowerCase(),
  );

  if (!enumKey) {
    return { logo: "", displayName: providerValue };
  }

  // Get the display name from VectorStoreProviders enum and logo from map
  const displayName = VectorStoreProviders[enumKey as keyof typeof VectorStoreProviders];
  const logo = resolveLogoSrc(vectorStoreProviderLogoMap[displayName as keyof typeof vectorStoreProviderLogoMap]) ?? "";

  return { logo, displayName };
};

export const getProviderSpecificFields = (providerValue: string, t: TFunction): VectorStoreFieldConfig[] => {
  return getVectorStoreProviderFields(t)[providerValue] || [];
};
