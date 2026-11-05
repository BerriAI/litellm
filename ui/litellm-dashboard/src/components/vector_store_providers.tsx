import type { ParseKeys } from "i18next";
import { getProviderLogoAndName, Providers, providerLogoMap } from "@/components/provider_info_helpers";
import awsLogo from "../../public/assets/logos/aws.svg";
import milvusLogo from "../../public/assets/logos/milvus.svg";
import mongodbLogo from "../../public/assets/logos/mongodb.svg";
import postgresqlLogo from "../../public/assets/logos/postgresql.svg";
import s3VectorLogo from "../../public/assets/logos/s3_vector.png";
import valkeyLogo from "../../public/assets/logos/valkey.svg";

export enum VectorStoreProviders {
  Bedrock = "Amazon Bedrock",
  S3Vectors = "Amazon S3 Vectors",
  OpenSearch = "Amazon OpenSearch",
  PgVector = "PostgreSQL pgvector (LiteLLM Connector)",
  VertexRagEngine = "Vertex AI RAG Engine",
  VertexAiSearch = "Vertex AI Search",
  OpenAI = "OpenAI",
  Azure = "Azure OpenAI",
  Milvus = "Milvus",
  MongoDB = "MongoDB (BETA)",
  Valkey = "Valkey",
}

export const vectorStoreProviderMap: Record<string, string> = {
  Bedrock: "bedrock",
  PgVector: "pg_vector",
  VertexRagEngine: "vertex_ai",
  VertexAiSearch: "vertex_ai/search_api",
  OpenAI: "openai",
  Azure: "azure",
  Milvus: "milvus",
  MongoDB: "mongodb",
  S3Vectors: "s3_vectors",
  OpenSearch: "opensearch",
  Valkey: "valkey",
};

export const vectorStoreProviderLogoMap: Record<string, string> = {
  [VectorStoreProviders.Bedrock]: providerLogoMap[Providers.Bedrock] ?? "",
  [VectorStoreProviders.PgVector]: postgresqlLogo.src,
  [VectorStoreProviders.VertexRagEngine]: providerLogoMap[Providers.Vertex_AI] ?? "",
  [VectorStoreProviders.VertexAiSearch]: providerLogoMap[Providers.Vertex_AI] ?? "",
  [VectorStoreProviders.OpenAI]: providerLogoMap[Providers.OpenAI] ?? "",
  [VectorStoreProviders.Azure]: providerLogoMap[Providers.Azure] ?? "",
  [VectorStoreProviders.Milvus]: milvusLogo.src,
  [VectorStoreProviders.MongoDB]: mongodbLogo.src,
  [VectorStoreProviders.S3Vectors]: s3VectorLogo.src,
  [VectorStoreProviders.OpenSearch]: awsLogo.src,
  [VectorStoreProviders.Valkey]: valkeyLogo.src,
};

// Define field types for provider-specific configurations
export interface VectorStoreFieldConfig {
  name: string;
  label: ParseKeys;
  tooltip: ParseKeys;
  placeholder?: string;
  required: boolean;
  type?: "text" | "password" | "select";
  options?: { value: string; label: string }[];
  initialValue?: string;
}

// Provider-specific field configurations
export const vectorStoreProviderFields: Record<string, VectorStoreFieldConfig[]> = {
  bedrock: [],
  pg_vector: [
    {
      name: "api_base",
      label: "vectorStores.providerFields.labels.apiBase",
      tooltip: "vectorStores.providerFields.tooltips.pgVector.apiBase",
      placeholder: "http://your-deployed-server:8000",
      required: true,
      type: "text",
    },
    {
      name: "api_key",
      label: "vectorStores.providerFields.labels.apiKey",
      tooltip: "vectorStores.providerFields.tooltips.pgVector.apiKey",
      placeholder: "your-deployed-api-key",
      required: true,
      type: "password",
    },
  ],
  vertex_rag_engine: [],
  "vertex_ai/search_api": [
    {
      name: "vertex_project",
      label: "vectorStores.providerFields.labels.vertexProject",
      tooltip: "vectorStores.providerFields.tooltips.vertexAiSearch.project",
      placeholder: "my-gcp-project-id",
      required: true,
      type: "text",
    },
    {
      name: "vertex_location",
      label: "vectorStores.providerFields.labels.vertexLocation",
      tooltip: "vectorStores.providerFields.tooltips.vertexAiSearch.location",
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
      label: "vectorStores.providerFields.labels.collectionIdOptional",
      tooltip: "vectorStores.providerFields.tooltips.vertexAiSearch.collectionId",
      placeholder: "e.g. my-custom-collection",
      required: false,
      type: "text",
    },
    {
      name: "vertex_engine_id",
      label: "vectorStores.providerFields.labels.engineIdOptional",
      tooltip: "vectorStores.providerFields.tooltips.vertexAiSearch.engineId",
      placeholder: "e.g. my-search-app_1234567890",
      required: false,
      type: "text",
    },
  ],
  openai: [
    {
      name: "api_key",
      label: "vectorStores.providerFields.labels.apiKey",
      tooltip: "vectorStores.providerFields.tooltips.openai.apiKey",
      placeholder: "sk-...",
      required: true,
      type: "password",
    },
  ],
  azure: [
    {
      name: "api_key",
      label: "vectorStores.providerFields.labels.apiKey",
      tooltip: "vectorStores.providerFields.tooltips.azure.apiKey",
      placeholder: "your-azure-api-key",
      required: true,
      type: "password",
    },
    {
      name: "api_base",
      label: "vectorStores.providerFields.labels.apiBase",
      tooltip: "vectorStores.providerFields.tooltips.azure.apiBase",
      placeholder: "https://your-resource.openai.azure.com/",
      required: true,
      type: "text",
    },
  ],
  milvus: [
    {
      name: "api_key",
      label: "vectorStores.providerFields.labels.apiKey",
      tooltip: "vectorStores.providerFields.tooltips.milvus.apiKey",
      placeholder: "username:password or api key",
      required: true,
      type: "password",
    },
    {
      name: "api_base",
      label: "vectorStores.providerFields.labels.apiBase",
      tooltip: "vectorStores.providerFields.tooltips.milvus.apiBase",
      placeholder: "https://your-milvus-endpoint.com/",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: "vectorStores.providerFields.labels.embeddingModel",
      tooltip: "vectorStores.providerFields.tooltips.milvus.embeddingModel",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
  ],
  mongodb: [
    {
      name: "api_base",
      label: "vectorStores.providerFields.labels.sidecarUrl",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.apiBase",
      placeholder: "http://127.0.0.1:8080",
      required: true,
      type: "text",
    },
    {
      name: "api_key",
      label: "vectorStores.providerFields.labels.sidecarApiKey",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.apiKey",
      placeholder: "Enter sidecar API key",
      required: true,
      type: "password",
    },
    {
      name: "mongodb_database",
      label: "vectorStores.providerFields.labels.database",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.database",
      placeholder: "sample_mflix",
      required: true,
      type: "text",
    },
    {
      name: "mongodb_collection",
      label: "vectorStores.providerFields.labels.collection",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.collection",
      placeholder: "embedded_movies",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: "vectorStores.providerFields.labels.embeddingModel",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.embeddingModel",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
    {
      name: "mongodb_embedding_field",
      label: "vectorStores.providerFields.labels.vectorFieldName",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.embeddingField",
      placeholder: "embedding",
      required: false,
      type: "text",
      initialValue: "embedding",
    },
    {
      name: "mongodb_text_field",
      label: "vectorStores.providerFields.labels.textField",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.textField",
      placeholder: "text",
      required: false,
      type: "text",
      initialValue: "text",
    },
    {
      name: "mongodb_num_candidates",
      label: "vectorStores.providerFields.labels.candidatesConsidered",
      tooltip: "vectorStores.providerFields.tooltips.mongodb.numCandidates",
      placeholder: "100",
      required: false,
      type: "text",
    },
  ],
  valkey: [
    {
      name: "valkey_host",
      label: "vectorStores.providerFields.labels.valkeyHost",
      tooltip: "vectorStores.providerFields.tooltips.valkey.host",
      placeholder: "my-valkey.example.com",
      required: true,
      type: "text",
    },
    {
      name: "valkey_port",
      label: "vectorStores.providerFields.labels.valkeyPort",
      tooltip: "vectorStores.providerFields.tooltips.valkey.port",
      placeholder: "6379",
      required: false,
      type: "text",
      initialValue: "6379",
    },
    {
      name: "valkey_password",
      label: "vectorStores.providerFields.labels.valkeyPassword",
      tooltip: "vectorStores.providerFields.tooltips.valkey.password",
      required: false,
      type: "password",
    },
    {
      name: "valkey_ssl",
      label: "vectorStores.providerFields.labels.useTls",
      tooltip: "vectorStores.providerFields.tooltips.valkey.ssl",
      required: false,
      type: "select",
      options: [
        { value: "false", label: "false" },
        { value: "true", label: "true" },
      ],
      initialValue: "false",
    },
    {
      name: "embedding_model",
      label: "vectorStores.providerFields.labels.embeddingModel",
      tooltip: "vectorStores.providerFields.tooltips.valkey.embeddingModel",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
    {
      name: "valkey_text_field",
      label: "vectorStores.providerFields.labels.textField",
      tooltip: "vectorStores.providerFields.tooltips.valkey.textField",
      placeholder: "text",
      required: false,
      type: "text",
      initialValue: "text",
    },
    {
      name: "valkey_embedding_field",
      label: "vectorStores.providerFields.labels.vectorFieldName",
      tooltip: "vectorStores.providerFields.tooltips.valkey.embeddingField",
      placeholder: "embedding",
      required: false,
      type: "text",
      initialValue: "embedding",
    },
  ],
  opensearch: [
    {
      name: "api_base",
      label: "vectorStores.providerFields.labels.opensearchEndpoint",
      tooltip: "vectorStores.providerFields.tooltips.opensearch.apiBase",
      placeholder: "https://abc123.us-east-1.aoss.amazonaws.com",
      required: true,
      type: "text",
    },
    {
      name: "aws_region_name",
      label: "vectorStores.providerFields.labels.awsRegionOptional",
      tooltip: "vectorStores.providerFields.tooltips.opensearch.region",
      placeholder: "us-east-1",
      required: false,
      type: "text",
    },
    {
      name: "embedding_model",
      label: "vectorStores.providerFields.labels.embeddingModel",
      tooltip: "vectorStores.providerFields.tooltips.opensearch.embeddingModel",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
    {
      name: "opensearch_embedding_field",
      label: "vectorStores.providerFields.labels.vectorFieldName",
      tooltip: "vectorStores.providerFields.tooltips.opensearch.embeddingField",
      placeholder: "embedding",
      required: false,
      type: "text",
      initialValue: "embedding",
    },
    {
      name: "opensearch_text_field",
      label: "vectorStores.providerFields.labels.textField",
      tooltip: "vectorStores.providerFields.tooltips.opensearch.textField",
      placeholder: "text",
      required: false,
      type: "text",
      initialValue: "text",
    },
    {
      name: "opensearch_index_name",
      label: "vectorStores.providerFields.labels.indexNameOptional",
      tooltip: "vectorStores.providerFields.tooltips.opensearch.indexName",
      placeholder: "my-knn-index",
      required: false,
      type: "text",
    },
  ],
  s3_vectors: [
    {
      name: "vector_bucket_name",
      label: "vectorStores.providerFields.labels.vectorBucketName",
      tooltip: "vectorStores.providerFields.tooltips.s3Vectors.bucketName",
      placeholder: "my-vector-bucket",
      required: true,
      type: "text",
    },
    {
      name: "index_name",
      label: "vectorStores.providerFields.labels.indexName",
      tooltip: "vectorStores.providerFields.tooltips.s3Vectors.indexName",
      placeholder: "my-vector-index",
      required: false,
      type: "text",
    },
    {
      name: "aws_region_name",
      label: "vectorStores.providerFields.labels.awsRegion",
      tooltip: "vectorStores.providerFields.tooltips.s3Vectors.region",
      placeholder: "us-west-2",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: "vectorStores.providerFields.labels.embeddingModel",
      tooltip: "vectorStores.providerFields.tooltips.s3Vectors.embeddingModel",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
  ],
};

export const getVectorStoreProviderLogoAndName = (providerValue: string): { logo: string; displayName: string } => {
  const enumKey = Object.keys(vectorStoreProviderMap).find(
    (key) => vectorStoreProviderMap[key].toLowerCase() === providerValue.toLowerCase(),
  );
  if (!enumKey) {
    return getProviderLogoAndName(providerValue);
  }
  const displayName = VectorStoreProviders[enumKey as keyof typeof VectorStoreProviders];
  return { logo: vectorStoreProviderLogoMap[displayName], displayName };
};

export const getProviderSpecificFields = (providerValue: string): VectorStoreFieldConfig[] => {
  return vectorStoreProviderFields[providerValue] || [];
};
