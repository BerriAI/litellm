export enum VectorStoreProviders {
  Bedrock = "Amazon Bedrock",
  S3Vectors = "Amazon S3 Vectors",
  PgVector = "PostgreSQL pgvector (LiteLLM Connector)",
  VertexRagEngine = "Vertex AI RAG Engine",
  OpenAI = "OpenAI",
  Azure = "Azure OpenAI",
  Milvus = "Milvus",
}

export const vectorStoreProviderMap: Record<string, string> = {
  Bedrock: "bedrock",
  PgVector: "pg_vector",
  VertexRagEngine: "vertex_ai",
  OpenAI: "openai",
  Azure: "azure",
  Milvus: "milvus",
  S3Vectors: "s3_vectors",
};

const asset_logos_folder = "../ui/assets/logos/";

export const vectorStoreProviderLogoMap: Record<string, string> = {
  [VectorStoreProviders.Bedrock]: `${asset_logos_folder}bedrock.svg`,
  [VectorStoreProviders.PgVector]: `${asset_logos_folder}postgresql.svg`, // Fallback to a generic database icon if needed
  [VectorStoreProviders.VertexRagEngine]: `${asset_logos_folder}google.svg`,
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
}

// Provider-specific field configurations
export const vectorStoreProviderFields: Record<string, VectorStoreFieldConfig[]> = {
  bedrock: [],
  pg_vector: [
    {
      name: "api_base",
      label: "API Base",
      tooltip: "Enter the base URL of your deployed litellm-pgvector server (e.g., http://your-server:8000)",
      placeholder: "http://your-deployed-server:8000",
      required: true,
      type: "text",
    },
    {
      name: "api_key",
      label: "API Key",
      tooltip: "Enter the API key from your deployed litellm-pgvector server",
      placeholder: "your-deployed-api-key",
      required: true,
      type: "password",
    },
  ],
  vertex_rag_engine: [],
  openai: [
    {
      name: "api_key",
      label: "API Key",
      tooltip: "Enter your OpenAI API key",
      placeholder: "sk-...",
      required: true,
      type: "password",
    },
  ],
  azure: [
    {
      name: "api_key",
      label: "API Key",
      tooltip: "Enter your Azure OpenAI API key",
      placeholder: "your-azure-api-key",
      required: true,
      type: "password",
    },
    {
      name: "api_base",
      label: "API Base",
      tooltip: "Enter your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)",
      placeholder: "https://your-resource.openai.azure.com/",
      required: true,
      type: "text",
    },
  ],
  milvus: [
    {
      name: "api_key",
      label: "API Key",
      tooltip:
        "To obtain a token, you should use a colon (:) to concatenate the username and password that you use to access your Milvus instance (e.g., username:password)",
      placeholder: "username:password or api key",
      required: true,
      type: "password",
    },
    {
      name: "api_base",
      label: "API Base",
      tooltip: "Enter your Milvus endpoint (e.g., https://your-milvus-endpoint.com/)",
      placeholder: "https://your-milvus-endpoint.com/",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: "Embedding Model",
      tooltip: "Select the embedding model to use",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
  ],
  s3_vectors: [
    {
      name: "vector_bucket_name",
      label: "Vector Bucket Name",
      tooltip: "S3 bucket name for vector storage (will be auto-created if it doesn't exist)",
      placeholder: "my-vector-bucket",
      required: true,
      type: "text",
    },
    {
      name: "index_name",
      label: "Index Name",
      tooltip: "Name for the vector index (optional, will be auto-generated if not provided)",
      placeholder: "my-vector-index",
      required: false,
      type: "text",
    },
    {
      name: "aws_region_name",
      label: "AWS Region",
      tooltip: "AWS region where the S3 bucket is located (e.g., us-west-2)",
      placeholder: "us-west-2",
      required: true,
      type: "text",
    },
    {
      name: "embedding_model",
      label: "Embedding Model",
      tooltip: "Select the embedding model to use for vector generation",
      placeholder: "text-embedding-3-small",
      required: true,
      type: "select",
    },
  ],
};

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
  const logo = vectorStoreProviderLogoMap[displayName as keyof typeof vectorStoreProviderLogoMap];

  return { logo, displayName };
};

export const getProviderSpecificFields = (providerValue: string): VectorStoreFieldConfig[] => {
  return vectorStoreProviderFields[providerValue] || [];
};
