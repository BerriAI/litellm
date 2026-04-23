import React, { useState, useEffect } from "react";
import { Form, Select } from "antd";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import {
  fetchAvailableModels,
  ModelGroup,
} from "../playground/llm_calls/fetch_models";

interface S3VectorsConfigProps {
  accessToken: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  providerParams: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onParamsChange: (params: Record<string, any>) => void;
}

const S3VectorsConfig: React.FC<S3VectorsConfigProps> = ({
  accessToken,
  providerParams,
  onParamsChange,
}) => {
  const [embeddingModels, setEmbeddingModels] = useState<ModelGroup[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      setIsLoadingModels(true);
      try {
        const models = await fetchAvailableModels(accessToken);
        // Filter for embedding models only
        const embeddingOnly = models.filter((model) => model.mode === "embedding");
        setEmbeddingModels(embeddingOnly);
      } catch (error) {
        console.error("Error fetching embedding models:", error);
      } finally {
        setIsLoadingModels(false);
      }
    };

    loadModels();
  }, [accessToken]);

  const handleFieldChange = (fieldName: string, value: string) => {
    onParamsChange({
      ...providerParams,
      [fieldName]: value,
    });
  };

  return (
    <>
      <div className="mb-4 flex gap-2 items-start p-3 rounded-md bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 text-blue-800 dark:text-blue-200">
        <Info className="h-4 w-4 mt-0.5 shrink-0" />
        <div className="flex-1">
          <div className="font-semibold">AWS S3 Vectors Setup</div>
          <p className="text-sm mt-1">
            AWS S3 Vectors allows you to store and query vector embeddings
            directly in S3:
          </p>
          <ul className="ml-4 mt-2 text-sm list-disc">
            <li>
              Vector buckets and indexes will be automatically created if they
              don&apos;t exist
            </li>
            <li>
              Vector dimensions are auto-detected from your selected embedding
              model
            </li>
            <li>
              Ensure your AWS credentials have permissions for S3 Vectors
              operations
            </li>
            <li>
              Learn more:{" "}
              <a
                href="https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vector-buckets.html"
                target="_blank"
                rel="noopener noreferrer"
                className="underline"
              >
                AWS S3 Vectors Documentation
              </a>
            </li>
          </ul>
        </div>
      </div>

      {/* Vector Bucket Name */}
      <Form.Item
        label={
          <span>
            Vector Bucket Name{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  S3 bucket name for vector storage (must be at least 3
                  characters, lowercase letters, numbers, hyphens, and
                  periods only)
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        required
        validateStatus={
          providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3
            ? "error"
            : undefined
        }
        help={
          providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3
            ? "Bucket name must be at least 3 characters"
            : undefined
        }
      >
        <Input
          value={providerParams.vector_bucket_name || ""}
          onChange={(e) => handleFieldChange("vector_bucket_name", e.target.value)}
          placeholder="my-vector-bucket (min 3 chars)"
          className="rounded-md"
        />
      </Form.Item>

      {/* Index Name (Optional) */}
      <Form.Item
        label={
          <span>
            Index Name{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  Name for the vector index (optional, will be auto-generated
                  if not provided). If provided, must be at least 3
                  characters.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        validateStatus={
          providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3
            ? "error"
            : undefined
        }
        help={
          providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3
            ? "Index name must be at least 3 characters if provided"
            : undefined
        }
      >
        <Input
          value={providerParams.index_name || ""}
          onChange={(e) => handleFieldChange("index_name", e.target.value)}
          placeholder="my-vector-index (optional, min 3 chars)"
          className="rounded-md"
        />
      </Form.Item>

      {/* AWS Region */}
      <Form.Item
        label={
          <span>
            AWS Region{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  AWS region where the S3 bucket is located (e.g., us-west-2)
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        required
      >
        <Input
          value={providerParams.aws_region_name || ""}
          onChange={(e) => handleFieldChange("aws_region_name", e.target.value)}
          placeholder="us-west-2"
          className="rounded-md"
        />
      </Form.Item>

      {/* Embedding Model */}
      <Form.Item
        label={
          <span>
            Embedding Model{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  Select the embedding model to use for vector generation
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        required
      >
        <Select
          value={providerParams.embedding_model || undefined}
          onChange={(value) => handleFieldChange("embedding_model", value)}
          placeholder="Select an embedding model"
          size="large"
          showSearch
          loading={isLoadingModels}
          filterOption={(input, option) =>
            (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
          }
          options={embeddingModels.map((model) => ({
            value: model.model_group,
            label: model.model_group,
          }))}
          style={{ width: "100%" }}
        />
      </Form.Item>
    </>
  );
};

export default S3VectorsConfig;
