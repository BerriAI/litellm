import React, { useState, useEffect, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ChevronDown, Info, Search } from "lucide-react";
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

interface FieldShellProps {
  label: string;
  required?: boolean;
  tooltip: string;
  htmlFor: string;
  error?: string;
  children: React.ReactNode;
}

function FieldShell({ label, required, tooltip, htmlFor, error, children }: FieldShellProps) {
  return (
    <div className="mb-4 space-y-1">
      <Label htmlFor={htmlFor} className="flex items-center gap-1">
        <span>
          {label}
          {required && <span className="text-destructive"> *</span>}
        </span>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </Label>
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

interface EmbeddingSelectProps {
  value: string | undefined;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  loading: boolean;
  id: string;
}

function EmbeddingSelect({ value, onChange, options, loading, id }: EmbeddingSelectProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const filtered = useMemo(
    () =>
      options.filter((o) =>
        query ? o.label.toLowerCase().includes(query.toLowerCase()) : true,
      ),
    [options, query],
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          id={id}
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-listbox`}
          className={cn(
            "h-9 w-full flex items-center justify-between rounded-md border border-input bg-background px-3 py-1 text-sm text-left",
          )}
        >
          <span className={value ? "" : "text-muted-foreground"}>
            {value || (loading ? "Loading models…" : "Select an embedding model")}
          </span>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        id={`${id}-listbox`}
        className="w-[var(--radix-popover-trigger-width)] p-2"
      >
        <div className="relative mb-2">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            autoFocus
            placeholder="Search models…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-8 pl-8"
          />
        </div>
        <div className="max-h-60 overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {loading ? "Loading models…" : "No matches"}
            </div>
          ) : (
            filtered.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className="w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent"
                onClick={() => {
                  onChange(opt.value);
                  setOpen(false);
                  setQuery("");
                }}
              >
                {opt.label}
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
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

  const bucketError =
    providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3
      ? "Bucket name must be at least 3 characters"
      : undefined;
  const indexError =
    providerParams.index_name &&
    providerParams.index_name.length > 0 &&
    providerParams.index_name.length < 3
      ? "Index name must be at least 3 characters if provided"
      : undefined;

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

      <FieldShell
        label="Vector Bucket Name"
        required
        htmlFor="s3-vector-bucket-name"
        tooltip="S3 bucket name for vector storage (must be at least 3 characters, lowercase letters, numbers, hyphens, and periods only)"
        error={bucketError}
      >
        <Input
          id="s3-vector-bucket-name"
          value={providerParams.vector_bucket_name || ""}
          onChange={(e) => handleFieldChange("vector_bucket_name", e.target.value)}
          placeholder="my-vector-bucket (min 3 chars)"
          className="rounded-md"
        />
      </FieldShell>

      <FieldShell
        label="Index Name"
        htmlFor="s3-index-name"
        tooltip="Name for the vector index (optional, will be auto-generated if not provided). If provided, must be at least 3 characters."
        error={indexError}
      >
        <Input
          id="s3-index-name"
          value={providerParams.index_name || ""}
          onChange={(e) => handleFieldChange("index_name", e.target.value)}
          placeholder="my-vector-index (optional, min 3 chars)"
          className="rounded-md"
        />
      </FieldShell>

      <FieldShell
        label="AWS Region"
        required
        htmlFor="s3-aws-region"
        tooltip="AWS region where the S3 bucket is located (e.g., us-west-2)"
      >
        <Input
          id="s3-aws-region"
          value={providerParams.aws_region_name || ""}
          onChange={(e) => handleFieldChange("aws_region_name", e.target.value)}
          placeholder="us-west-2"
          className="rounded-md"
        />
      </FieldShell>

      <FieldShell
        label="Embedding Model"
        required
        htmlFor="s3-embedding-model"
        tooltip="Select the embedding model to use for vector generation"
      >
        <EmbeddingSelect
          id="s3-embedding-model"
          value={providerParams.embedding_model || undefined}
          onChange={(v) => handleFieldChange("embedding_model", v)}
          loading={isLoadingModels}
          options={embeddingModels.map((m) => ({
            label: m.model_group,
            value: m.model_group,
          }))}
        />
      </FieldShell>
    </>
  );
};

export default S3VectorsConfig;
