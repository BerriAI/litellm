"use client";

import { useMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings";
import { useUpdateMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings";
import { toast } from "@/lib/toast";
import { Skeleton } from "@/components/ui/skeleton";
import { CircleCheck, CircleHelp, Info, Save, X } from "lucide-react";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import MCPSemanticFilterTestPanel from "./MCPSemanticFilterTestPanel";
import { getCurlCommand, runSemanticFilterTest, TestResult } from "./semanticFilterTestUtils";

interface MCPSemanticFilterSettingsProps {
  accessToken: string | null;
}

interface MCPSemanticFilterStoredValues {
  enabled?: boolean;
  embedding_model?: string;
  top_k?: number;
  similarity_threshold?: number;
}

interface MCPSemanticFilterFieldSchema {
  properties?: { enabled?: { description?: string } };
}

interface MCPSemanticFilterFormValues {
  enabled: boolean;
  embedding_model: string;
  top_k: number | null;
  similarity_threshold: number;
}

const DEFAULT_FORM_VALUES: MCPSemanticFilterFormValues = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: 10,
  similarity_threshold: 0.3,
};

const NO_STORED_VALUES: MCPSemanticFilterStoredValues = {};

const TOP_K_MIN = 1;
const TOP_K_MAX = 100;

const SIMILARITY_THRESHOLD_MARKS = [
  { value: 0, label: "0.0" },
  { value: 0.3, label: "0.3" },
  { value: 0.5, label: "0.5" },
  { value: 0.7, label: "0.7" },
  { value: 1, label: "1.0" },
];

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const clampTopK = (value: number | null): number | null =>
  value === null ? null : Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, value));

const parseTopK = (raw: string, rawAsNumber: number): number | null =>
  raw === "" || Number.isNaN(rawAsNumber) ? null : rawAsNumber;

const SaveSuccessAlert = () => {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return null;
  }

  return (
    <Alert variant="success" className="mb-4">
      <CircleCheck />
      <AlertTitle>Settings saved successfully</AlertTitle>
      <AlertAction>
        <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={() => setDismissed(true)}>
          <X className="size-4" />
        </Button>
      </AlertAction>
    </Alert>
  );
};

export default function MCPSemanticFilterSettings({ accessToken }: MCPSemanticFilterSettingsProps) {
  const { data, isLoading, isError, error } = useMCPSemanticFilterSettings();
  const {
    mutate: updateSettings,
    isPending: isUpdating,
    error: updateError,
  } = useUpdateMCPSemanticFilterSettings(accessToken || "");
  const form = useForm<MCPSemanticFilterFormValues>({ defaultValues: DEFAULT_FORM_VALUES });
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [embeddingModels, setEmbeddingModels] = useState<ModelGroup[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);

  // Test section state
  const [testQuery, setTestQuery] = useState("");
  const [testModel, setTestModel] = useState<string>("gpt-4o");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [testError, setTestError] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState(false);

  const schema: MCPSemanticFilterFieldSchema | undefined = data?.field_schema;
  const values: MCPSemanticFilterStoredValues = data?.values ?? NO_STORED_VALUES;

  useEffect(() => {
    const loadEmbeddingModels = async () => {
      if (!accessToken) return;
      try {
        setLoadingModels(true);
        const models = await fetchAvailableModels(accessToken);
        const embeddingOnly = models.filter((model) => model.mode === "embedding");
        setEmbeddingModels(embeddingOnly);
      } catch (error) {
        console.error("Error fetching embedding models:", error);
      } finally {
        setLoadingModels(false);
      }
    };

    loadEmbeddingModels();
  }, [accessToken]);

  useEffect(() => {
    if (values) {
      form.reset({
        enabled: values.enabled ?? DEFAULT_FORM_VALUES.enabled,
        embedding_model: values.embedding_model ?? DEFAULT_FORM_VALUES.embedding_model,
        top_k: values.top_k ?? DEFAULT_FORM_VALUES.top_k,
        similarity_threshold: values.similarity_threshold ?? DEFAULT_FORM_VALUES.similarity_threshold,
      });
      setIsDirty(false);
    }
  }, [values, form]);

  const commitChange = <TValue,>(onChange: (value: TValue) => void, value: TValue) => {
    onChange(value);
    setIsDirty(true);
  };

  const handleSave = (formValues: MCPSemanticFilterFormValues) => {
    updateSettings(formValues, {
      onSuccess: () => {
        setIsDirty(false);
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
        toast.success("Settings updated successfully. Changes will be applied across all pods within 10 seconds.");
      },
      onError: (error) => {
        toast.fromError(error);
      },
    });
  };

  const handleTest = async () => {
    if (!accessToken) {
      return;
    }

    await runSemanticFilterTest({
      accessToken,
      testModel,
      testQuery,
      setIsTesting,
      setTestResult,
      setTestError,
    });
  };

  if (!accessToken) {
    return (
      <div className="p-6 text-center text-muted-foreground">Please log in to configure semantic filter settings.</div>
    );
  }

  return (
    <div style={{ width: "100%" }}>
      {isLoading ? (
        <div className="flex flex-col gap-3">
          <Skeleton className="h-4 w-2/5" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/5" />
        </div>
      ) : isError ? (
        <Alert variant="error" className="mb-6">
          <AlertTitle>Could not load MCP Semantic Filter settings</AlertTitle>
          {error instanceof Error && <AlertDescription>{error.message}</AlertDescription>}
        </Alert>
      ) : (
        <>
          <Alert variant="info" className="mb-6">
            <Info />
            <AlertTitle>Semantic Tool Filtering</AlertTitle>
            <AlertDescription>
              Filter MCP tools semantically based on query relevance. This reduces context window size and improves tool
              selection accuracy. Click &apos;Save Settings&apos; to apply changes across all pods (takes effect within
              10 seconds).
            </AlertDescription>
          </Alert>

          {saveSuccess && <SaveSuccessAlert />}

          {updateError && (
            <Alert variant="error" className="mb-4">
              <AlertTitle>Could not update settings</AlertTitle>
              {updateError instanceof Error && <AlertDescription>{updateError.message}</AlertDescription>}
            </Alert>
          )}

          <div className="grid grid-cols-1 gap-x-6 lg:grid-cols-2">
            {/* Left Column - Settings */}
            <div>
              <TooltipProvider>
                <form onSubmit={(event) => event.preventDefault()} noValidate>
                  <Card className="mb-4">
                    <CardContent>
                      <FieldGroup>
                        <FormField
                          control={form.control}
                          name="enabled"
                          label={labelWithHint(
                            "Enable Semantic Filtering",
                            "When enabled, only the most relevant MCP tools will be included in requests based on semantic similarity",
                          )}
                          description={schema?.properties?.enabled?.description}
                        >
                          {({ value, onChange, onBlur, id }) => (
                            <Switch
                              id={id}
                              checked={value}
                              onCheckedChange={(checked) => commitChange(onChange, checked)}
                              onBlur={onBlur}
                              disabled={isUpdating}
                            />
                          )}
                        </FormField>
                      </FieldGroup>
                    </CardContent>
                  </Card>

                  <Card className="mb-4">
                    <CardHeader className="border-b">
                      <CardTitle>Configuration</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <FieldGroup>
                        <FormField
                          control={form.control}
                          name="embedding_model"
                          label={labelWithHint(
                            "Embedding Model",
                            "The model used to generate embeddings for semantic matching",
                          )}
                        >
                          {({ value, onChange, id }) => (
                            <SearchSelect
                              inputId={id}
                              options={embeddingModels.map((model) => ({
                                label: model.model_group,
                                value: model.model_group,
                              }))}
                              value={value}
                              onValueChange={(selected) => commitChange(onChange, selected)}
                              allowClear={false}
                              placeholder={loadingModels ? "Loading models..." : "Select embedding model"}
                              emptyText={loadingModels ? "Loading..." : "No embedding models available"}
                              disabled={isUpdating || loadingModels}
                            />
                          )}
                        </FormField>

                        <FormField
                          control={form.control}
                          name="top_k"
                          label={labelWithHint("Top K Results", "Maximum number of tools to return after filtering")}
                        >
                          {({ ref, value, onChange, onBlur, id }) => (
                            <Input
                              id={id}
                              ref={ref}
                              type="number"
                              min={TOP_K_MIN}
                              max={TOP_K_MAX}
                              value={value ?? ""}
                              onChange={(event) =>
                                commitChange(onChange, parseTopK(event.target.value, event.target.valueAsNumber))
                              }
                              onBlur={() => {
                                onChange(clampTopK(value));
                                onBlur();
                              }}
                              disabled={isUpdating}
                            />
                          )}
                        </FormField>

                        <FormField
                          control={form.control}
                          name="similarity_threshold"
                          label={labelWithHint(
                            "Similarity Threshold",
                            "Minimum similarity score (0-1) for a tool to be included",
                          )}
                        >
                          {({ value, onChange, id }) => (
                            <div className="w-full">
                              <Slider
                                id={id}
                                min={0}
                                max={1}
                                step={0.05}
                                value={[value]}
                                onValueChange={(next) => commitChange(onChange, Array.isArray(next) ? next[0] : next)}
                                disabled={isUpdating}
                              />
                              <div className="relative mt-2 h-4 text-xs text-muted-foreground">
                                {SIMILARITY_THRESHOLD_MARKS.map((mark) => (
                                  <span
                                    key={mark.value}
                                    className="absolute -translate-x-1/2"
                                    style={{ left: `${mark.value * 100}%` }}
                                  >
                                    {mark.label}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </FormField>
                      </FieldGroup>
                    </CardContent>
                  </Card>

                  <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
                    <Button
                      type="button"
                      onClick={() => void form.handleSubmit(handleSave)()}
                      disabled={!isDirty || isUpdating}
                    >
                      {isUpdating ? <UiLoadingSpinner className="size-4" /> : <Save />}
                      Save Settings
                    </Button>
                  </div>
                </form>
              </TooltipProvider>
            </div>

            {/* Right Column - Test Configuration */}
            <div>
              <MCPSemanticFilterTestPanel
                accessToken={accessToken}
                testQuery={testQuery}
                setTestQuery={setTestQuery}
                testModel={testModel}
                setTestModel={setTestModel}
                isTesting={isTesting}
                onTest={handleTest}
                filterEnabled={!!values.enabled}
                testResult={testResult}
                testError={testError}
                curlCommand={getCurlCommand(testModel, testQuery)}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
