"use client";

import { useMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings";
import { useUpdateMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings";
import NotificationManager from "@/components/molecules/notifications_manager";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CheckCircle, HelpCircle, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import {
  fetchAvailableModels,
  ModelGroup,
} from "@/components/playground/llm_calls/fetch_models";
import MCPSemanticFilterTestPanel from "./MCPSemanticFilterTestPanel";
import {
  getCurlCommand,
  runSemanticFilterTest,
  TestResult,
} from "./semanticFilterTestUtils";

interface MCPSemanticFilterSettingsProps {
  accessToken: string | null;
}

interface FormValues {
  enabled: boolean;
  embedding_model: string;
  top_k: number;
  similarity_threshold: number;
}

function HelpTip({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <HelpCircle className="inline-block ml-1 h-3 w-3 text-muted-foreground cursor-help" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">{children}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export default function MCPSemanticFilterSettings({
  accessToken,
}: MCPSemanticFilterSettingsProps) {
  const { data, isLoading, isError, error } = useMCPSemanticFilterSettings();
  const {
    mutate: updateSettings,
    isPending: isUpdating,
    error: updateError,
  } = useUpdateMCPSemanticFilterSettings(accessToken || "");
  const form = useForm<FormValues>({
    defaultValues: {
      enabled: false,
      embedding_model: "text-embedding-3-small",
      top_k: 10,
      similarity_threshold: 0.3,
    },
    mode: "onSubmit",
  });
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [embeddingModels, setEmbeddingModels] = useState<ModelGroup[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);

  const [testQuery, setTestQuery] = useState("");
  const [testModel, setTestModel] = useState<string>("gpt-4o");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [isTesting, setIsTesting] = useState(false);

  const schema = data?.field_schema;
  const values = useMemo(() => data?.values ?? {}, [data?.values]);

  useEffect(() => {
    const loadEmbeddingModels = async () => {
      if (!accessToken) return;
      try {
        setLoadingModels(true);
        const models = await fetchAvailableModels(accessToken);
        setEmbeddingModels(models.filter((m) => m.mode === "embedding"));
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
        enabled: values.enabled ?? false,
        embedding_model: values.embedding_model ?? "text-embedding-3-small",
        top_k: values.top_k ?? 10,
        similarity_threshold: values.similarity_threshold ?? 0.3,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values]);

  const isDirty = form.formState.isDirty;

  const handleSave = form.handleSubmit((formValues) => {
    updateSettings(formValues, {
      onSuccess: () => {
        form.reset(formValues); // clears dirty state
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
        NotificationManager.success(
          "Settings updated successfully. Changes will be applied across all pods within 10 seconds.",
        );
      },
      onError: (err) => {
        NotificationManager.fromBackend(err);
      },
    });
  });

  const handleTest = async () => {
    if (!accessToken) return;
    await runSemanticFilterTest({
      accessToken,
      testModel,
      testQuery,
      setIsTesting,
      setTestResult,
    });
  };

  if (!accessToken) {
    return (
      <div className="p-6 text-center text-muted-foreground">
        Please log in to configure semantic filter settings.
      </div>
    );
  }

  return (
    <div className="w-full">
      {isLoading ? (
        <Skeleton className="h-32 w-full" />
      ) : isError ? (
        <Alert variant="destructive" className="mb-6">
          <AlertTitle>Could not load MCP Semantic Filter settings</AlertTitle>
          {error instanceof Error && (
            <AlertDescription>{error.message}</AlertDescription>
          )}
        </Alert>
      ) : (
        <>
          <Alert className="mb-6">
            <AlertTitle>Semantic Tool Filtering</AlertTitle>
            <AlertDescription>
              Filter MCP tools semantically based on query relevance. This
              reduces context window size and improves tool selection accuracy.
              Click &apos;Save Settings&apos; to apply changes across all pods
              (takes effect within 10 seconds).
            </AlertDescription>
          </Alert>

          {saveSuccess && (
            <Alert className="mb-4 border-emerald-300 bg-emerald-50 text-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-200">
              <CheckCircle className="h-4 w-4" />
              <AlertTitle>Settings saved successfully</AlertTitle>
            </Alert>
          )}

          {updateError && (
            <Alert variant="destructive" className="mb-4">
              <AlertTitle>Could not update settings</AlertTitle>
              {updateError instanceof Error && (
                <AlertDescription>{updateError.message}</AlertDescription>
              )}
            </Alert>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FormProvider {...form}>
              <form
                onSubmit={handleSave}
                className="space-y-4"
              >
                <Card className="p-4">
                  <div className="flex items-center justify-between">
                    <Label>
                      <span className="font-semibold">
                        Enable Semantic Filtering
                      </span>
                      <HelpTip>
                        When enabled, only the most relevant MCP tools will be
                        included in requests based on semantic similarity
                      </HelpTip>
                    </Label>
                    <Controller
                      control={form.control}
                      name="enabled"
                      render={({ field }) => (
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                          disabled={isUpdating}
                        />
                      )}
                    />
                  </div>
                  <p className="text-sm text-muted-foreground mt-2">
                    {schema?.properties?.enabled?.description}
                  </p>
                </Card>

                <Card className="p-4">
                  <h4 className="text-base font-semibold mb-4">
                    Configuration
                  </h4>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>
                        <span className="font-semibold">Embedding Model</span>
                        <HelpTip>
                          The model used to generate embeddings for semantic
                          matching
                        </HelpTip>
                      </Label>
                      <Controller
                        control={form.control}
                        name="embedding_model"
                        render={({ field }) => (
                          <Select
                            value={field.value ?? ""}
                            onValueChange={(v) => field.onChange(v)}
                            disabled={isUpdating || loadingModels}
                          >
                            <SelectTrigger>
                              <SelectValue
                                placeholder={
                                  loadingModels
                                    ? "Loading models..."
                                    : "Select embedding model"
                                }
                              />
                            </SelectTrigger>
                            <SelectContent>
                              {embeddingModels.length === 0 ? (
                                <div className="py-2 px-3 text-sm text-muted-foreground">
                                  {loadingModels
                                    ? "Loading..."
                                    : "No embedding models available"}
                                </div>
                              ) : (
                                embeddingModels.map((model) => (
                                  <SelectItem
                                    key={model.model_group}
                                    value={model.model_group}
                                  >
                                    {model.model_group}
                                  </SelectItem>
                                ))
                              )}
                            </SelectContent>
                          </Select>
                        )}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="top_k">
                        <span className="font-semibold">Top K Results</span>
                        <HelpTip>
                          Maximum number of tools to return after filtering
                        </HelpTip>
                      </Label>
                      <Input
                        id="top_k"
                        type="number"
                        min={1}
                        max={100}
                        disabled={isUpdating}
                        {...form.register("top_k", { valueAsNumber: true })}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>
                        <span className="font-semibold">
                          Similarity Threshold
                        </span>
                        <HelpTip>
                          Minimum similarity score (0-1) for a tool to be
                          included
                        </HelpTip>
                      </Label>
                      <Controller
                        control={form.control}
                        name="similarity_threshold"
                        render={({ field }) => (
                          <div>
                            <Slider
                              value={[field.value]}
                              onValueChange={([v]) => field.onChange(v)}
                              min={0}
                              max={1}
                              step={0.05}
                              disabled={isUpdating}
                            />
                            <div className="flex justify-between text-xs text-muted-foreground mt-2">
                              <span>0.0</span>
                              <span>0.3</span>
                              <span>0.5</span>
                              <span>0.7</span>
                              <span>1.0</span>
                            </div>
                            <div className="text-right text-sm text-muted-foreground mt-1">
                              Current: {Number(field.value).toFixed(2)}
                            </div>
                          </div>
                        )}
                      />
                    </div>
                  </div>
                </Card>

                <div className="flex justify-end">
                  <Button
                    type="submit"
                    disabled={!isDirty || isUpdating}
                  >
                    <Save className="h-4 w-4" />
                    {isUpdating ? "Saving…" : "Save Settings"}
                  </Button>
                </div>
              </form>
            </FormProvider>

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
              curlCommand={getCurlCommand(testModel, testQuery)}
            />
          </div>
        </>
      )}
    </div>
  );
}
