"use client";

import { useWebSearchInterceptionSettings } from "@/app/(dashboard)/hooks/webSearchInterceptionSettings/useWebSearchInterceptionSettings";
import { useUpdateWebSearchInterceptionSettings } from "@/app/(dashboard)/hooks/webSearchInterceptionSettings/useUpdateWebSearchInterceptionSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import { Skeleton } from "@/components/ui/skeleton";
import { CircleHelp, Info, Save } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Providers, provider_map } from "@/components/provider_info_helpers";
import { fetchSearchTools } from "@/components/networking";

interface WebSearchInterceptionStoredValues {
  enabled?: boolean;
  enabled_providers?: string[];
  search_tool_name?: string | null;
  max_agentic_loops?: number | null;
}

interface WebSearchInterceptionFieldSchema {
  properties?: {
    enabled?: { description?: string };
    enabled_providers?: { description?: string };
    search_tool_name?: { description?: string };
    max_agentic_loops?: { description?: string };
  };
}

interface WebSearchInterceptionFormValues {
  enabled: boolean;
  enabled_providers: string[];
  search_tool_name: string | null;
  max_agentic_loops: number | null;
}

const NO_STORED_VALUES: WebSearchInterceptionStoredValues = {};

const MAX_AGENTIC_LOOPS_MIN = 1;

const PROVIDER_OPTIONS = Object.entries(provider_map)
  .map(([enumKey, providerValue]) => ({
    label: Providers[enumKey as keyof typeof Providers] ?? providerValue,
    value: providerValue,
  }))
  .sort((a, b) => a.label.localeCompare(b.label));

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const parseLoops = (raw: string, rawAsNumber: number): number | null =>
  raw === "" || Number.isNaN(rawAsNumber) ? null : rawAsNumber;

const toFormValues = (values: WebSearchInterceptionStoredValues): WebSearchInterceptionFormValues => ({
  enabled: values.enabled ?? false,
  enabled_providers: values.enabled_providers ?? [],
  search_tool_name: values.search_tool_name ?? null,
  max_agentic_loops: values.max_agentic_loops ?? null,
});

const readSearchToolNames = (response: unknown): string[] => {
  const payload = response as { search_tools?: unknown; data?: unknown } | null;
  const tools = Array.isArray(payload?.search_tools) ? payload.search_tools : payload?.data;
  if (!Array.isArray(tools)) {
    return [];
  }
  return tools
    .map((tool: { search_tool_name?: string }) => tool?.search_tool_name)
    .filter((name: unknown): name is string => typeof name === "string" && name.length > 0);
};

const useSearchToolNames = (accessToken: string) => {
  const [searchTools, setSearchTools] = useState<string[]>([]);
  const [loadingSearchTools, setLoadingSearchTools] = useState(true);

  useEffect(() => {
    const loadSearchTools = async () => {
      if (!accessToken) return;
      try {
        setSearchTools(readSearchToolNames(await fetchSearchTools(accessToken)));
      } catch (loadError) {
        console.error("Error fetching search tools:", loadError);
      } finally {
        setLoadingSearchTools(false);
      }
    };

    loadSearchTools();
  }, [accessToken]);

  return { searchTools, loadingSearchTools };
};

interface WebSearchInterceptionFormProps {
  accessToken: string;
  initial: WebSearchInterceptionFormValues;
  schema: WebSearchInterceptionFieldSchema | undefined;
}

function WebSearchInterceptionForm({ accessToken, initial, schema }: WebSearchInterceptionFormProps) {
  const {
    mutate: updateSettings,
    isPending: isUpdating,
    error: updateError,
  } = useUpdateWebSearchInterceptionSettings(accessToken);
  const { searchTools, loadingSearchTools } = useSearchToolNames(accessToken);
  const form = useForm<WebSearchInterceptionFormValues>({ defaultValues: initial });
  const isDirty = form.formState.isDirty;

  const handleSave = (formValues: WebSearchInterceptionFormValues) => {
    updateSettings(formValues, {
      onSuccess: () => {
        form.reset(formValues);
        toast.success("Settings updated successfully. Changes will be applied across all pods within 10 seconds.");
      },
      onError: (saveError) => {
        toast.fromError(saveError);
      },
    });
  };

  return (
    <>
      {updateError && (
        <Alert variant="error" className="mb-4">
          <AlertTitle>Could not update settings</AlertTitle>
          {updateError instanceof Error && <AlertDescription>{updateError.message}</AlertDescription>}
        </Alert>
      )}

      <TooltipProvider>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <Card className="mb-4">
            <CardContent>
              <FieldGroup>
                <FormField
                  control={form.control}
                  name="enabled"
                  label={labelWithHint(
                    "Enable Web Search Interception",
                    "When enabled, web search tool calls are executed server-side through the selected search tool",
                  )}
                  description={schema?.properties?.enabled?.description}
                >
                  {({ value, onChange, onBlur, id }) => (
                    <Switch id={id} checked={value} onCheckedChange={onChange} onBlur={onBlur} disabled={isUpdating} />
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="enabled_providers"
                  label={labelWithHint(
                    "Providers",
                    "Which LLM providers to intercept for. Leave empty to intercept Bedrock only.",
                  )}
                  description={schema?.properties?.enabled_providers?.description}
                >
                  {({ value, onChange, id }) => (
                    <MultiSelect
                      id={id}
                      options={PROVIDER_OPTIONS}
                      value={value}
                      onValueChange={onChange}
                      placeholder="Select providers (defaults to Bedrock)"
                      allowCustomValues
                      disabled={isUpdating}
                    />
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="search_tool_name"
                  label={labelWithHint(
                    "Search Tool",
                    "Which configured search tool runs the searches. Leave empty to use the first one available.",
                  )}
                  description={schema?.properties?.search_tool_name?.description}
                >
                  {({ value, onChange, id }) => (
                    <SearchSelect
                      inputId={id}
                      options={searchTools.map((name) => ({ label: name, value: name }))}
                      value={value}
                      onValueChange={onChange}
                      placeholder="Select a search tool (defaults to the first available)"
                      disabled={isUpdating || loadingSearchTools}
                    />
                  )}
                </FormField>

                <FormField
                  control={form.control}
                  name="max_agentic_loops"
                  label={labelWithHint(
                    "Max Agentic Loops",
                    "How many follow-up model calls one intercepted request may chain. Leave empty for the default of 3.",
                  )}
                  description={schema?.properties?.max_agentic_loops?.description}
                >
                  {({ value, onChange, onBlur, id, ref }) => (
                    <Input
                      id={id}
                      ref={ref}
                      type="number"
                      min={MAX_AGENTIC_LOOPS_MIN}
                      value={value ?? ""}
                      onChange={(event) => onChange(parseLoops(event.target.value, event.target.valueAsNumber))}
                      onBlur={onBlur}
                      disabled={isUpdating}
                    />
                  )}
                </FormField>
              </FieldGroup>
            </CardContent>
          </Card>

          <div className="flex justify-end gap-2">
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
    </>
  );
}

export default function WebSearchInterceptionSettings() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useWebSearchInterceptionSettings();

  if (!accessToken) {
    return (
      <div className="p-6 text-center text-muted-foreground">
        Please log in to configure web search interception settings.
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-4 w-2/5" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/5" />
      </div>
    );
  }

  if (isError) {
    return (
      <Alert variant="error" className="mb-6">
        <AlertTitle>Could not load web search interception settings</AlertTitle>
        {error instanceof Error && <AlertDescription>{error.message}</AlertDescription>}
      </Alert>
    );
  }

  const values: WebSearchInterceptionStoredValues = data?.values ?? NO_STORED_VALUES;

  return (
    <div className="w-full">
      <Alert variant="info" className="mb-6">
        <Info />
        <AlertTitle>Web Search Interception</AlertTitle>
        <AlertDescription>
          Serve web search tool calls from a configured search tool instead of passing them upstream, so models without
          native web search can still answer with fresh results. Click &apos;Save Settings&apos; to apply changes across
          all pods (takes effect within 10 seconds).
        </AlertDescription>
      </Alert>

      <WebSearchInterceptionForm
        key={JSON.stringify(values)}
        accessToken={accessToken}
        initial={toFormValues(values)}
        schema={data?.field_schema}
      />
    </div>
  );
}
