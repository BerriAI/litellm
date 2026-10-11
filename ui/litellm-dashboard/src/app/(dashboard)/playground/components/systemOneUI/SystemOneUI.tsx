"use client";

import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { cn } from "@/lib/cva.config";
import { DECISIONS_DOCS_URL } from "@/lib/decisionModels";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { uiHref } from "@/utils/uiHref";
import { useMutation } from "@tanstack/react-query";
import { BookOpen, Eraser, LoaderCircle, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makePlaygroundDecisionRequest, type PlaygroundDecisionRequest } from "../../llm_calls/system_one";
import { DECISION_PRESETS, emptyPayload, matchPreset, presetPayload, type DecisionPreset } from "./lib/example";
import { payloadModel, withPayloadModel } from "./lib/payloadModel";
import CurlPreview from "./CurlPreview";
import ResponseView from "./ResponseView";
import SystemOneForm from "./SystemOneForm";
import OpenAIDecisionsForm from "./OpenAIDecisionsForm";
import { validateSystemOnePayload, validateOpenAIDecisionsPayload, errorCount } from "./lib/validatePayload";
import { useDecisionModels, type ApiKeySource } from "./useDecisionModels";

interface SystemOneUIProps {
  accessToken: string | null;
  disabledPersonalKeyCreation?: boolean;
}

type EditorView = "form" | "curl";

interface SystemOneSendVariables {
  request: PlaygroundDecisionRequest;
  apiKey: string;
  signal: AbortSignal;
}

export const NOUL_HELP = "A yes / no question. The model returns the probability that the answer is yes.";
const CUSTOM_PRESET = "custom";

interface PresetPickerProps {
  activePreset: DecisionPreset | undefined;
  onPick: (id: string | null) => void;
}

function PresetPicker({ activePreset, onPick }: PresetPickerProps) {
  return (
    <Select value={activePreset?.id ?? CUSTOM_PRESET} onValueChange={onPick}>
      <SelectTrigger className="w-56" aria-label="Example">
        <SelectValue>{activePreset?.label ?? "Custom"}</SelectValue>
      </SelectTrigger>
      <SelectContent>
        {activePreset === undefined && (
          <SelectItem value={CUSTOM_PRESET} disabled>
            Custom
          </SelectItem>
        )}
        {DECISION_PRESETS.map((preset) => (
          <SelectItem key={preset.id} value={preset.id}>
            {preset.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function getCustomProxyBaseUrl(): string | undefined {
  return typeof window === "undefined" ? undefined : window.sessionStorage.getItem("customProxyBaseUrl") || undefined;
}

export default function SystemOneUI({ accessToken, disabledPersonalKeyCreation = false }: SystemOneUIProps) {
  const [apiKeySource, setApiKeySource] = useState<ApiKeySource>(disabledPersonalKeyCreation ? "custom" : "session");
  const [customApiKey, setCustomApiKey] = useState("");
  const effectiveApiKey = apiKeySource === "session" ? accessToken || "" : customApiKey.trim();
  const { decisionModels, isLoaded: decisionModelsLoaded } = useDecisionModels(
    apiKeySource,
    effectiveApiKey,
    getCustomProxyBaseUrl(),
  );
  const [endpoint, setEndpoint] = useState<PlaygroundDecisionRequest["endpoint"]>("/v1/systemone");
  const [drafts, setDrafts] = useState<Partial<Record<PlaygroundDecisionRequest["endpoint"], string>>>({});
  const [view, setView] = useState<EditorView>("form");
  const rawPayload = drafts[endpoint] ?? presetPayload(DECISION_PRESETS[0], endpoint, decisionModels[0]);
  const draftModel = drafts[endpoint] === undefined ? decisionModels[0] : payloadModel(rawPayload);
  const activePreset = matchPreset(endpoint, rawPayload);
  const clearedPayload = emptyPayload(endpoint, draftModel);
  const activeController = useRef<AbortController | null>(null);
  const validated = useMemo(
    () =>
      endpoint === "/v1/decisions"
        ? ({ endpoint, validation: validateOpenAIDecisionsPayload(rawPayload) } as const)
        : ({ endpoint, validation: validateSystemOnePayload(rawPayload, endpoint) } as const),
    [rawPayload, endpoint],
  );
  const validation = validated.validation;

  const systemOne = useMutation({
    mutationFn: ({ request, apiKey, signal }: SystemOneSendVariables) =>
      makePlaygroundDecisionRequest(request, apiKey, getCustomProxyBaseUrl(), { signal }),
  });
  const isLoading = systemOne.isPending;
  const hasResult = isLoading || systemOne.data !== undefined || systemOne.error !== null;
  const { reset: resetSystemOne } = systemOne;

  useEffect(() => () => activeController.current?.abort(), []);

  useEffect(() => {
    activeController.current?.abort();
    activeController.current = null;
    resetSystemOne();
  }, [effectiveApiKey, resetSystemOne]);

  function clearRequestState() {
    activeController.current?.abort();
    activeController.current = null;
    systemOne.reset();
  }

  function handleReset() {
    handlePayloadChange(presetPayload(DECISION_PRESETS[0], endpoint, draftModel ?? decisionModels[0]));
  }

  function handlePayloadChange(value: string) {
    if (value !== rawPayload) {
      clearRequestState();
      setDrafts((current) => ({ ...current, [endpoint]: value }));
    }
  }

  function handlePresetPick(id: string | null) {
    const preset = DECISION_PRESETS.find((candidate) => candidate.id === id);
    if (preset !== undefined) {
      handlePayloadChange(presetPayload(preset, endpoint, draftModel));
    }
  }

  function handleModelPick(model: string | null) {
    const next = model === null ? undefined : withPayloadModel(rawPayload, model);
    if (next !== undefined) {
      handlePayloadChange(next);
    }
  }

  function handleSend() {
    const request: PlaygroundDecisionRequest | undefined =
      validated.endpoint === "/v1/decisions"
        ? validated.validation.payload && { endpoint: validated.endpoint, payload: validated.validation.payload }
        : validated.validation.payload && { endpoint: validated.endpoint, payload: validated.validation.payload };
    if (!request || !effectiveApiKey) {
      return;
    }
    clearRequestState();
    setDrafts((current) => ({ ...current, [endpoint]: rawPayload }));
    const controller = new AbortController();
    activeController.current = controller;
    const variables = { request, apiKey: effectiveApiKey, signal: controller.signal };
    systemOne.mutate(variables);
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-auto p-4 xl:overflow-hidden">
      <div
        className={cn(
          "mx-auto grid w-full gap-4 xl:min-h-0 xl:flex-1",
          hasResult ? "max-w-7xl xl:grid-cols-2" : "max-w-3xl",
        )}
      >
        <section className="flex min-h-96 flex-col gap-3 xl:min-h-0" aria-label="Decisions request editor">
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={endpoint}
              onValueChange={(value) => {
                if (value === "/v1/decisions" || value === "/v1/systemone") {
                  clearRequestState();
                  setEndpoint(value);
                }
              }}
            >
              <SelectTrigger className="w-44" aria-label="Decision endpoint">
                <SelectValue>{endpoint}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="/v1/decisions">/v1/decisions</SelectItem>
                <SelectItem value="/v1/systemone">/v1/systemone</SelectItem>
              </SelectContent>
            </Select>
            <div className="w-56">
              <SearchSelect
                aria-label="Decision model"
                options={decisionModels.map((model) => ({ label: model, value: model }))}
                value={payloadModel(rawPayload) ?? null}
                onValueChange={handleModelPick}
                placeholder={decisionModels.length > 0 ? "Pick a decision model" : "No decision models yet"}
                emptyText="No decision models found"
                allowClear={false}
              />
            </div>
            {decisionModelsLoaded && decisionModels.length === 0 && (
              <a href={uiHref("models-and-endpoints")} className="text-sm text-primary underline">
                Add a decision model
              </a>
            )}
            <Select
              value={apiKeySource}
              onValueChange={(value) => {
                if (value === "session" || value === "custom") {
                  setApiKeySource(value);
                }
              }}
              disabled={disabledPersonalKeyCreation}
            >
              <SelectTrigger className="w-44" aria-label="Virtual Key Source">
                <SelectValue>{apiKeySource === "custom" ? "Virtual Key" : "Current UI Session"}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="session" disabled={!accessToken}>
                  Current UI Session
                </SelectItem>
                <SelectItem value="custom">Virtual Key</SelectItem>
              </SelectContent>
            </Select>
            {apiKeySource === "custom" && (
              <Input
                type="password"
                aria-label="Virtual Key"
                value={customApiKey}
                onChange={(event) => setCustomApiKey(event.target.value)}
                placeholder="Enter Virtual Key"
                className="w-56"
              />
            )}
          </div>

          <Tabs
            value={view}
            onValueChange={(value) => (value === "form" || value === "curl") && setView(value)}
            className="min-h-0 flex-1"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <TabsList aria-label="Request editor view">
                <TabsTrigger value="form">Form</TabsTrigger>
                <TabsTrigger value="curl">cURL</TabsTrigger>
              </TabsList>
              <div className="flex items-center gap-1">
                <PresetPicker activePreset={activePreset} onPick={handlePresetPick} />
                <SimpleTooltip content="How to call /v1/decisions and /v1/systemone">
                  <a
                    href={DECISIONS_DOCS_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="How to call /v1/decisions and /v1/systemone"
                    className={cn(buttonVariants({ variant: "ghost", size: "icon" }), "shrink-0")}
                  >
                    <BookOpen />
                  </a>
                </SimpleTooltip>
              </div>
            </div>
            <TabsContent value="form" className="flex min-h-0 flex-col">
              {validated.endpoint === "/v1/decisions" ? (
                <OpenAIDecisionsForm
                  value={rawPayload}
                  onChange={handlePayloadChange}
                  validation={validated.validation}
                  onReset={handleReset}
                />
              ) : (
                <SystemOneForm
                  value={rawPayload}
                  onChange={handlePayloadChange}
                  validation={validated.validation}
                  onReset={handleReset}
                />
              )}
            </TabsContent>
            <TabsContent value="curl" className="flex min-h-0 flex-col">
              <CurlPreview
                baseUrl={getCustomProxyBaseUrl() ?? getProxyBaseUrl()}
                endpoint={endpoint}
                payload={validation.payload === undefined ? rawPayload : JSON.stringify(validation.payload)}
                authHeaderName={getGlobalLitellmHeaderName()}
                errorCount={errorCount(validation)}
              />
            </TabsContent>
          </Tabs>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="ghost"
              onClick={() => handlePayloadChange(clearedPayload)}
              disabled={rawPayload === clearedPayload}
            >
              <Eraser />
              Clear
            </Button>
            <div className="ml-auto flex items-center gap-2">
              {isLoading && (
                <Button variant="outline" onClick={clearRequestState}>
                  Cancel request
                </Button>
              )}
              <Button
                size="lg"
                className="min-w-32"
                onClick={handleSend}
                disabled={!validation.isValid || isLoading || !effectiveApiKey}
              >
                {isLoading ? <LoaderCircle className="animate-spin" /> : <Send />}
                Send
              </Button>
            </div>
          </div>
        </section>

        {hasResult && (
          <section className="grid content-start gap-4 xl:min-h-0 xl:overflow-auto" aria-label="Decisions results">
            <ResponseView
              response={systemOne.data?.response}
              view={view}
              fallbackModel={validation.payload?.model}
              latencyMs={systemOne.data?.latencyMs}
              error={systemOne.error?.message}
              isLoading={isLoading}
            />
          </section>
        )}
      </div>
    </div>
  );
}
