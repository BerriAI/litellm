"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { DECISIONS_DOCS_URL } from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";
import { useMutation } from "@tanstack/react-query";
import { Code, Info, LoaderCircle, RotateCcw, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makePlaygroundDecisionRequest, type PlaygroundDecisionRequest } from "../../llm_calls/system_one";
import { decisionsExample, openAIDecisionsExample } from "./lib/example";
import { payloadModel, withPayloadModel } from "./lib/payloadModel";
import JsonEditor from "./JsonEditor";
import QuestionBreakdown from "./QuestionBreakdown";
import ResponseView from "./ResponseView";
import SystemOneForm from "./SystemOneForm";
import OpenAIDecisionsForm from "./OpenAIDecisionsForm";
import { validateSystemOnePayload, validateOpenAIDecisionsPayload } from "./lib/validatePayload";
import { useDecisionModels, type ApiKeySource } from "./useDecisionModels";

interface SystemOneUIProps {
  accessToken: string | null;
  disabledPersonalKeyCreation?: boolean;
}

type EditorView = "form" | "json";

interface SystemOneSendVariables {
  request: PlaygroundDecisionRequest;
  apiKey: string;
  signal: AbortSignal;
}

export const NOUL_HELP = "noul is a yes / no question. The model returns the probability that the answer is yes.";
const DECISION_MODELS_DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/44231";

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
  const example = endpoint === "/v1/decisions" ? openAIDecisionsExample : decisionsExample;
  const examplePayload = JSON.stringify(example(decisionModels[0]), null, 2);
  const rawPayload = drafts[endpoint] ?? examplePayload;
  const activeController = useRef<AbortController | null>(null);
  const validated = useMemo(
    () =>
      endpoint === "/v1/decisions"
        ? ({ endpoint, validation: validateOpenAIDecisionsPayload(rawPayload) } as const)
        : ({ endpoint, validation: validateSystemOnePayload(rawPayload, endpoint) } as const),
    [rawPayload, endpoint],
  );
  const validation = validated.validation;
  const hasSyntaxError = validation.issues.some((issue) => issue.path === "syntax");

  const systemOne = useMutation({
    mutationFn: ({ request, apiKey, signal }: SystemOneSendVariables) =>
      makePlaygroundDecisionRequest(request, apiKey, getCustomProxyBaseUrl(), { signal }),
  });
  const isLoading = systemOne.isPending;
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

  function handlePayloadChange(value: string) {
    if (value !== rawPayload) {
      clearRequestState();
      setDrafts((current) => ({ ...current, [endpoint]: value }));
    }
  }

  function handleResetExample() {
    clearRequestState();
    setDrafts((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== endpoint)));
  }

  function handleModelPick(model: string | null) {
    const next = model === null ? undefined : withPayloadModel(rawPayload, model);
    if (next !== undefined) {
      handlePayloadChange(next);
    }
  }

  function handleFormatJson() {
    if (rawPayload.trim() && !hasSyntaxError) {
      handlePayloadChange(JSON.stringify(JSON.parse(rawPayload), null, 2));
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
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-auto p-4 xl:overflow-hidden">
      <section className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-muted-foreground">Endpoint</span>
          <Select
            value={endpoint}
            onValueChange={(value) => {
              if (value === "/v1/decisions" || value === "/v1/systemone") {
                clearRequestState();
                setEndpoint(value);
              }
            }}
          >
            <SelectTrigger className="w-80" aria-label="Decision endpoint">
              <SelectValue>{endpoint}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="/v1/decisions">/v1/decisions</SelectItem>
              <SelectItem value="/v1/systemone">/v1/systemone</SelectItem>
            </SelectContent>
          </Select>
          <span className="ml-2 text-sm font-medium text-muted-foreground">Model</span>
          <div className="w-80">
            <SearchSelect
              aria-label="Decision model"
              options={decisionModels.map((model) => ({ label: model, value: model }))}
              value={payloadModel(rawPayload) ?? null}
              onValueChange={handleModelPick}
              placeholder={decisionModels.length > 0 ? "Pick a decision model" : "No decision models yet"}
              emptyText="No decision models found"
              disabled={hasSyntaxError}
              allowClear={false}
            />
          </div>
          {decisionModelsLoaded && decisionModels.length === 0 && (
            <a href={uiHref("models-and-endpoints")} className="text-sm text-primary underline">
              Add a decision model
            </a>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Virtual Key Source</span>
            <Select
              value={apiKeySource}
              onValueChange={(value) => {
                if (value === "session" || value === "custom") {
                  setApiKeySource(value);
                }
              }}
              disabled={disabledPersonalKeyCreation}
            >
              <SelectTrigger className="w-48" aria-label="Virtual Key Source">
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
          <div className="ml-auto flex items-center gap-2">
            <Button variant="outline" onClick={handleResetExample} disabled={rawPayload === examplePayload}>
              <RotateCcw />
              Reset example
            </Button>
            {view === "json" && (
              <Button variant="outline" onClick={handleFormatJson} disabled={!rawPayload.trim() || hasSyntaxError}>
                <Code />
                Format JSON
              </Button>
            )}
            {isLoading && (
              <Button variant="outline" onClick={clearRequestState}>
                Cancel request
              </Button>
            )}
            <Button onClick={handleSend} disabled={!validation.isValid || isLoading || !effectiveApiKey}>
              {isLoading ? <LoaderCircle className="animate-spin" /> : <Send />}
              Send
            </Button>
          </div>
        </div>
        <Alert role="note" aria-label="Decision endpoint notice">
          <Info />
          <AlertTitle>{endpoint}</AlertTitle>
          <AlertDescription>
            {endpoint === "/v1/decisions" ? (
              "Uses input and a questions array with predicate, choice, and score questions. Edit the form or JSON and inspect the response JSON in the OpenAI Decisions format."
            ) : (
              <>
                Uses state and a questions object with{" "}
                <SimpleTooltip content={NOUL_HELP} className="underline decoration-dotted underline-offset-2">
                  noul
                </SimpleTooltip>
                , choice, and score questions in the System One format.
              </>
            )}{" "}
            Pick a decision model under Model, or omit model to use the proxy&apos;s configured default.{" "}
            <a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
              How to call /v1/decisions and /v1/systemone
            </a>
            {" · "}
            <a href={DECISION_MODELS_DISCUSSION_URL} target="_blank" rel="noopener noreferrer" className="underline">
              Give us feedback on what you want for decision models
            </a>
          </AlertDescription>
        </Alert>
      </section>

      <div className="grid gap-4 xl:min-h-0 xl:flex-1 xl:grid-cols-2">
        <section className="flex min-h-96 flex-col xl:min-h-0" aria-label="Decisions request editor">
          <Tabs
            value={view}
            onValueChange={(value) => (value === "form" || value === "json") && setView(value)}
            className="min-h-0 flex-1"
          >
            <TabsList aria-label="Request editor view">
              <TabsTrigger value="form">Form</TabsTrigger>
              <TabsTrigger value="json">JSON</TabsTrigger>
            </TabsList>
            <TabsContent value="form" className="flex min-h-0 flex-col">
              {validated.endpoint === "/v1/decisions" ? (
                <OpenAIDecisionsForm
                  value={rawPayload}
                  onChange={handlePayloadChange}
                  validation={validated.validation}
                  onOpenJson={() => setView("json")}
                />
              ) : (
                <SystemOneForm
                  value={rawPayload}
                  onChange={handlePayloadChange}
                  validation={validated.validation}
                  onOpenJson={() => setView("json")}
                />
              )}
            </TabsContent>
            <TabsContent value="json" className="flex min-h-0 flex-col">
              <JsonEditor
                value={rawPayload}
                onChange={handlePayloadChange}
                validation={validation}
                label={endpoint === "/v1/decisions" ? "Decisions JSON payload" : "System One JSON payload"}
                placeholder={`Paste or write a ${endpoint} request`}
              />
            </TabsContent>
          </Tabs>
        </section>
        <section className="grid content-start gap-4 xl:min-h-0 xl:overflow-auto" aria-label="Decisions results">
          <ResponseView
            response={systemOne.data?.response}
            view={view}
            fallbackModel={validation.payload?.model}
            latencyMs={systemOne.data?.latencyMs}
            error={systemOne.error?.message}
            isLoading={isLoading}
          />
          {view === "form" && validated.endpoint === "/v1/systemone" && (
            <QuestionBreakdown payload={validated.validation.payload} />
          )}
        </section>
      </div>
    </div>
  );
}
