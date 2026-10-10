"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { DECISIONS_DOCS_URL } from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";
import { useMutation } from "@tanstack/react-query";
import { Code, Info, LoaderCircle, RotateCcw, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makeSystemOneRequest } from "../../llm_calls/system_one";
import { PLACEHOLDER_DECISION_MODEL, SYSTEM_ONE_EXAMPLE, decisionsExample } from "./lib/example";
import { payloadModel, withPayloadModel } from "./lib/payloadModel";
import type { DecisionEndpoint, PlaygroundRequest } from "./lib/schemas";
import JsonEditor from "./JsonEditor";
import QuestionBreakdown from "./QuestionBreakdown";
import ResponseView from "./ResponseView";
import SystemOneForm from "./SystemOneForm";
import { validateSystemOnePayload } from "./lib/validatePayload";
import { useDecisionModels, type ApiKeySource } from "./useDecisionModels";

interface SystemOneUIProps {
  accessToken: string | null;
  disabledPersonalKeyCreation?: boolean;
}

type EditorView = "form" | "json";

interface SystemOneSendVariables {
  payload: PlaygroundRequest;
  endpoint: DecisionEndpoint;
  apiKey: string;
  signal: AbortSignal;
}

const EXAMPLE_PAYLOAD = JSON.stringify(SYSTEM_ONE_EXAMPLE, null, 2);
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
  const [chosenEndpoint, setChosenEndpoint] = useState<DecisionEndpoint | null>(null);
  const endpoint: DecisionEndpoint =
    chosenEndpoint ?? (decisionModels.length > 0 ? "/v1/systemone" : "/typesafe/v1/systemone");
  const [drafts, setDrafts] = useState<Partial<Record<DecisionEndpoint, string>>>({});
  const [view, setView] = useState<EditorView>("form");
  const examplePayload =
    endpoint === "/v1/systemone"
      ? JSON.stringify(decisionsExample(decisionModels[0] ?? PLACEHOLDER_DECISION_MODEL), null, 2)
      : EXAMPLE_PAYLOAD;
  const rawPayload = drafts[endpoint] ?? examplePayload;
  const activeController = useRef<AbortController | null>(null);
  const validation = useMemo(() => validateSystemOnePayload(rawPayload, endpoint), [rawPayload, endpoint]);
  const hasSyntaxError = validation.issues.some((issue) => issue.path === "syntax");

  const systemOne = useMutation({
    mutationFn: ({ payload, apiKey, signal, endpoint }: SystemOneSendVariables) =>
      makeSystemOneRequest(payload, apiKey, getCustomProxyBaseUrl(), { signal, endpoint }),
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
      setChosenEndpoint(endpoint);
      setDrafts((current) => ({ ...current, [endpoint]: value }));
    }
  }

  function handleResetExample() {
    clearRequestState();
    setChosenEndpoint(endpoint);
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
    if (!validation.payload || !effectiveApiKey) {
      return;
    }
    clearRequestState();
    setChosenEndpoint(endpoint);
    const controller = new AbortController();
    activeController.current = controller;
    const variables = { payload: validation.payload, apiKey: effectiveApiKey, signal: controller.signal, endpoint };
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
              if (value === "/v1/systemone" || value === "/typesafe/v1/systemone") {
                clearRequestState();
                setChosenEndpoint(value);
              }
            }}
          >
            <SelectTrigger className="w-80" aria-label="Decision endpoint">
              <SelectValue>
                {endpoint === "/v1/systemone" ? "System One · /v1/systemone" : "TypeSafe · /typesafe/v1/systemone"}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="/v1/systemone">System One · /v1/systemone</SelectItem>
              <SelectItem value="/typesafe/v1/systemone">TypeSafe · /typesafe/v1/systemone</SelectItem>
            </SelectContent>
          </Select>
          {endpoint === "/v1/systemone" && (
            <>
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
            </>
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
          <AlertTitle>
            {endpoint === "/v1/systemone" ? "Decision models · System One" : "TypeSafe Jev · System One"}
          </AlertTitle>
          <AlertDescription>
            {endpoint === "/v1/systemone"
              ? "Sends choice, noul, and score questions through /v1/systemone to a decision model on your proxy. Pick one under Model, or omit model to use the proxy's configured default."
              : "Sends requests through /typesafe/v1/systemone and requires TYPESAFE_API_KEY on the proxy."}{" "}
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
        <section className="flex min-h-96 flex-col xl:min-h-0" aria-label="System One request editor">
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
              <SystemOneForm
                value={rawPayload}
                onChange={handlePayloadChange}
                validation={validation}
                onOpenJson={() => setView("json")}
              />
            </TabsContent>
            <TabsContent value="json" className="flex min-h-0 flex-col">
              <JsonEditor value={rawPayload} onChange={handlePayloadChange} validation={validation} />
            </TabsContent>
          </Tabs>
        </section>
        <section className="grid content-start gap-4 xl:min-h-0 xl:overflow-auto" aria-label="System One results">
          <ResponseView
            response={systemOne.data?.response}
            fallbackModel={validation.payload?.model}
            latencyMs={systemOne.data?.latencyMs}
            error={systemOne.error?.message}
            isLoading={isLoading}
          />
          <QuestionBreakdown payload={validation.payload} />
        </section>
      </div>
    </div>
  );
}
