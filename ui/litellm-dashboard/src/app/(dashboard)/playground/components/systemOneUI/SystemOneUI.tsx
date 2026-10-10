"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { DECISIONS_DOCS_URL } from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";
import { useMutation } from "@tanstack/react-query";
import { Code, Info, LoaderCircle, RotateCcw, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makeSystemOneRequest } from "../../llm_calls/system_one";
import {
  PLACEHOLDER_DECISION_MODEL,
  SYSTEM_ONE_EXAMPLE,
  decisionsExample,
  openAIDecisionsExample,
} from "./lib/example";
import { formFromJson, formIssues, formToJson, type DecisionsForm as DecisionsFormValue } from "./lib/form";
import { payloadModel, withPayloadModel } from "./lib/payloadModel";
import type { DecisionEndpoint, PlaygroundRequest } from "./lib/schemas";
import DecisionsForm from "./DecisionsForm";
import JsonEditor from "./JsonEditor";
import QuestionBreakdown from "./QuestionBreakdown";
import ResponseView from "./ResponseView";
import { validateSystemOnePayload } from "./lib/validatePayload";
import { useDecisionModels, type ApiKeySource } from "./useDecisionModels";

interface SystemOneUIProps {
  accessToken: string | null;
  disabledPersonalKeyCreation?: boolean;
}

interface SystemOneSendVariables {
  payload: PlaygroundRequest;
  endpoint: DecisionEndpoint;
  apiKey: string;
  signal: AbortSignal;
}

const EXAMPLE_PAYLOAD = JSON.stringify(SYSTEM_ONE_EXAMPLE, null, 2);
const ENDPOINT_LABELS: Record<DecisionEndpoint, string> = {
  "/v1/decisions": "Decisions · /v1/decisions",
  "/v1/systemone": "System One · /v1/systemone",
  "/typesafe/v1/systemone": "TypeSafe · /typesafe/v1/systemone",
};
const ENDPOINT_NOTICES: Record<DecisionEndpoint, { title: string; body: string }> = {
  "/v1/decisions": {
    title: "Decision models · OpenAI Decisions shape",
    body: "Sends choice, predicate, and score questions through /v1/decisions to a decision model on your proxy. Pick one under Model, or omit model to use the proxy's configured default.",
  },
  "/v1/systemone": {
    title: "Decision models · System One shape",
    body: "Sends choice, noul, and score questions through /v1/systemone to a decision model on your proxy. Pick one under Model, or omit model to use the proxy's configured default.",
  },
  "/typesafe/v1/systemone": {
    title: "TypeSafe Jev · System One",
    body: "Sends requests through /typesafe/v1/systemone and requires TYPESAFE_API_KEY on the proxy.",
  },
};
function examplePayloadFor(endpoint: DecisionEndpoint, model: string): string {
  if (endpoint === "/typesafe/v1/systemone") {
    return EXAMPLE_PAYLOAD;
  }
  const example = endpoint === "/v1/decisions" ? openAIDecisionsExample(model) : decisionsExample(model);
  return JSON.stringify(example, null, 2);
}
const isDecisionEndpoint = (value: string): value is DecisionEndpoint => Object.hasOwn(ENDPOINT_LABELS, value);
const usesProxyModels = (endpoint: DecisionEndpoint): boolean => endpoint !== "/typesafe/v1/systemone";
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
    chosenEndpoint ?? (decisionModels.length > 0 ? "/v1/decisions" : "/typesafe/v1/systemone");
  const [drafts, setDrafts] = useState<Partial<Record<DecisionEndpoint, string>>>({});
  const [forms, setForms] = useState<Partial<Record<DecisionEndpoint, DecisionsFormValue>>>({});
  const [editor, setEditor] = useState<"form" | "json">("form");
  const examplePayload = examplePayloadFor(endpoint, decisionModels[0] ?? PLACEHOLDER_DECISION_MODEL);
  const rawPayload = drafts[endpoint] ?? examplePayload;
  const parsedForm = useMemo(() => formFromJson(rawPayload), [rawPayload]);
  const form = forms[endpoint] ?? (parsedForm.ok ? parsedForm.form : undefined);
  const activeEditor = editor === "form" && form !== undefined ? "form" : "json";
  const formOnlyIssues = activeEditor === "form" && form !== undefined ? formIssues(form, endpoint) : [];
  const formUnavailableReason =
    editor === "form" && !parsedForm.ok && forms[endpoint] === undefined ? parsedForm.reason : null;
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

  function handleFormChange(next: DecisionsFormValue) {
    clearRequestState();
    setChosenEndpoint(endpoint);
    setForms((current) => ({ ...current, [endpoint]: next }));
    setDrafts((current) => ({ ...current, [endpoint]: formToJson(next, endpoint) }));
  }

  function handleJsonChange(value: string) {
    setForms((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== endpoint)));
    handlePayloadChange(value);
  }

  function handleResetExample() {
    clearRequestState();
    setChosenEndpoint(endpoint);
    setDrafts((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== endpoint)));
    setForms((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== endpoint)));
  }

  function handleModelPick(model: string | null) {
    if (activeEditor === "form" && form !== undefined) {
      handleFormChange({ ...form, model: model ?? undefined });
      return;
    }
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
              if (value !== null && isDecisionEndpoint(value)) {
                clearRequestState();
                setChosenEndpoint(value);
              }
            }}
          >
            <SelectTrigger className="w-80" aria-label="Decision endpoint">
              <SelectValue>{ENDPOINT_LABELS[endpoint]}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {Object.entries(ENDPOINT_LABELS).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {usesProxyModels(endpoint) && (
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
            <span className="text-sm font-medium text-muted-foreground">Editor</span>
            <Tabs
              value={activeEditor}
              onValueChange={(value) => {
                if (value === "form" || value === "json") {
                  setEditor(value);
                }
              }}
            >
              <TabsList aria-label="Request editor">
                <TabsTrigger value="form">Form</TabsTrigger>
                <TabsTrigger value="json">JSON</TabsTrigger>
              </TabsList>
            </Tabs>
            <Button variant="outline" onClick={handleResetExample} disabled={drafts[endpoint] === undefined}>
              <RotateCcw />
              Reset example
            </Button>
            {activeEditor === "json" && (
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
            <Button
              onClick={handleSend}
              disabled={!validation.isValid || formOnlyIssues.length > 0 || isLoading || !effectiveApiKey}
            >
              {isLoading ? <LoaderCircle className="animate-spin" /> : <Send />}
              Send
            </Button>
          </div>
        </div>
        <Alert role="note" aria-label="Decision endpoint notice">
          <Info />
          <AlertTitle>{ENDPOINT_NOTICES[endpoint].title}</AlertTitle>
          <AlertDescription>
            {ENDPOINT_NOTICES[endpoint].body}{" "}
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
        <section className="flex min-h-96 flex-col gap-2 xl:min-h-0" aria-label="Decisions request editor">
          {formUnavailableReason !== null && (
            <p role="status" className="text-xs text-muted-foreground">
              {formUnavailableReason}. Showing JSON instead
            </p>
          )}
          {activeEditor === "form" && form !== undefined ? (
            <DecisionsForm
              form={form}
              endpoint={endpoint}
              issues={[...formOnlyIssues, ...validation.issues.map((issue) => issue.message)]}
              onChange={handleFormChange}
            />
          ) : (
            <JsonEditor value={rawPayload} onChange={handleJsonChange} validation={validation} />
          )}
        </section>
        <section className="grid content-start gap-4 xl:min-h-0 xl:overflow-auto" aria-label="Decisions results">
          <ResponseView
            response={systemOne.data?.response}
            fallbackModel={validation.payload?.model}
            latencyMs={systemOne.data?.latencyMs}
            error={systemOne.error?.message}
            isLoading={isLoading}
          />
          {activeEditor === "json" && <QuestionBreakdown payload={validation.payload} />}
        </section>
      </div>
    </div>
  );
}
