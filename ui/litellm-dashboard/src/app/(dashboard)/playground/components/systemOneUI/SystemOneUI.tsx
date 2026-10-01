"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useMutation } from "@tanstack/react-query";
import { Code, Info, LoaderCircle, Send } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makeSystemOneRequest } from "../../llm_calls/system_one";
import { SYSTEM_ONE_PRESETS } from "./system_one_presets";
import type { SystemOneRequest } from "./system_one_schemas";
import SystemOneQuestionBreakdown from "./SystemOneQuestionBreakdown";
import SystemOneResponseView from "./SystemOneResponseView";
import { validateSystemOnePayload } from "./validate_system_one_payload";

interface SystemOneUIProps {
  accessToken: string | null;
  disabledPersonalKeyCreation?: boolean;
}

type ApiKeySource = "session" | "custom";

interface SystemOneSendVariables {
  payload: SystemOneRequest;
  apiKey: string;
  signal: AbortSignal;
}

const INITIAL_PRESET = SYSTEM_ONE_PRESETS[0];

function getCustomProxyBaseUrl(): string | undefined {
  return typeof window === "undefined" ? undefined : window.sessionStorage.getItem("customProxyBaseUrl") || undefined;
}

export default function SystemOneUI({ accessToken, disabledPersonalKeyCreation = false }: SystemOneUIProps) {
  const [apiKeySource, setApiKeySource] = useState<ApiKeySource>(disabledPersonalKeyCreation ? "custom" : "session");
  const [customApiKey, setCustomApiKey] = useState("");
  const [selectedPresetId, setSelectedPresetId] = useState(INITIAL_PRESET.id);
  const [rawPayload, setRawPayload] = useState(() => JSON.stringify(INITIAL_PRESET.payload, null, 2));
  const activeController = useRef<AbortController | null>(null);
  const validation = useMemo(() => validateSystemOnePayload(rawPayload), [rawPayload]);
  const effectiveApiKey = apiKeySource === "session" ? accessToken || "" : customApiKey.trim();
  const hasSyntaxError = validation.issues.some((issue) => issue.path === "syntax");

  const systemOne = useMutation({
    mutationFn: ({ payload, apiKey, signal }: SystemOneSendVariables) =>
      makeSystemOneRequest(payload, apiKey, getCustomProxyBaseUrl(), signal),
  });
  const isLoading = systemOne.isPending;

  useEffect(() => () => activeController.current?.abort(), []);

  function clearRequestState() {
    activeController.current?.abort();
    activeController.current = null;
    systemOne.reset();
  }

  function handlePayloadChange(value: string) {
    if (value !== rawPayload) {
      clearRequestState();
      setRawPayload(value);
    }
  }

  function handlePresetChange(value: string | null) {
    const preset = SYSTEM_ONE_PRESETS.find((entry) => entry.id === value);
    if (!preset) {
      return;
    }
    handlePayloadChange(JSON.stringify(preset.payload, null, 2));
    setSelectedPresetId(preset.id);
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
    const controller = new AbortController();
    activeController.current = controller;
    systemOne.mutate({ payload: validation.payload, apiKey: effectiveApiKey, signal: controller.signal });
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-auto p-4">
      <section className="grid gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Virtual Key Source</span>
            <Select
              value={apiKeySource}
              onValueChange={(value) => {
                if (value === "session" || value === "custom") {
                  clearRequestState();
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
                onChange={(event) => {
                  clearRequestState();
                  setCustomApiKey(event.target.value);
                }}
                placeholder="Enter Virtual Key"
                className="w-56"
              />
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">Preset</span>
            <Select value={selectedPresetId} onValueChange={handlePresetChange}>
              <SelectTrigger className="min-w-56" aria-label="System One preset">
                <SelectValue>{SYSTEM_ONE_PRESETS.find((preset) => preset.id === selectedPresetId)?.name}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                {SYSTEM_ONE_PRESETS.map((preset) => (
                  <SelectItem key={preset.id} value={preset.id}>
                    {preset.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button variant="outline" onClick={handleFormatJson} disabled={!rawPayload.trim() || hasSyntaxError}>
              <Code />
              Format JSON
            </Button>
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
        <Alert role="note" aria-label="System One beta notice">
          <Info />
          <AlertTitle>Beta: TypeSafe Jev only for now</AlertTitle>
          <AlertDescription>
            Sends System One requests (choice, noul, score) through /typesafe/v1/systemone and requires TYPESAFE_API_KEY
            on the proxy. Support for more System One-compatible models is in progress.
          </AlertDescription>
        </Alert>
      </section>

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-2">
        <section className="flex min-h-96 flex-col gap-3" aria-label="System One request editor">
          <Textarea
            aria-label="System One JSON payload"
            value={rawPayload}
            onChange={(event) => handlePayloadChange(event.target.value)}
            className="min-h-80 flex-1 resize-y font-mono text-xs"
            spellCheck={false}
          />
          {validation.issues.length > 0 && (
            <div className="grid gap-2" aria-label="Payload validation issues">
              {validation.issues.map((issue, index) => (
                <Alert key={`${issue.path}-${index}`} variant={issue.severity === "error" ? "destructive" : "default"}>
                  <AlertTitle>
                    {issue.severity === "error" ? "Error" : "Warning"}: {issue.path}
                  </AlertTitle>
                  <AlertDescription>{issue.message}</AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </section>
        <section className="grid content-start gap-4" aria-label="System One results">
          <SystemOneQuestionBreakdown payload={validation.payload} />
          <SystemOneResponseView
            response={systemOne.data?.response}
            fallbackModel={validation.payload?.model}
            latencyMs={systemOne.data?.latencyMs}
            error={systemOne.error?.message}
            isLoading={isLoading}
          />
        </section>
      </div>
    </div>
  );
}
