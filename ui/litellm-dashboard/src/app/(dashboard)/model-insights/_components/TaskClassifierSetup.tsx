"use client";

import Link from "next/link";
import React from "react";

import { apiClient } from "@/components/networking";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { extractErrorMessage } from "@/utils/errorUtils";
import { uiHref } from "@/utils/uiHref";

export interface TaskClassifierModel {
  id: string;
  name: string;
  provider: "typesafe" | "laya" | "bespoke";
  model: string;
}

export interface TaskClassifierSettings {
  enabled: boolean;
  model_id: string | null;
  models: TaskClassifierModel[];
  batch_size: number;
  message_logging_enabled: boolean;
}

export const TASK_CLASSIFIER_ENDPOINT = "/model-insights/task-classifier";

const PROVIDER_LABELS: Record<TaskClassifierModel["provider"], string> = {
  typesafe: "TypeSafe Jev",
  laya: "Laya",
  bespoke: "Bespoke Nimble",
};

export default function TaskClassifierSetup({
  accessToken,
  canEdit,
  onSaved,
}: {
  accessToken: string | null;
  canEdit: boolean;
  onSaved?: (settings: TaskClassifierSettings) => void;
}) {
  const [settings, setSettings] = React.useState<TaskClassifierSettings | null>(null);
  const [modelId, setModelId] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);

  React.useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    apiClient
      .get<TaskClassifierSettings>(TASK_CLASSIFIER_ENDPOINT, { accessToken })
      .then((response) => {
        if (cancelled) return;
        setSettings(response);
        setModelId(response.model_id ?? response.models[0]?.id ?? null);
        setError(null);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(extractErrorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const save = async (enabled: boolean) => {
    if (!accessToken) return;
    setSaving(true);
    setError(null);
    try {
      const response = await apiClient.put<TaskClassifierSettings>(TASK_CLASSIFIER_ENDPOINT, {
        accessToken,
        body: { enabled, model_id: enabled ? modelId : null },
      });
      setSettings(response);
      setModelId(response.model_id ?? modelId);
      onSaved?.(response);
    } catch (err: unknown) {
      setError(extractErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  if (!settings) {
    return error ? (
      <Alert variant="destructive">
        <AlertTitle>Could not load task classifier settings</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    ) : null;
  }

  const selected = settings.models.find((model) => model.id === settings.model_id);
  const validChoice = settings.models.some((model) => model.id === modelId);

  if (settings.enabled && selected) {
    return (
      <section
        aria-label="Task classifier"
        className="flex flex-wrap items-center justify-between gap-3 rounded-md border p-3"
      >
        <p className="text-sm">
          Classifying untagged requests with <span className="font-medium">{selected.name}</span> (
          {PROVIDER_LABELS[selected.provider]} {selected.model}) in background batches of up to {settings.batch_size}
        </p>
        {canEdit && (
          <Button variant="outline" size="sm" disabled={saving} onClick={() => void save(false)}>
            Turn off
          </Button>
        )}
        {error && <p className="w-full text-sm text-destructive">{error}</p>}
      </section>
    );
  }

  return (
    <Alert>
      <AlertTitle>Task classification has not been set up</AlertTitle>
      <AlertDescription className="space-y-3">
        <p>
          {settings.enabled
            ? "The saved System One model is no longer configured, so new requests are not being classified."
            : "Requests without a task: tag show up as Uncategorized."}{" "}
          Pick a System One model and the gateway will label new requests in the background, in batches of up to{" "}
          {settings.batch_size}, without slowing the request path. Requests that already carry a task: tag keep it
        </p>
        <p>
          The last user message of each untagged request is sent to the model you pick, which can add cost. Past
          requests are not backfilled
        </p>
        {!settings.message_logging_enabled && (
          <p className="font-medium">
            Prompt storage in spend logs is off, so there is nothing to classify yet. Turn on
            store_prompts_in_spend_logs to let the classifier see request text
          </p>
        )}
        {settings.models.length === 0 && (
          <p>
            No System One model is configured. Add a TypeSafe Jev, Laya or Bespoke Nimble model in{" "}
            <Link className="underline underline-offset-2" href={uiHref("models-and-endpoints")}>
              Models + Endpoints
            </Link>
            , then come back here
          </p>
        )}
        {settings.models.length > 0 && canEdit && (
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={validChoice ? modelId : null}
              items={settings.models.map((model) => ({ value: model.id, label: model.name }))}
              onValueChange={(value) => setModelId(value)}
            >
              <SelectTrigger className="w-72" aria-label="System One model">
                <SelectValue placeholder="Select a System One model" />
              </SelectTrigger>
              <SelectContent>
                {settings.models.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name} · {PROVIDER_LABELS[model.provider]} {model.model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button disabled={!validChoice || saving} onClick={() => void save(true)}>
              {saving ? "Saving" : "Turn on task classification"}
            </Button>
          </div>
        )}
        {settings.models.length > 0 && !canEdit && <p>Ask a proxy admin to turn on task classification</p>}
        {error && <p className="text-destructive">{error}</p>}
      </AlertDescription>
    </Alert>
  );
}
