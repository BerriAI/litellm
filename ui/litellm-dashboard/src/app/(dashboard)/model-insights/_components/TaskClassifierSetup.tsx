"use client";

import React from "react";
import { Sparkles } from "lucide-react";

import { apiClient } from "@/components/networking";
import { extractErrorMessage } from "@/utils/errorUtils";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ClassifierProvider, TaskClassifierProviderOption, TaskClassifierStatus } from "./modelInsightsData";

const CLASSIFIER_PATH = "/model-insights/task-classifier";

const firstReady = (providers: TaskClassifierProviderOption[]) =>
  providers.find((option) => option.ready) ?? providers[0] ?? null;

type ClassifierFormProps = {
  status: TaskClassifierStatus;
  saving: boolean;
  onSave: (provider: ClassifierProvider, model: string) => void;
  onCancel?: () => void;
};

const ClassifierForm = ({ status, saving, onSave, onCancel }: ClassifierFormProps) => {
  const initial =
    status.providers.find((option) => option.provider === status.configured?.provider) ?? firstReady(status.providers);
  const [provider, setProvider] = React.useState<ClassifierProvider | null>(initial?.provider ?? null);
  const selected = status.providers.find((option) => option.provider === provider) ?? null;
  const [model, setModel] = React.useState<string | null>(
    status.configured?.provider === provider ? status.configured.model : selected?.models[0] ?? null,
  );
  const selectProvider = (value: ClassifierProvider | null) => {
    const next = status.providers.find((option) => option.provider === value);
    setProvider(next?.provider ?? null);
    setModel(next?.models[0] ?? null);
  };
  const providerItems = status.providers.map((option) => ({ label: option.label, value: option.provider }));
  const modelItems = (selected?.models ?? []).map((name) => ({ label: name, value: name }));

  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
        System One provider
        <Select items={providerItems} value={provider} onValueChange={selectProvider}>
          <SelectTrigger className="w-52" aria-label="Classifier provider">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {status.providers.map((option) => (
              <SelectItem key={option.provider} value={option.provider} disabled={!option.ready}>
                {option.label}
                {!option.ready && ` (set ${option.missing_env.join(", ")})`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      <label className="flex flex-col gap-1 text-xs font-medium text-muted-foreground">
        Model
        <Select items={modelItems} value={model} onValueChange={(value) => setModel(value)}>
          <SelectTrigger className="w-56" aria-label="Classifier model">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {modelItems.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </label>
      <Button
        disabled={saving || !selected?.ready || !model}
        onClick={() => provider && model && onSave(provider, model)}
      >
        {saving ? "Saving..." : "Enable classifier"}
      </Button>
      {onCancel && (
        <Button variant="ghost" onClick={onCancel} disabled={saving}>
          Cancel
        </Button>
      )}
    </div>
  );
};

type TaskClassifierSetupProps = {
  accessToken: string;
  status: TaskClassifierStatus;
  onChange: (status: TaskClassifierStatus) => void;
};

export default function TaskClassifierSetup({ accessToken, status, onChange }: TaskClassifierSetupProps) {
  const [editing, setEditing] = React.useState(false);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const configured = status.configured;
  const configuredLabel = status.providers.find((option) => option.provider === configured?.provider)?.label;

  const submit = (request: Promise<TaskClassifierStatus>) => {
    setSaving(true);
    request
      .then((next) => {
        setError(null);
        setEditing(false);
        onChange(next);
      })
      .catch((err: unknown) => setError(extractErrorMessage(err)))
      .finally(() => setSaving(false));
  };
  const save = (provider: ClassifierProvider, model: string) =>
    submit(apiClient.put<TaskClassifierStatus>(CLASSIFIER_PATH, { accessToken, body: { provider, model } }));
  const disable = () => submit(apiClient.delete<TaskClassifierStatus>(CLASSIFIER_PATH, { accessToken }));

  const errorAlert = error && (
    <Alert variant="destructive">
      <AlertTitle>Could not update the task classifier</AlertTitle>
      <AlertDescription>{error}</AlertDescription>
    </Alert>
  );

  if (!configured) {
    return (
      <div className="space-y-3">
        <Alert>
          <Sparkles />
          <AlertTitle>Task classifier is not set up</AlertTitle>
          <AlertDescription>
            Pick a System One model and LiteLLM will label each logged request with its task in background batches, off
            the request path. Requests logged before setup stay Uncategorized
          </AlertDescription>
        </Alert>
        <ClassifierForm status={status} saving={saving} onSave={save} />
        {errorAlert}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {editing ? (
        <ClassifierForm status={status} saving={saving} onSave={save} onCancel={() => setEditing(false)} />
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <span className="flex items-center gap-2 text-muted-foreground">
            <Sparkles className="size-4" />
            Tasks auto-classified by {configuredLabel ?? configured.provider} ·{" "}
            <span className="font-medium text-foreground">{configured.model}</span>
          </span>
          <span className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setEditing(true)} disabled={saving}>
              Change
            </Button>
            <Button variant="ghost" size="sm" onClick={disable} disabled={saving}>
              Turn off
            </Button>
          </span>
        </div>
      )}
      {errorAlert}
    </div>
  );
}
