"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { DurationInput } from "./DurationInput";
import { ActivityScope, type ActivitySelection } from "./ActivityScope";
import {
  analysisModelOptions,
  durationLabel,
  normalizeFilters,
  scopeLabel,
  type AnalysisModelInfo,
  type Settings,
} from "./lensData";

function validateSample(selection: ActivitySelection) {
  const hours = selection.lookback_hours ?? 24;
  if (!Number.isInteger(hours) || hours < 1 || hours > 8760)
    throw new Error("Choose a time range between 1 hour and 365 days");
  const percent = selection.sample_percent ?? 100;
  if (!Number.isFinite(percent) || percent <= 0 || percent > 100)
    throw new Error("Choose a sampling percentage greater than 0 and up to 100");
  if (selection.sample_size != null && (!Number.isInteger(selection.sample_size) || selection.sample_size < 1))
    throw new Error("Choose a positive maximum or leave it blank for no limit");
}

export function LensSetup({
  initial,
  mode = initial ? "edit" : "new",
  models,
  modelDetails = [],
  modelsLoading = false,
  modelsError,
  accessToken,
  ready = true,
  onClose,
  onSave,
}: {
  initial?: Settings;
  mode?: "new" | "edit" | "duplicate";
  models: string[];
  modelDetails?: AnalysisModelInfo[];
  modelsLoading?: boolean;
  modelsError?: string;
  accessToken: string;
  ready?: boolean;
  onClose: () => void;
  onSave: (settings: Settings) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [previewReady, setPreviewReady] = useState(false);
  const [manualSelection, setManualSelection] = useState(!!initial?.execution_ids?.length);
  const [name, setName] = useState(initial?.name ?? "");
  const initialSelection: Required<ActivitySelection> = {
    source: initial?.source ?? "traces",
    service: initial?.service ?? "",
    filters: initial?.filters ?? [],
    lookback_hours: initial?.lookback_hours ?? 24,
    sample_size: initial?.sample_size ?? null,
    sample_percent: initial?.sample_percent ?? 100,
    team_id: initial?.team_id ?? "",
    execution_ids: initial?.execution_ids ?? [],
  };
  const [selection, setSelection] = useState(initialSelection);
  const [context, setContext] = useState(initial?.context ?? "");
  const [questions, setQuestions] = useState(initial?.checks?.map((c) => c.instruction).join("\n") ?? "");
  const [model, setModel] = useState(initial?.model ?? "");
  const [budget, setBudget] = useState(initial?.monthly_budget ?? 100);
  const [repeat, setRepeat] = useState(mode === "edit" && !!initial?.enabled);
  const [interval, setInterval] = useState(initial?.interval_minutes ?? 30);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const suggestedName = questions.trim().split("\n")[0] || context.trim().split("\n")[0] || "Investigation";
  const title = name.trim() || suggestedName.slice(0, 100);
  const changeSelection = (next: ActivitySelection) => {
    const pool = (s: ActivitySelection) =>
      JSON.stringify([s.source, s.service, s.filters, s.lookback_hours, s.team_id]);
    setSelection({
      ...selection,
      ...next,
      execution_ids: pool(next) === pool(selection) ? next.execution_ids ?? [] : [],
    });
  };
  const validate = () => {
    normalizeFilters(selection.filters ?? []);
    if (step >= 2 && manualSelection && !selection.execution_ids?.length)
      throw new Error("Choose at least one run or turn off individual selection");
    validateSample(selection);
    if (step >= 1 && !context.trim() && !questions.trim())
      throw new Error("Describe the expected behavior or what to look out for");
    if (questions.split("\n").some((q) => q.trim() && q.trim().length < 3))
      throw new Error("Use at least three characters for each check");
  };
  const next = () => {
    try {
      validate();
      setError("");
      setStep(step + 1);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Check your settings");
    }
  };
  const save = async () => {
    setBusy(true);
    setError("");
    try {
      validate();
      const settings: Settings = {
        ...initial,
        ...selection,
        name: title,
        context: context.trim(),
        model,
        monthly_budget: budget,
        enabled: repeat,
        interval_minutes: interval,
        concurrency: initial?.concurrency ?? 8,
        filters: normalizeFilters(selection.filters ?? []),
        checks: questions
          .split("\n")
          .map((q) => q.trim())
          .filter(Boolean)
          .map(
            (instruction) =>
              initial?.checks?.find((c) => c.instruction === instruction) ?? {
                id: crypto.randomUUID(),
                instruction,
                enabled: true,
              },
          ),
      };
      await onSave(settings);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save investigation");
    } finally {
      setBusy(false);
    }
  };
  const unsupported = modelDetails.some((m) => m.model_group === model && m.mode && m.mode !== "chat");
  const budgetValid = Number.isFinite(budget) && budget > 0 && budget <= 100000;
  const canRun = ready || mode === "edit";
  const modelValid = !!model && !unsupported;
  const intervalRangeValid = interval >= 1 && interval <= 10080;
  const intervalValid = !repeat || (Number.isInteger(interval) && intervalRangeValid);
  const configurationValid = modelValid && budgetValid && intervalValid;
  const canSave = !busy && canRun && configurationValid;
  const createLabel = repeat ? "Run and monitor" : "Run investigation";
  const saveLabel = mode === "edit" ? "Save changes" : createLabel;
  const headings = [
    "Which activity should we investigate?",
    "What should Lens look for?",
    "How much activity should we review?",
    mode === "edit" ? "Review changes" : "Ready to investigate",
  ];
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[90dvh] flex-col gap-6 overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="text-xl">{headings[step]}</DialogTitle>
          <DialogDescription>
            {
              [
                "Start with an agent, or use filters to investigate any recorded activity.",
                "Describe the expected behavior, the questions you have, or both.",
                "Choose a time range and how much of it to review.",
                "Your selected model reviews the activity through LiteLLM.",
              ][step]
            }
          </DialogDescription>
        </DialogHeader>
        <nav aria-label="Investigation setup" className="flex gap-2 text-xs">
          {["Activity", "Expectations", "Sample", "Review"].map((label, index) => (
            <button
              key={label}
              disabled={index > step || busy}
              aria-current={index === step ? "step" : undefined}
              onClick={() => {
                setStep(index);
                setError("");
              }}
              className={`flex-1 border-t-2 pt-2 text-left ${index === step ? "border-foreground font-medium" : "border-border text-muted-foreground"}`}
            >
              {index + 1}. {label}
            </button>
          ))}
        </nav>
        <div className="min-h-0 overflow-y-auto pr-1 space-y-5">
          {(step === 0 || step === 2) && (
            <ActivityScope
              key={step}
              value={selection}
              onChange={changeSelection}
              accessToken={accessToken}
              nameField={
                step === 0 ? (
                  <label className="grid gap-2 text-sm font-medium">
                    Investigation name
                    <Input
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="e.g. Support quality"
                      maxLength={100}
                    />
                  </label>
                ) : undefined
              }
              mode={step === 0 ? "scope" : "activity"}
              onPreviewReady={setPreviewReady}
              manualSelection={manualSelection}
              onManualSelection={setManualSelection}
            />
          )}
          {step === 1 && (
            <>
              <label className="grid gap-2 text-sm font-medium">
                What should the agent be doing?
                <Textarea
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  maxLength={6000}
                  rows={4}
                  placeholder="Answer the customer's question using verified sources and explain when information is missing."
                />
              </label>
              <label className="grid gap-2 text-sm font-medium">
                What should we look out for?
                <Textarea
                  value={questions}
                  onChange={(e) => setQuestions(e.target.value)}
                  rows={4}
                  placeholder="Repeated searches that do not add useful information.&#10;Answers that contradict the sources."
                />
              </label>
              <p className="text-xs text-muted-foreground">One check per line. You can leave either field blank.</p>
              <Button
                variant="link"
                className="h-auto px-0"
                onClick={() => {
                  if (!context.trim())
                    setContext(
                      "Answer the user's question using verified sources. Explain when information is missing.",
                    );
                  if (!questions.trim())
                    setQuestions(
                      "Find repeated work that adds no useful information.\nFind claims that contradict the available evidence.",
                    );
                }}
              >
                Use an example
              </Button>
            </>
          )}
          {step === 3 && (
            <>
              <div className="space-y-2 border-b pb-4 text-sm">
                <p className="font-medium">{title}</p>
                <p className="text-muted-foreground">
                  {scopeLabel(selection)} · Last {durationLabel(selection.lookback_hours ?? 24, "hours")}
                </p>
                <p className="text-muted-foreground">
                  {selection.sample_percent ?? 100}% sample
                  {selection.sample_size ? `, up to ${selection.sample_size} runs` : ", no count limit"}
                  {selection.execution_ids?.length ? ` · ${selection.execution_ids.length} manually selected` : ""}
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Analysis model</p>
                <SearchSelect
                  aria-label="Analysis model"
                  options={analysisModelOptions(models, modelDetails)}
                  value={model}
                  onValueChange={(value) => setModel(value ?? "")}
                  placeholder={modelsLoading ? "Loading models…" : "Choose a model"}
                  disabled={modelsLoading}
                  emptyText="No matching models configured on this gateway"
                />
                {modelsError && (
                  <p role="alert" className="text-sm text-destructive">
                    Could not load models: {modelsError}
                  </p>
                )}
                {unsupported && (
                  <p role="alert" className="text-sm text-destructive">
                    Choose a chat model that supports JSON output.
                  </p>
                )}
              </div>
              <p className="text-sm text-muted-foreground">
                {repeat ? `Repeats every ${durationLabel(interval, "minutes")}` : "Runs once"} · ${budget} monthly limit
              </p>
              <details>
                <summary className="cursor-pointer text-sm font-medium">Advanced options</summary>
                <div className="mt-4 grid gap-5 sm:grid-cols-2">
                  <label className="grid content-start gap-2 text-sm font-medium">
                    Monthly limit (USD)
                    <Input
                      type="number"
                      min="0.01"
                      max="100000"
                      step="1"
                      value={budget}
                      onChange={(e) => setBudget(Number(e.target.value))}
                    />
                  </label>
                  <div className="space-y-3">
                    <label className="flex items-center gap-2 text-sm font-medium">
                      <input
                        type="checkbox"
                        checked={repeat}
                        onChange={(e) => setRepeat(e.target.checked)}
                        className="size-4 rounded border-input accent-foreground"
                      />
                      Repeat this investigation
                    </label>
                    {repeat && (
                      <DurationInput
                        label="Repeat every"
                        value={interval}
                        onChange={setInterval}
                        base="minutes"
                        max={10080}
                      />
                    )}
                  </div>
                </div>
              </details>
              <p className="text-xs leading-5 text-muted-foreground">
                Analysis is charged to your worker&apos;s virtual key; its permissions and limits apply.
              </p>
            </>
          )}
          {!ready && mode !== "edit" && (
            <p role="status" className="text-sm text-amber-700">
              The worker or trace storage is unavailable. Your draft is safe; you can start when it reconnects.
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter className="border-t pt-4">
          <Button variant="outline" disabled={busy} onClick={() => (step ? setStep(step - 1) : onClose())}>
            {step ? "Back" : "Cancel"}
          </Button>
          {step < 3 ? (
            <Button disabled={step === 2 && !previewReady} onClick={next}>
              Continue
            </Button>
          ) : (
            <Button disabled={!canSave} onClick={() => void save()}>
              {busy ? "Saving…" : saveLabel}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
