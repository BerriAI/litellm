"use client";

import { useState } from "react";
import { X } from "lucide-react";
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
import { WatchPicker } from "./WatchPicker";
import {
  analysisModelOptions,
  initialWatches,
  isWatch,
  normalizeFilters,
  watchChecks,
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

function newCheck(instruction = ""): Settings["checks"][number] {
  return { id: crypto.randomUUID(), instruction, enabled: true };
}

export function LensSetup({
  initial,
  mode = initial ? "edit" : "new",
  models,
  modelDetails = [],
  modelsLoading = false,
  modelsError,
  defaultModel,
  defaultSource = "traces",
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
  defaultModel?: string;
  defaultSource?: Settings["source"];
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
    source: initial?.source ?? defaultSource,
    service: initial?.service ?? "",
    agent_name: initial?.agent_name ?? "",
    filters: initial?.filters ?? [],
    lookback_hours: initial?.lookback_hours ?? 24,
    sample_size: initial?.sample_size ?? null,
    sample_percent: initial?.sample_percent ?? 100,
    team_id: initial?.team_id ?? "",
    execution_ids: initial?.execution_ids ?? [],
  };
  const [selection, setSelection] = useState(initialSelection);
  const [context, setContext] = useState(initial?.context ?? "");
  const [watching, setWatching] = useState<ReadonlySet<string>>(() => initialWatches(initial?.checks));
  const [questions, setQuestions] = useState(() => (initial?.checks ?? []).filter((check) => !isWatch(check)));
  const [selectedModel, setModel] = useState<string | null>(initial?.model ?? null);
  const model = selectedModel ?? defaultModel ?? "";
  const [budget, setBudget] = useState(initial?.monthly_budget ?? 100);
  const [repeat, setRepeat] = useState(mode === "edit" && !!initial?.enabled);
  const [interval, setInterval] = useState(initial?.interval_minutes ?? 30);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const filledChecks = questions.filter((check) => check.instruction.trim());
  const suggestedName = filledChecks[0]?.instruction.trim() || context.trim().split("\n")[0] || "Investigation";
  const title = name.trim() || suggestedName.slice(0, 100);
  const changeSelection = (next: ActivitySelection) => {
    const pool = (s: ActivitySelection) =>
      JSON.stringify([s.source, s.service, s.agent_name, s.filters, s.lookback_hours, s.team_id]);
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
    const nothingToCheck = !context.trim() && !filledChecks.length && !watching.size;
    if (step >= 1 && nothingToCheck) throw new Error("Describe the expected behavior or pick something to watch for");
    if (filledChecks.some((check) => check.instruction.trim().length < 3))
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
        checks: [
          ...watchChecks(watching),
          ...filledChecks.map((check) => ({ ...check, instruction: check.instruction.trim() })),
        ],
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
  const modelsReady = !modelsLoading && !modelsError;
  const unavailable = !!model && modelsReady && !models.includes(model);
  const supported = !unsupported && !unavailable;
  const preservingSavedModel = mode === "edit" && model === initial?.model;
  const modelReady = modelsReady || preservingSavedModel;
  const modelValid = !!model && supported && modelReady;
  const intervalRangeValid = interval >= 1 && interval <= 10080;
  const intervalValid = !repeat || (Number.isInteger(interval) && intervalRangeValid);
  const configurationValid = modelValid && budgetValid && intervalValid;
  const runReady = canRun && (mode === "edit" || previewReady);
  const selectionValid = !manualSelection || !!selection.execution_ids?.length;
  const validSettings = configurationValid && selectionValid;
  const canSave = !busy && runReady && validSettings;
  const createLabel = repeat ? "Run and monitor" : "Run investigation";
  const saveLabel = mode === "edit" ? "Save changes" : createLabel;
  const headings = [
    "Which activity should we investigate?",
    "What should Lens look for?",
    mode === "edit" ? "Review changes" : "Ready to investigate",
  ];
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent
        className={`flex max-h-[90dvh] flex-col gap-6 overflow-hidden ${["sm:max-w-xl", "sm:max-w-2xl", "sm:max-w-3xl"][step]}`}
      >
        <DialogHeader>
          <DialogTitle className="text-xl">{headings[step]}</DialogTitle>
          <DialogDescription className="sr-only">
            {
              [
                "Start with an agent, or use filters to investigate any recorded activity.",
                "Describe the expected behavior, the questions you have, or both.",
                "Review the selected activity, then start your investigation.",
              ][step]
            }
          </DialogDescription>
        </DialogHeader>
        <nav aria-label="Investigation setup" className="flex gap-2 text-xs">
          {["Activity", "Expectations", "Run"].map((label, index) => (
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
              <WatchPicker
                selected={watching}
                onChange={setWatching}
                onAddCustom={() => setQuestions([...questions, newCheck()])}
              />
              <fieldset className="space-y-2">
                <legend className="sr-only">Custom checks</legend>
                {questions.map((check, index) => (
                  <div key={check.id} className="flex items-start gap-2">
                    <Textarea
                      aria-label={`Check ${index + 1}`}
                      value={check.instruction}
                      onChange={(event) =>
                        setQuestions(
                          questions.map((item) =>
                            item.id === check.id ? { ...item, instruction: event.target.value } : item,
                          ),
                        )
                      }
                      rows={2}
                      placeholder="e.g. Quotes a price without checking the pricing tool"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Remove check ${index + 1}`}
                      onClick={() => setQuestions(questions.filter((item) => item.id !== check.id))}
                    >
                      <X className="size-4" />
                    </Button>
                  </div>
                ))}
              </fieldset>
            </>
          )}
          {step === 2 && (
            <>
              <details open={!modelValid || undefined}>
                <summary className="cursor-pointer text-sm font-medium">Advanced options</summary>
                <div className="mt-4 space-y-5">
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
                    {unavailable && (
                      <p role="alert" className="text-sm text-destructive">
                        {model} is no longer available. Choose another analysis model.
                      </p>
                    )}
                    {unsupported && (
                      <p role="alert" className="text-sm text-destructive">
                        Choose a chat model that supports JSON output.
                      </p>
                    )}
                  </div>
                  <div className="grid gap-5 sm:grid-cols-2">
                    <label className="grid content-start gap-2 text-sm font-medium">
                      Maximum runs (optional)
                      <Input
                        type="number"
                        min="1"
                        placeholder="No limit"
                        value={selection.sample_size ?? ""}
                        onChange={(event) =>
                          setSelection({
                            ...selection,
                            sample_size: event.target.value ? Number(event.target.value) : null,
                          })
                        }
                      />
                    </label>
                    <label className="flex items-center gap-2 text-sm font-medium">
                      <input
                        type="checkbox"
                        checked={manualSelection}
                        onChange={(event) => {
                          setManualSelection(event.target.checked);
                          setSelection({ ...selection, execution_ids: [] });
                        }}
                      />
                      Choose individual runs
                    </label>
                  </div>
                  <div className="grid gap-5 sm:grid-cols-2">
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
                </div>
              </details>
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
          {step < 2 ? (
            <Button onClick={next}>Continue</Button>
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
