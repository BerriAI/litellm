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
import { ActivityScope, type ActivitySelection } from "./ActivityScope";
import {
  analysisModelOptions,
  durationLabel,
  normalizeFilters,
  starterQuestions,
  type AnalysisModelInfo,
  type Settings,
} from "./engineData";

import { SearchSelect } from "@/components/shared/SearchSelect";
import { DurationInput } from "./DurationInput";

export function EngineSetup({
  initial,
  models,
  modelDetails = [],
  modelsLoading = false,
  modelsError,
  accessToken,
  onClose,
  onSave,
}: {
  initial?: Settings;
  models: string[];
  modelDetails?: AnalysisModelInfo[];
  modelsLoading?: boolean;
  modelsError?: string;
  accessToken: string;
  onClose: () => void;
  onSave: (settings: Settings) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState(initial?.name ?? "");
  const [source, setSource] = useState<Settings["source"]>(initial?.source ?? "traces");
  const [lookback, setLookback] = useState(initial?.lookback_hours ?? 24);
  const [service, setService] = useState(initial?.service ?? "");
  const [filters, setFilters] = useState<NonNullable<Settings["filters"]>>(initial?.filters ?? []);
  const [context, setContext] = useState(initial?.context ?? "");
  const [questions, setQuestions] = useState(
    initial?.checks.map((c) => c.instruction).join("\n") ?? starterQuestions.join("\n"),
  );
  const [model, setModel] = useState(initial?.model ?? "");
  const [enabled, setEnabled] = useState(initial?.enabled ?? false);
  const [budget, setBudget] = useState(initial?.monthly_budget ?? 20);
  const [sampleSize, setSampleSize] = useState(initial?.sample_size ?? 100);
  const [interval, setInterval] = useState(initial?.interval_minutes ?? 15);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const settings = (): Settings => ({
    name: name.trim(),
    source,
    lookback_hours: lookback,
    service: service.trim(),
    context,
    filters: normalizeFilters(filters),
    model,
    enabled,
    monthly_budget: budget,
    sample_size: sampleSize,
    interval_minutes: interval,
    checks: questions
      .split("\n")
      .filter((q) => q.trim())
      .map((instruction) => {
        const previous = initial?.checks.find((c) => c.instruction === instruction.trim());
        return previous ?? { id: crypto.randomUUID(), instruction: instruction.trim(), enabled: true };
      }),
  });
  const execute = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };
  const next = () => {
    try {
      normalizeFilters(filters);
      if (!Number.isInteger(lookback) || lookback < 1 || lookback > 720)
        throw new Error("Choose a history window between 1 and 720 hours");
      if (!name.trim()) throw new Error("Give this lens a name");
      if (step === 1 && !questions.trim()) throw new Error("Add at least one question");
      setError("");
      setStep(step + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Check your settings");
    }
  };

  const changeSelection = (selection: ActivitySelection) => {
    setSource(selection.source);
    setLookback(selection.lookback_hours ?? 24);
    setService(selection.service ?? "");
    setFilters(selection.filters ?? []);
  };
  const saveLabel = () => {
    if (busy) return "Saving…";
    if (initial) return "Save changes";
    return enabled ? "Start monitoring" : "Run analysis";
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-3xl max-h-[90vh] flex flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle>{initial ? "Edit lens" : "Set up a lens"}</DialogTitle>
          <DialogDescription>
            {
              [
                "Choose the activity you want to understand",
                "Tell Lens what matters to you",
                "Review your selection and start analysis",
              ][step]
            }
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-2" aria-label={`Step ${step + 1} of 3`}>
          {["Activity", "Questions", "Review & run"].map((label, i) => (
            <div
              key={label}
              className={`flex-1 border-t-2 pt-2 text-xs ${i <= step ? "border-foreground text-foreground" : "border-border text-muted-foreground"}`}
            >
              {i + 1}. {label}
            </div>
          ))}
        </div>
        <div className="min-h-0 overflow-y-auto space-y-4 pr-1">
          {step === 0 && (
            <>
              <label className="grid gap-2 text-sm">
                Name
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Research quality"
                  maxLength={100}
                />
              </label>
              <ActivityScope
                accessToken={accessToken}
                value={{ source, service, filters, lookback_hours: lookback }}
                onChange={changeSelection}
              />
            </>
          )}
          {step === 1 && (
            <>
              <label className="grid gap-2 text-sm">
                What does a good run look like?
                <Textarea
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  rows={3}
                  placeholder="Our swarm researches a question and produces a cited report that incorporates the fact-checker's corrections."
                />
              </label>
              <label className="grid gap-2 text-sm">
                Questions & checks
                <Textarea value={questions} onChange={(e) => setQuestions(e.target.value)} rows={7} />
              </label>
              <p className="text-xs text-muted-foreground">
                One instruction per line. Ask about usage patterns, successful behavior, or a specific problem. Findings
                include evidence from your runs.
              </p>
            </>
          )}
          {step === 2 && (
            <>
              <div className="rounded-lg border p-4 text-sm space-y-2">
                <p className="font-medium">{name}</p>
                <p>
                  {source === "requests" ? "LLM requests" : "Agent runs"} · {service || "All activity"} ·{" "}
                  {`Last ${durationLabel(lookback, "hours")}`}
                </p>
                {filters.map((f) => (
                  <p key={f.key} className="text-muted-foreground">
                    {f.key} is {f.value}
                  </p>
                ))}
                <p className="text-muted-foreground">
                  Up to {sampleSize} matching runs · {questions.split("\n").filter((q) => q.trim()).length} questions
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-sm">Analysis model</p>
                <SearchSelect
                  aria-label="Analysis model"
                  options={analysisModelOptions(models, modelDetails)}
                  value={model}
                  onValueChange={(value) => setModel(value ?? "")}
                  placeholder={modelsLoading ? "Loading models…" : "Search models or providers"}
                  disabled={modelsLoading}
                  emptyText="No matching models configured on this gateway"
                />
                {modelsError && (
                  <p role="alert" className="text-sm text-destructive">
                    Could not load models: {modelsError}
                  </p>
                )}
                {modelDetails.some((item) => item.model_group === model && item.mode && item.mode !== "chat") && (
                  <p role="alert" className="text-sm text-destructive">
                    Choose a chat model that supports JSON output.
                  </p>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Trace content is sent to this model through LiteLLM. Choose a model approved for your data.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <label className="grid gap-2 text-sm">
                  Monthly limit (USD)
                  <Input
                    type="number"
                    min="0.01"
                    step="1"
                    value={budget}
                    onChange={(e) => setBudget(Number(e.target.value))}
                  />
                </label>
                <label className="grid gap-2 text-sm">
                  Runs per scan
                  <Input
                    type="number"
                    min="1"
                    max="500"
                    value={sampleSize}
                    onChange={(e) => setSampleSize(Number(e.target.value))}
                  />
                </label>
              </div>
              <fieldset className="space-y-3">
                <legend className="mb-2 text-sm font-medium">When to run</legend>
                <label className="flex items-center gap-2 text-sm">
                  <input type="radio" name="lens-schedule" checked={!enabled} onChange={() => setEnabled(false)} />
                  Run once, then manually
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input type="radio" name="lens-schedule" checked={enabled} onChange={() => setEnabled(true)} />
                  Run now and keep monitoring
                </label>
                {enabled && (
                  <>
                    <DurationInput
                      label="Check every"
                      value={interval}
                      onChange={setInterval}
                      base="minutes"
                      max={10080}
                    />
                    <p className="text-xs text-muted-foreground">
                      From 1 minute to 7 days. Scans never overlap; the next interval starts after a scan finishes.
                    </p>
                  </>
                )}
              </fieldset>
              <div className="rounded-lg bg-muted/40 p-3 text-sm text-muted-foreground">
                {initial
                  ? "Changes apply to future scans. You can recheck recent runs from the lens page."
                  : "The first scan reviews your selected time window. New activity becomes eligible after two minutes. You can leave this page while it runs."}{" "}
                Larger workloads are sampled; coverage is shown with every scan.
              </div>
            </>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => (step ? setStep(step - 1) : onClose())}>
            {step ? "Back" : "Cancel"}
          </Button>
          {step < 2 ? (
            <Button onClick={next}>Continue</Button>
          ) : (
            <Button
              disabled={
                busy ||
                !model ||
                budget <= 0 ||
                (enabled && (!Number.isInteger(interval) || interval < 1 || interval > 10080)) ||
                modelDetails.some((item) => item.model_group === model && item.mode && item.mode !== "chat")
              }
              onClick={() => execute(() => onSave(settings()))}
            >
              {saveLabel()}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
