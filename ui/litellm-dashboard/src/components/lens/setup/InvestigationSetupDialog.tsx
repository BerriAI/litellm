"use client";

import { FormProvider } from "react-hook-form";
import { useZodForm } from "@/lib/forms/useZodForm";
import { investigationSchema, investigationDefaults, investigationSettings } from "./investigationSchema";
import { ScopeStep } from "./steps/ScopeStep";
import { ExpectationsStep } from "./steps/ExpectationsStep";
import { RunStep } from "./steps/RunStep";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { type ActivitySelection, type Settings } from "../model/types";
import { type AnalysisModelInfo } from "./fields/analysisModels";

export function InvestigationSetupDialog({
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
  const form = useZodForm(investigationSchema(step), {
    defaultValues: investigationDefaults(initial, mode, defaultSource),
  });
  const { selection, selectedModel, budget, repeat, interval, manualSelection } = form.watch();
  const model = selectedModel ?? defaultModel ?? "";
  const setModel = (model: string) => form.setValue("selectedModel", model);
  const setSelection = (next: ActivitySelection) => {
    form.setValue("selection.source", next.source);
    form.setValue("selection.service", next.service ?? "");
    form.setValue("selection.agent_name", next.agent_name ?? "");
    form.setValue("selection.lookback_hours", next.lookback_hours ?? 24);
    form.setValue("selection.sample_size", next.sample_size ?? null);
    form.setValue("selection.sample_percent", next.sample_percent ?? 100);
    form.setValue("selection.team_id", next.team_id ?? "");
    form.setValue("selection.execution_ids", next.execution_ids ?? []);
    if (next.filters !== selection.filters) {
      if (next.filters?.length === selection.filters.length) {
        next.filters.forEach((filter, index) => {
          form.setValue(`selection.filters.${index}.key`, filter.key);
          form.setValue(`selection.filters.${index}.value`, filter.value);
        });
      } else form.setValue("selection.filters", next.filters ?? []);
    }
  };
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
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
    const result = investigationSchema(step).safeParse(form.getValues());
    if (!result.success) throw new Error(result.error.issues[0].message);
    return result.data;
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
      const settings = investigationSettings(validate(), initial, model);
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
    <FormProvider {...form}>
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
              <ScopeStep
                step={step}
                changeSelection={changeSelection}
                accessToken={accessToken}
                setPreviewReady={setPreviewReady}
              />
            )}
            {step === 1 && <ExpectationsStep />}
            {step === 2 && (
              <RunStep
                modelValid={modelValid}
                models={models}
                modelDetails={modelDetails}
                model={model}
                setModel={setModel}
                modelsLoading={modelsLoading}
                modelsError={modelsError}
                unavailable={unavailable}
                unsupported={unsupported}
              />
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
    </FormProvider>
  );
}
