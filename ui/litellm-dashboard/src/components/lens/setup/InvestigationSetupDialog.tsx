"use client";

import { FormProvider, useWatch } from "react-hook-form";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  investigationSchema,
  investigationDefaults,
  investigationSettings,
  investigationStepFields,
} from "./investigationSchema";
import { ScopeStep } from "./steps/ScopeStep";
import { ExpectationsStep } from "./steps/ExpectationsStep";
import { RunStep } from "./steps/RunStep";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { type Settings } from "../model/types";
import { type AnalysisModelInfo } from "./fields/analysisModels";
import { cn } from "@/lib/cva.config";

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
  const [error, setError] = useState("");
  const form = useZodForm(investigationSchema, {
    defaultValues: investigationDefaults(initial, mode, defaultSource),
    mode: "onChange",
  });
  const { control, setValue, subscribe, trigger, formState } = form;
  const [selectedModel, repeat] = useWatch({
    control,
    name: ["selectedModel", "repeat"],
  });
  const model = selectedModel ?? defaultModel ?? "";
  useEffect(
    () =>
      subscribe({
        name: [
          "selection.source",
          "selection.service",
          "selection.agent_name",
          "selection.filters",
          "selection.lookback_hours",
          "selection.team_id",
        ],
        formState: { values: true },
        callback: ({ values }) => {
          if (values.selection.execution_ids.length) setValue("selection.execution_ids", []);
        },
      }),
    [setValue, subscribe],
  );
  const next = async () => {
    if (await trigger(investigationStepFields[step])) setStep(step + 1);
  };
  const save = form.handleSubmit(async (values) => {
    setError("");
    try {
      await onSave(investigationSettings(values, initial, model));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save investigation");
    }
  });
  const unsupported = modelDetails.some((m) => m.model_group === model && m.mode && m.mode !== "chat");
  const canRun = ready || mode === "edit";
  const modelsReady = !modelsLoading && !modelsError;
  const unavailable = !!model && modelsReady && !models.includes(model);
  const supported = !unsupported && !unavailable;
  const preservingSavedModel = mode === "edit" && model === initial?.model;
  const modelReady = modelsReady || preservingSavedModel;
  const modelValid = !!model && supported && modelReady;
  const runReady = canRun && (mode === "edit" || previewReady);
  const formReady = formState.isValid && !formState.isSubmitting;
  const canSave = formReady && modelValid && runReady;
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
          if (!open && !formState.isSubmitting) onClose();
        }}
      >
        <DialogContent
          className={cn(
            "flex max-h-[90dvh] flex-col gap-6 overflow-hidden",
            ["sm:max-w-xl", "sm:max-w-2xl", "sm:max-w-3xl"][step],
          )}
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
                disabled={index > step || formState.isSubmitting}
                aria-current={index === step ? "step" : undefined}
                onClick={() => {
                  setStep(index);
                }}
                data-state={index === step ? "active" : "inactive"}
                className="flex-1 border-t-2 border-border pt-2 text-left text-muted-foreground data-[state=active]:border-foreground data-[state=active]:font-medium data-[state=active]:text-foreground"
              >
                {index + 1}. {label}
              </button>
            ))}
          </nav>
          <div className="min-h-0 overflow-y-auto pr-1 space-y-5">
            {(step === 0 || step === 2) && (
              <ScopeStep step={step} accessToken={accessToken} setPreviewReady={setPreviewReady} />
            )}
            {step === 1 && <ExpectationsStep />}
            {step === 2 && (
              <RunStep
                modelValid={modelValid}
                models={models}
                modelDetails={modelDetails}
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
            <Button
              variant="outline"
              disabled={formState.isSubmitting}
              onClick={() => (step ? setStep(step - 1) : onClose())}
            >
              {step ? "Back" : "Cancel"}
            </Button>
            {step < 2 ? (
              <Button onClick={() => void next()}>Continue</Button>
            ) : (
              <Button disabled={!canSave} onClick={() => void save()}>
                {formState.isSubmitting ? "Saving…" : saveLabel}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </FormProvider>
  );
}
