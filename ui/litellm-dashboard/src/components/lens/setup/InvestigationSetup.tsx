"use client";

import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { useEffect, useState } from "react";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  investigationSchema,
  investigationDefaults,
  investigationSettings,
  type InvestigationInput,
  type InvestigationOutput,
} from "./investigationSchema";
import { ScopeFields } from "./fields/ScopeFields";
import { SampleFields } from "./fields/SampleFields";
import { ExpectationsFields } from "./fields/ExpectationsFields";
import { RunFields } from "./fields/RunFields";
import { MatchingActivityPreview } from "./MatchingActivityPreview";
import { useMatchingActivity } from "./useMatchingActivity";
import { TraceSheet } from "../investigations/TraceSheet";
import { type Settings } from "../model/types";
import { type AnalysisModelInfo } from "./fields/analysisModels";

type SetupMode = "new" | "edit" | "duplicate";

const TITLES: Record<SetupMode, string> = {
  new: "New investigation",
  edit: "Edit investigation",
  duplicate: "Duplicate investigation",
};

interface ModelGate {
  readonly modelValid: boolean;
  readonly unavailable: boolean;
  readonly unsupported: boolean;
}

/** An edit may keep its saved model while the model list is loading or failing; a new run needs a verified one. */
function modelGate(input: {
  model: string;
  models: string[];
  modelDetails: AnalysisModelInfo[];
  modelsLoading: boolean;
  modelsError?: string;
  savedModel?: string;
  mode: SetupMode;
}): ModelGate {
  const { model, models, modelDetails, modelsLoading, modelsError, savedModel, mode } = input;
  const unsupported = modelDetails.some((m) => m.model_group === model && m.mode && m.mode !== "chat");
  const modelsReady = !modelsLoading && !modelsError;
  const unavailable = !!model && modelsReady && !models.includes(model);
  const preservingSavedModel = mode === "edit" && model === savedModel;
  const supported = !unsupported && !unavailable;
  const modelReady = modelsReady || preservingSavedModel;
  return { modelValid: !!model && supported && modelReady, unavailable, unsupported };
}

function saveLabelFor(mode: SetupMode, repeat: boolean): string {
  if (mode === "edit") return "Save changes";
  return repeat ? "Run and monitor" : "Run investigation";
}

interface SetupProps {
  initial?: Settings;
  mode?: SetupMode;
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
}

/** Replaces the Investigations tab body: matching activity on the left reacts live to the settings on the right. */
export function InvestigationSetup(props: SetupProps) {
  const { initial, mode = initial ? "edit" : "new", defaultSource = "traces" } = props;
  const form = useZodForm(investigationSchema, {
    defaultValues: investigationDefaults(initial, mode, defaultSource),
    mode: "onChange",
  });
  return (
    <FormProvider {...form}>
      <SetupEditor {...props} mode={mode} form={form} />
    </FormProvider>
  );
}

function SetupEditor({
  initial,
  mode,
  models,
  modelDetails = [],
  modelsLoading = false,
  modelsError,
  defaultModel,
  accessToken,
  ready = true,
  onClose,
  onSave,
  form,
}: SetupProps & { mode: SetupMode; form: UseFormReturn<InvestigationInput, unknown, InvestigationOutput> }) {
  const [error, setError] = useState("");
  const [trace, setTrace] = useState<{ id: string; ref?: string } | null>(null);
  const { control, register, setValue, subscribe, formState } = form;
  const [selectedModel, repeat] = useWatch({ control, name: ["selectedModel", "repeat"] });
  const activity = useMatchingActivity(accessToken);
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
  const save = form.handleSubmit(async (values) => {
    setError("");
    try {
      await onSave(investigationSettings(values, initial, model));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save investigation");
    }
  });
  const gateInput = { model, models, modelDetails, modelsLoading, modelsError, savedModel: initial?.model, mode };
  const gate = modelGate(gateInput);
  const runReady = mode === "edit" || (ready && activity.canReview);
  const formReady = formState.isValid && !formState.isSubmitting;
  const canSave = formReady && gate.modelValid && runReady;
  const saveLabel = saveLabelFor(mode, repeat);
  const offline = !ready && mode !== "edit";
  return (
    <section aria-label={TITLES[mode]} className="flex min-w-0 flex-1 flex-col gap-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{TITLES[mode]}</h2>
          <p className="text-xs text-muted-foreground">
            Matching activity on the left updates as you change the settings on the right.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" disabled={formState.isSubmitting} onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={!canSave} onClick={() => void save()}>
            {formState.isSubmitting ? "Saving…" : saveLabel}
          </Button>
        </div>
      </header>
      {offline && (
        <p role="status" className="text-sm text-amber-700">
          The worker or trace storage is unavailable. Your draft is safe; you can start when it reconnects.
        </p>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <div className="grid min-w-0 gap-8 lg:grid-cols-2">
        <div className="min-w-0 space-y-3 lg:sticky lg:top-0 lg:self-start">
          <MatchingActivityPreview
            {...activity.preview}
            onOpen={(run) => setTrace({ id: run.trace_id, ref: run.trace_ref })}
          />
          {activity.manualSelection && activity.selectedRunCount > 0 && (
            <Button variant="outline" size="sm" onClick={activity.clearSelection}>
              Clear {activity.selectedRunCount} selected runs
            </Button>
          )}
        </div>
        <div className="min-w-0 space-y-8">
          <section aria-label="Activity" className="space-y-5">
            <h3 className="text-sm font-semibold">Activity</h3>
            <ScopeFields
              {...activity.scope}
              nameField={
                <label className="grid gap-2 text-sm font-medium">
                  Investigation name
                  <Input {...register("name")} placeholder="e.g. Support quality" />
                </label>
              }
            />
            <SampleFields />
          </section>
          <section aria-label="Expectations" className="space-y-5">
            <h3 className="text-sm font-semibold">Expectations</h3>
            <ExpectationsFields />
          </section>
          <section aria-label="Run" className="space-y-5">
            <h3 className="text-sm font-semibold">Run</h3>
            <RunFields
              modelValid={gate.modelValid}
              models={models}
              modelDetails={modelDetails}
              modelsLoading={modelsLoading}
              modelsError={modelsError}
              unavailable={gate.unavailable}
              unsupported={gate.unsupported}
            />
          </section>
        </div>
      </div>
      {trace && (
        <TraceSheet
          open
          traceId={trace.id}
          traceRef={trace.ref}
          accessToken={accessToken}
          onClose={() => setTrace(null)}
        />
      )}
    </section>
  );
}
