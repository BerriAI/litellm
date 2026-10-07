"use client";

import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { useEffect, useState } from "react";
import { ChevronLeft } from "lucide-react";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  investigationSchema,
  investigationDefaults,
  investigationSettings,
  investigationStepFields,
  type InvestigationInput,
  type InvestigationOutput,
  type SetupStep as SetupStepId,
} from "./investigationSchema";
import { nextSetupStep, SetupStep, SetupSteps } from "./SetupSteps";
import { ScopeFields } from "./fields/ScopeFields";
import { SampleFields } from "./fields/SampleFields";
import { ExpectationsFields } from "./fields/ExpectationsFields";
import { RunFields } from "./fields/RunFields";
import { MatchingActivityPreview } from "./MatchingActivityPreview";
import { useMatchingActivity } from "./useMatchingActivity";
import { useAnalysisModels } from "./fields/useAnalysisModels";
import { modelGate } from "./fields/analysisModels";
import { Inspector } from "@/components/shared/Inspector";
import { TraceEvidence } from "../investigations/Evidence";
import { FINDING_PANEL_WIDTH_KEY } from "../storage";
import type { Execution } from "./useMatchingActivity";
import { durationLabel } from "../model/format";
import { type Settings } from "../model/types";

type SetupMode = "new" | "edit" | "duplicate";

const TITLES: Record<SetupMode, string> = {
  new: "New investigation",
  edit: "Edit investigation",
  duplicate: "Duplicate investigation",
};

function saveLabelFor(mode: SetupMode, repeat: boolean): string {
  if (mode === "edit") return "Save changes";
  return repeat ? "Run and monitor" : "Run investigation";
}

function activitySummary(selection: InvestigationInput["selection"]): string {
  const who = selection.agent_name || selection.service || "All activity";
  const conditions = selection.filters.length ? ` · ${selection.filters.length} conditions` : "";
  return `${who} · Last ${durationLabel(selection.lookback_hours ?? 24, "hours")}${conditions}`;
}

function criteriaSummary(values: Pick<InvestigationInput, "context" | "watching" | "questions">): string {
  const checks = values.watching.length + values.questions.filter((q) => q.instruction.trim()).length;
  const checksLabel = `${checks} ${checks === 1 ? "check" : "checks"}`;
  const headline = values.context.split("\n")[0]?.trim();
  return headline ? `${headline} · ${checksLabel}` : checksLabel;
}

interface SetupProps {
  initial?: Settings;
  mode: SetupMode;
  defaultSource?: Settings["source"];
  ready?: boolean;
  onClose: () => void;
  onSave: (settings: Settings) => Promise<void>;
}

/** Replaces the Investigations tab body: a three-step setup on the left, the activity it matches on the right. */
export function InvestigationSetup(props: SetupProps) {
  const { initial, mode, defaultSource = "traces" } = props;
  const form = useZodForm(investigationSchema, {
    defaultValues: investigationDefaults(initial, mode, defaultSource),
    mode: "onChange",
  });
  return (
    <FormProvider {...form}>
      <SetupEditor {...props} form={form} />
    </FormProvider>
  );
}

function SetupEditor({
  initial,
  mode,
  ready = true,
  onClose,
  onSave,
  form,
}: SetupProps & { form: UseFormReturn<InvestigationInput, unknown, InvestigationOutput> }) {
  const analysis = useAnalysisModels();
  const [step, setStep] = useState<SetupStepId>("activity");
  const [error, setError] = useState("");
  const [trace, setTrace] = useState<Execution | null>(null);
  const { control, register, setValue, subscribe, trigger, formState } = form;
  const [selectedModel, repeat, selection, context, watching, questions] = useWatch({
    control,
    name: ["selectedModel", "repeat", "selection", "context", "watching", "questions"],
  });
  const activity = useMatchingActivity();
  const traceRuns = activity.preview.page.executions.filter((run) => run.source === "traces");
  const model = selectedModel ?? analysis.defaultModel ?? "";
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
    const following = nextSetupStep(step);
    if (following && (await trigger(investigationStepFields[step]))) setStep(following);
  };
  const save = form.handleSubmit(async (values) => {
    setError("");
    try {
      await onSave(investigationSettings(values, initial, model));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save investigation");
    }
  });
  const gate = modelGate(analysis, model, mode === "edit" && model === initial?.model);
  const runReady = mode === "edit" || (ready && activity.canReview);
  const formReady = formState.isValid && !formState.isSubmitting;
  const canSave = formReady && gate.modelValid && runReady;
  const saveLabel = saveLabelFor(mode, repeat);
  const offline = !ready && mode !== "edit";
  return (
    <section aria-label={TITLES[mode]} className="flex min-w-0 flex-1 flex-col gap-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-start gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            className="-ml-2 shrink-0 text-muted-foreground"
            aria-label="Back to investigations"
            disabled={formState.isSubmitting}
            onClick={onClose}
          >
            <ChevronLeft className="size-5" />
          </Button>
          <div>
            <h2 className="text-lg font-semibold tracking-tight">{TITLES[mode]}</h2>
            <p className="text-xs text-muted-foreground">
              Matching activity on the right updates as you change the setup.
            </p>
          </div>
        </div>
        <Button variant="outline" disabled={formState.isSubmitting} onClick={onClose}>
          Cancel
        </Button>
      </header>
      {offline && (
        <p role="status" className="text-sm text-warning">
          The worker or trace storage is unavailable. Your draft is safe; you can start when it reconnects.
        </p>
      )}
      <div className="grid min-w-0 gap-8 lg:grid-cols-2">
        <SetupSteps aria-label="Investigation setup" current={step} onOpen={setStep}>
          <SetupStep
            id="activity"
            heading="Activity"
            description="Which traces or requests to review"
            summary={activitySummary(selection)}
          >
            <label className="grid gap-2 text-sm font-medium">
              Investigation name
              <Input {...register("name")} placeholder="e.g. Support quality" />
            </label>
            <ScopeFields {...activity.scope} />
            <SampleFields />
            <div className="flex justify-end pt-1">
              <Button onClick={() => void next()}>Continue</Button>
            </div>
          </SetupStep>
          <SetupStep
            id="criteria"
            heading="Criteria"
            description="What the agent should do and what to watch for"
            summary={criteriaSummary({ context, watching, questions })}
          >
            <ExpectationsFields />
            <div className="flex justify-end pt-1">
              <Button onClick={() => void next()}>Continue</Button>
            </div>
          </SetupStep>
          <SetupStep id="run" heading="Run" description="Schedule, analysis model, and budget" summary="">
            <RunFields models={analysis} gate={gate} />
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            <div className="flex justify-end pt-1">
              <Button disabled={!canSave} onClick={() => void save()}>
                {formState.isSubmitting ? "Saving…" : saveLabel}
              </Button>
            </div>
          </SetupStep>
        </SetupSteps>
        <Inspector.Root
          items={traceRuns}
          itemKey={(run) => run.id}
          selected={trace}
          onSelectedChange={setTrace}
          noun="run"
          storageKey={FINDING_PANEL_WIDTH_KEY}
        >
          <MatchingActivityPreview {...activity.preview} className="min-w-0 lg:sticky lg:top-0" onOpen={setTrace} />
          <Inspector.Panel label="Run details" testId="run-panel">
            {(run: Execution) => (
              <TraceEvidence
                traceId={run.trace_id}
                traceRef={run.trace_ref}
                initialSpanId={null}
                onBack={() => setTrace(null)}
              />
            )}
          </Inspector.Panel>
        </Inspector.Root>
      </div>
    </section>
  );
}
