"use client";

import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { useEffect, useState, type ReactNode } from "react";
import { ArrowLeft } from "lucide-react";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { FieldError } from "@/components/ui/field";
import {
  investigationSchema,
  investigationDefaults,
  investigationSettings,
  investigationStepFields,
  type InvestigationInput,
  type InvestigationOutput,
} from "./investigationSchema";
import { draftFromParams, useSetupDraftRoute, useSetupStepRoute } from "./setupRoute";
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
import { type TraceRef, traceKey, traceRefOf } from "../traces/routing";
import { durationLabel, scopeLabel } from "../model/format";
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

function activitySummary(selection: InvestigationInput["selection"], manual: boolean): string {
  const span = `Last ${durationLabel(selection.lookback_hours ?? 24, "hours")}`;
  const picked = selection.execution_ids.length;
  if (manual) return `${picked} picked ${picked === 1 ? "run" : "runs"} · ${span}`;
  return `${scopeLabel(selection)} · ${span}`;
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
  ready?: boolean;
  onClose: () => void;
  onSave: (settings: Settings) => Promise<void>;
}

/** Replaces the Investigations tab body: a three-step setup on the left, the activity it matches on the right. */
export function InvestigationSetup(props: SetupProps) {
  const { initial, mode } = props;
  const draft = useSetupDraftRoute();
  const form = useZodForm(investigationSchema, {
    defaultValues: mode === "new" ? draftFromParams(draft.params) : investigationDefaults(initial, mode),
    mode: "onChange",
  });
  const { saveDraft } = draft;
  useEffect(() => {
    if (mode !== "new") return;
    return form.subscribe({ formState: { values: true }, callback: ({ values }) => saveDraft(values) });
  }, [form, mode, saveDraft]);
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
  const [step, setStep] = useSetupStepRoute();
  const [error, setError] = useState("");
  const [trace, setTrace] = useState<TraceRef | null>(null);
  const { control, register, setValue, subscribe, trigger, formState } = form;
  const [selectedModel, repeat, selection, manualSelection, context, watching, questions] = useWatch({
    control,
    name: ["selectedModel", "repeat", "selection", "manualSelection", "context", "watching", "questions"],
  });
  const activity = useMatchingActivity();
  const traceRuns = activity.preview.page.executions.flatMap((run) => (run.summary ? [traceRefOf(run.summary)] : []));
  const model = selectedModel ?? analysis.defaultModel ?? "";
  useEffect(
    () =>
      subscribe({
        name: ["selection.q", "selection.lookback_hours"],
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
    <section aria-label={TITLES[mode]} className="flex min-w-0 flex-1 flex-col">
      <header className="flex items-center gap-1 border-b pb-3">
        <Button
          variant="ghost"
          size="icon-sm"
          className="shrink-0 text-muted-foreground"
          aria-label="Back to investigations"
          disabled={formState.isSubmitting}
          onClick={onClose}
        >
          <ArrowLeft className="size-4" />
        </Button>
        <input
          {...register("name")}
          aria-label="Investigation name"
          placeholder={TITLES[mode]}
          autoComplete="off"
          className="h-8 min-w-0 flex-1 appearance-none rounded-md border-0 bg-transparent px-2 py-0 text-lg font-semibold tracking-tight shadow-none ring-0 outline-none placeholder:text-muted-foreground/60 hover:bg-muted/50 focus:bg-muted/50 focus:ring-0 focus:outline-none"
        />
        <Button variant="ghost" size="sm" disabled={formState.isSubmitting} onClick={onClose}>
          Cancel
        </Button>
      </header>
      {offline && (
        <p
          role="status"
          className="mt-4 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-sm text-warning"
        >
          The worker or trace storage is unavailable. Your draft is safe; you can start when it reconnects.
        </p>
      )}
      <div className="grid min-w-0 flex-1 items-start gap-8 pt-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] xl:gap-10">
        <SetupSteps aria-label="Investigation setup" current={step} onOpen={setStep}>
          <SetupStep
            id="activity"
            heading="Activity"
            description="Which runs to review"
            summary={activitySummary(selection, manualSelection)}
          >
            <ScopeFields {...activity.scope} />
            <SampleFields eligible={activity.preview.page.eligible} />
            <StepFooter>
              <Button onClick={() => void next()}>Continue</Button>
            </StepFooter>
          </SetupStep>
          <SetupStep
            id="criteria"
            heading="Criteria"
            description="What the agent should do and what to watch for"
            summary={criteriaSummary({ context, watching, questions })}
          >
            <ExpectationsFields />
            <StepFooter>
              <Button onClick={() => void next()}>Continue</Button>
            </StepFooter>
          </SetupStep>
          <SetupStep id="run" heading="Schedule" description="How often to run, the model, and a budget" summary="">
            <RunFields models={analysis} gate={gate} />
            <FieldError>{formState.errors.selection?.execution_ids?.message || error}</FieldError>
            <StepFooter>
              <Button disabled={!canSave} onClick={() => void save()}>
                {formState.isSubmitting ? "Saving…" : saveLabel}
              </Button>
            </StepFooter>
          </SetupStep>
        </SetupSteps>
        <Inspector.Root
          items={traceRuns}
          itemKey={traceKey}
          selected={trace}
          onSelectedChange={setTrace}
          noun="run"
          storageKey={FINDING_PANEL_WIDTH_KEY}
        >
          <MatchingActivityPreview {...activity.preview} className="min-w-0 lg:sticky lg:top-0" />
          <Inspector.Panel label="Run details" testId="run-panel">
            {(run: TraceRef) => (
              <TraceEvidence
                traceId={run.traceId}
                traceRef={run.traceRef}
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

function StepFooter({ children }: { children: ReactNode }) {
  return <div className="flex justify-end border-t pt-4">{children}</div>;
}
