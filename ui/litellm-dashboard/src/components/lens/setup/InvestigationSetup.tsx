"use client";

import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { useEffect, useState, type ReactNode } from "react";
import { Check, ChevronLeft } from "lucide-react";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cva.config";
import {
  investigationSchema,
  investigationDefaults,
  investigationSettings,
  investigationStepFields,
  SETUP_STEPS,
  type InvestigationInput,
  type InvestigationOutput,
  type SetupStep,
} from "./investigationSchema";
import { ScopeFields } from "./fields/ScopeFields";
import { SampleFields } from "./fields/SampleFields";
import { ExpectationsFields } from "./fields/ExpectationsFields";
import { RunFields } from "./fields/RunFields";
import { MatchingActivityPreview } from "./MatchingActivityPreview";
import { useMatchingActivity } from "./useMatchingActivity";
import { useAnalysisModels } from "./fields/useAnalysisModels";
import { TraceSheet } from "../investigations/TraceSheet";
import { durationLabel } from "../model/format";
import { type Settings } from "../model/types";
import { type AnalysisModelInfo } from "../model/types";

type SetupMode = "new" | "edit" | "duplicate";
type StepState = "done" | "current" | "upcoming";

const TITLES: Record<SetupMode, string> = {
  new: "New investigation",
  edit: "Edit investigation",
  duplicate: "Duplicate investigation",
};

const STEPS: Readonly<Record<SetupStep, { title: string; description: string }>> = {
  activity: { title: "Activity", description: "Which traces or requests to review" },
  criteria: { title: "Criteria", description: "What the agent should do and what to watch for" },
  run: { title: "Run", description: "Schedule, analysis model, and budget" },
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

const STEP_BADGE: Record<StepState, string> = {
  done: "border-foreground text-foreground",
  current: "border-foreground bg-foreground text-background",
  upcoming: "border-border text-muted-foreground",
};

/** One row of the vertical stepper: finished steps collapse to a summary and reopen on click. */
function SetupStepRow({
  step,
  index,
  state,
  summary,
  onOpen,
  children,
}: {
  step: SetupStep;
  index: number;
  state: StepState;
  summary: string;
  onOpen: () => void;
  children: ReactNode;
}) {
  const { title, description } = STEPS[step];
  const last = index === SETUP_STEPS.length - 1;
  return (
    <li className={cn("relative flex gap-4", !last && "pb-8")}>
      {!last && <span aria-hidden="true" className="absolute top-7 bottom-0 left-3.5 w-px bg-border" />}
      <span
        aria-hidden="true"
        className={cn(
          "z-raised flex size-7 shrink-0 items-center justify-center rounded-full border bg-card text-xs font-medium",
          STEP_BADGE[state],
        )}
      >
        {state === "done" ? <Check className="size-3.5" /> : index + 1}
      </span>
      <div className="min-w-0 flex-1 pt-0.5">
        <button
          type="button"
          disabled={state !== "done"}
          aria-current={state === "current" ? "step" : undefined}
          onClick={onOpen}
          className="flex w-full flex-col items-start gap-0.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-default"
        >
          <span className={cn("text-sm font-semibold", state === "upcoming" && "text-muted-foreground")}>{title}</span>
          <span className="text-xs text-muted-foreground">{state === "done" ? summary : description}</span>
        </button>
        {state === "current" && <div className="mt-4 space-y-5">{children}</div>}
      </div>
    </li>
  );
}

interface SetupProps {
  initial?: Settings;
  mode?: SetupMode;
  defaultSource?: Settings["source"];
  accessToken: string;
  ready?: boolean;
  onClose: () => void;
  onSave: (settings: Settings) => Promise<void>;
}

/** Replaces the Investigations tab body: a three-step setup on the left, the activity it matches on the right. */
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
  accessToken,
  ready = true,
  onClose,
  onSave,
  form,
}: SetupProps & { mode: SetupMode; form: UseFormReturn<InvestigationInput, unknown, InvestigationOutput> }) {
  const { models, modelDetails, modelsLoading, modelsError, defaultModel } = useAnalysisModels();
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");
  const [trace, setTrace] = useState<{ id: string; ref?: string } | null>(null);
  const { control, register, setValue, subscribe, trigger, formState } = form;
  const [selectedModel, repeat, selection, context, watching, questions] = useWatch({
    control,
    name: ["selectedModel", "repeat", "selection", "context", "watching", "questions"],
  });
  const activity = useMatchingActivity();
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
    if (await trigger(investigationStepFields[SETUP_STEPS[step]])) setStep(step + 1);
  };
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
  const summaries = [activitySummary(selection), criteriaSummary({ context, watching, questions }), ""];
  const stateOf = (index: number): StepState => {
    if (index < step) return "done";
    return index === step ? "current" : "upcoming";
  };
  const stepContent = (key: SetupStep) => {
    if (key === "activity")
      return (
        <>
          <label className="grid gap-2 text-sm font-medium">
            Investigation name
            <Input {...register("name")} placeholder="e.g. Support quality" />
          </label>
          <ScopeFields {...activity.scope} />
          <SampleFields />
        </>
      );
    if (key === "criteria") return <ExpectationsFields />;
    return (
      <RunFields
        modelValid={gate.modelValid}
        models={models}
        modelDetails={modelDetails}
        modelsLoading={modelsLoading}
        modelsError={modelsError}
        unavailable={gate.unavailable}
        unsupported={gate.unsupported}
      />
    );
  };
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
        <p role="status" className="text-sm text-amber-700">
          The worker or trace storage is unavailable. Your draft is safe; you can start when it reconnects.
        </p>
      )}
      <div className="grid min-w-0 gap-8 lg:grid-cols-2">
        <ol aria-label="Investigation setup" className="min-w-0">
          {SETUP_STEPS.map((key, index) => (
            <SetupStepRow
              key={key}
              step={key}
              index={index}
              state={stateOf(index)}
              summary={summaries[index]}
              onOpen={() => setStep(index)}
            >
              {stepContent(key)}
              {error && key === "run" && (
                <p role="alert" className="text-sm text-destructive">
                  {error}
                </p>
              )}
              <div className="flex justify-end pt-1">
                {key !== "run" ? (
                  <Button onClick={() => void next()}>Continue</Button>
                ) : (
                  <Button disabled={!canSave} onClick={() => void save()}>
                    {formState.isSubmitting ? "Saving…" : saveLabel}
                  </Button>
                )}
              </div>
            </SetupStepRow>
          ))}
        </ol>
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
