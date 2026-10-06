"use client";

import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { useEffect, useMemo, useState, type ReactNode } from "react";
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
import { DRAFT_FIELDS, draftFromParams, useSetupDraftRoute, useSetupStepRoute } from "./setupRoute";
import { useDebouncedValue } from "./useDebouncedValue";
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
import { durationLabel, scopeLabel } from "../model/format";
import { type Settings } from "../model/types";

type SetupMode = "new" | "edit" | "duplicate";

const DRAFT_URL_DEBOUNCE_MS = 400;

const TITLES: Record<SetupMode, string> = {
  new: "New investigation",
  edit: "Edit investigation",
  duplicate: "Duplicate investigation",
};

function saveLabelFor(mode: SetupMode, repeat: boolean): string {
  if (mode === "edit") return "Save changes";
  return repeat ? "Run and monitor" : "Run investigation";
}

function ActivitySummary() {
  const [selection, manual] = useWatch<InvestigationInput, ["selection", "manualSelection"]>({
    name: ["selection", "manualSelection"],
  });
  return activitySummary(selection, manual);
}

function CriteriaSummary() {
  const [context, watching, questions] = useWatch<InvestigationInput, ["context", "watching", "questions"]>({
    name: ["context", "watching", "questions"],
  });
  return criteriaSummary({ context, watching, questions });
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
  defaultSource?: Settings["source"];
  ready?: boolean;
  onClose: () => void;
  onSave: (settings: Settings) => Promise<void>;
}

/** Replaces the Investigations tab body: a three-step setup on the left, the activity it matches on the right. */
export function InvestigationSetup(props: SetupProps) {
  const { initial, mode, defaultSource = "traces" } = props;
  const draft = useSetupDraftRoute(defaultSource);
  const fromUrl = mode === "new" && !initial;
  const form = useZodForm(investigationSchema, {
    defaultValues: fromUrl
      ? draftFromParams(draft.params, defaultSource)
      : investigationDefaults(initial, mode, defaultSource),
    mode: "onChange",
  });
  return (
    <FormProvider {...form}>
      {fromUrl && <DraftUrlSync defaultSource={defaultSource} />}
      <SetupEditor {...props} form={form} />
    </FormProvider>
  );
}

/** Writes the draft to the URL once typing pauses; only this empty component re-renders per keystroke. */
function DraftUrlSync({ defaultSource }: { defaultSource: Settings["source"] }) {
  const [name, selection, context, watching, questions, repeat, interval] = useWatch<
    InvestigationInput,
    typeof DRAFT_FIELDS
  >({ name: DRAFT_FIELDS });
  const draft = { name, selection, context, watching, questions, repeat, interval };
  const { value } = useDebouncedValue(draft, DRAFT_URL_DEBOUNCE_MS);
  const { saveDraft } = useSetupDraftRoute(defaultSource);
  useEffect(() => saveDraft(value), [value, saveDraft]);
  return null;
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
  const [trace, setTrace] = useState<Execution | null>(null);
  const { control, register, trigger, formState } = form;
  const [selectedModel, repeat] = useWatch({ control, name: ["selectedModel", "repeat"] });
  const activity = useMatchingActivity();
  const { executions } = activity.preview.page;
  const traceRuns = useMemo(() => executions.filter((run) => run.source === "traces"), [executions]);
  const model = selectedModel ?? analysis.defaultModel ?? "";
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
    <section aria-label={TITLES[mode]} className="flex min-h-0 min-w-0 flex-1 flex-col">
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
          data-1p-ignore
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
      <div className="grid min-h-0 min-w-0 flex-1 items-start gap-8 pt-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)] xl:gap-10">
        <SetupSteps
          aria-label="Investigation setup"
          current={step}
          onOpen={setStep}
          className="lg:-mx-2 lg:-my-1 lg:max-h-full lg:overflow-y-auto lg:px-2 lg:py-1"
        >
          <SetupStep
            id="activity"
            heading="Activity"
            description="Which traces or requests to review"
            summary={<ActivitySummary />}
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
            summary={<CriteriaSummary />}
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
          itemKey={(run) => run.id}
          selected={trace}
          onSelectedChange={setTrace}
          noun="run"
          storageKey={FINDING_PANEL_WIDTH_KEY}
        >
          <MatchingActivityPreview {...activity.preview} className="min-w-0 lg:max-h-full" onOpen={setTrace} />
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

function StepFooter({ children }: { children: ReactNode }) {
  return <div className="flex justify-end border-t pt-4">{children}</div>;
}
