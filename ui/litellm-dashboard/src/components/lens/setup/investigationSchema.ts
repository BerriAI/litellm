import { z } from "zod";
import type { Settings } from "../model/types";
import { normalizeFilters } from "./filters";
import { initialWatches, isWatch, watchChecks } from "../model/watches";

const selectionFields = {
  source: z.enum(["traces", "requests", "both"]),
  service: z.string(),
  agent_name: z.string(),
  filters: z.array(z.object({ key: z.string(), value: z.string() })),
  lookback_hours: z.custom<number>(),
  sample_size: z.custom<number | null>(),
  sample_percent: z.custom<number>(),
  team_id: z.string(),
  execution_ids: z.array(z.string()),
};

const draftFields = {
  name: z.string(),
  selection: z.object(selectionFields),
  context: z.string(),
  watching: z.array(z.string()),
  questions: z.array(z.object({ id: z.string(), instruction: z.string(), enabled: z.boolean() })),
  selectedModel: z.string().nullable(),
  budget: z.custom<number>(),
  repeat: z.boolean(),
  interval: z.custom<number>(),
  manualSelection: z.boolean(),
};
const draftSchema = z.object(draftFields);

type InvestigationDraft = z.infer<typeof draftSchema>;

function validateFilters(draft: InvestigationDraft, ctx: z.RefinementCtx) {
  draft.selection.filters.forEach((filter, index) => {
    if (!filter.key.trim()) {
      ctx.addIssue({
        code: "custom",
        message: "Choose a key and value for every condition, or remove it",
        path: ["selection", "filters", index, "key"],
      });
    }
    if (!filter.value.trim()) {
      ctx.addIssue({
        code: "custom",
        message: "Choose a key and value for every condition, or remove it",
        path: ["selection", "filters", index, "value"],
      });
    }
  });
}

function validateManualSelection(draft: InvestigationDraft, ctx: z.RefinementCtx) {
  if (draft.manualSelection && !draft.selection.execution_ids.length) {
    ctx.addIssue({
      code: "custom",
      message: "Choose at least one run or turn off individual selection",
      path: ["selection", "execution_ids"],
    });
  }
}

function validateSampleWindow(draft: InvestigationDraft, ctx: z.RefinementCtx) {
  const selection = draft.selection;
  const hours = selection.lookback_hours ?? 24;
  if (!Number.isInteger(hours) || hours < 1) {
    ctx.addIssue({
      code: "custom",
      message: "Choose a time range of at least 1 hour",
      path: ["selection", "lookback_hours"],
    });
  }
  const percent = selection.sample_percent ?? 100;
  if (!Number.isFinite(percent) || percent <= 0 || percent > 100) {
    ctx.addIssue({
      code: "custom",
      message: "Choose a sampling percentage greater than 0 and up to 100",
      path: ["selection", "sample_percent"],
    });
  }
  if (selection.sample_size != null && (!Number.isInteger(selection.sample_size) || selection.sample_size < 1)) {
    ctx.addIssue({
      code: "custom",
      message: "Choose a positive maximum or leave it blank for no limit",
      path: ["selection", "sample_size"],
    });
  }
}

function validateBudgetAndSchedule(draft: InvestigationDraft, ctx: z.RefinementCtx) {
  if (!Number.isFinite(draft.budget) || draft.budget <= 0) {
    ctx.addIssue({
      code: "custom",
      message: "Choose a monthly limit greater than zero",
      path: ["budget"],
    });
  }
  const intervalOutOfRange = draft.interval < 1;
  const intervalInvalid = !Number.isInteger(draft.interval) || intervalOutOfRange;
  if (draft.repeat && intervalInvalid) {
    ctx.addIssue({
      code: "custom",
      message: "Choose a repeat interval of at least 1 minute",
      path: ["interval"],
    });
  }
}

function validateExpectations(draft: InvestigationDraft, ctx: z.RefinementCtx) {
  const checks = draft.questions.filter((check) => check.instruction.trim());
  if (!draft.context.trim() && !checks.length && !draft.watching.length) {
    ctx.addIssue({
      code: "custom",
      message: "Describe the expected behavior or pick something to watch for",
      path: ["context"],
    });
  }
  draft.questions.forEach((check, index) => {
    if (check.instruction.trim() && check.instruction.trim().length < 3) {
      ctx.addIssue({
        code: "custom",
        message: "Use at least three characters for each check",
        path: ["questions", index, "instruction"],
      });
    }
  });
}

export const investigationSchema = draftSchema
  .superRefine((draft, ctx) => {
    validateFilters(draft, ctx);
    validateManualSelection(draft, ctx);
    validateSampleWindow(draft, ctx);
    validateBudgetAndSchedule(draft, ctx);
    validateExpectations(draft, ctx);
  })
  .transform((draft) => ({
    ...draft,
    context: draft.context.trim(),
    selection: { ...draft.selection, filters: normalizeFilters(draft.selection.filters) },
    questions: draft.questions
      .filter((check) => check.instruction.trim())
      .map((check) => ({ ...check, instruction: check.instruction.trim() })),
  }));

export type InvestigationInput = z.input<typeof investigationSchema>;
export type InvestigationOutput = z.output<typeof investigationSchema>;

export const SETUP_STEPS = ["activity", "criteria", "run"] as const;
export type SetupStep = (typeof SETUP_STEPS)[number];

type SelectionField = `selection.${keyof InvestigationInput["selection"]}`;
export type InvestigationField = Exclude<keyof InvestigationInput, "selection"> | SelectionField;

/** Every form field belongs to exactly one setup step; adding a schema field without a step fails to type-check. */
const stepOfField = {
  name: "activity",
  "selection.source": "activity",
  "selection.service": "activity",
  "selection.agent_name": "activity",
  "selection.filters": "activity",
  "selection.team_id": "activity",
  "selection.lookback_hours": "activity",
  "selection.sample_percent": "activity",
  context: "criteria",
  questions: "criteria",
  watching: "criteria",
  "selection.execution_ids": "run",
  "selection.sample_size": "run",
  selectedModel: "run",
  budget: "run",
  interval: "run",
  repeat: "run",
  manualSelection: "run",
} as const satisfies Record<InvestigationField, SetupStep>;

const fields = Object.keys(stepOfField) as readonly InvestigationField[];

export const investigationStepFields: Readonly<Record<SetupStep, readonly InvestigationField[]>> = {
  activity: fields.filter((field) => stepOfField[field] === "activity"),
  criteria: fields.filter((field) => stepOfField[field] === "criteria"),
  run: fields.filter((field) => stepOfField[field] === "run"),
};

function activitySelectionDefaults(
  initial: Settings | undefined,
  defaultSource: Settings["source"],
): InvestigationInput["selection"] {
  return {
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
}

export function investigationDefaults(
  initial: Settings | undefined,
  mode: "new" | "edit" | "duplicate",
  defaultSource: Settings["source"],
): InvestigationInput {
  return {
    name: initial?.name ?? "",
    selection: activitySelectionDefaults(initial, defaultSource),
    context: initial?.context ?? "",
    watching: [...initialWatches(initial?.checks)],
    questions: (initial?.checks ?? []).filter((check) => !isWatch(check)),
    selectedModel: initial?.model ?? null,
    budget: initial?.monthly_budget ?? 100,
    repeat: mode === "new" || (mode === "edit" && !!initial?.enabled),
    interval: initial?.interval_minutes ?? 15,
    manualSelection: !!initial?.execution_ids?.length,
  };
}

export function investigationSettings(
  draft: InvestigationOutput,
  initial: Settings | undefined,
  model: string,
): Settings {
  const suggestedName = draft.questions[0]?.instruction || draft.context.split("\n")[0] || "Investigation";
  return {
    ...initial,
    ...draft.selection,
    name: draft.name.trim() || suggestedName,
    context: draft.context,
    model,
    monthly_budget: draft.budget,
    enabled: draft.repeat,
    interval_minutes: draft.interval,
    concurrency: initial?.concurrency ?? 8,
    checks: [...watchChecks(new Set(draft.watching)), ...draft.questions],
  };
}
