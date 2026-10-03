import { z } from "zod";
import type { Settings } from "../model/types";
import { normalizeFilters } from "./filters";
import { initialWatches, isWatch, watchChecks } from "./watches";

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

function sampleValidationError(selection: z.infer<typeof draftSchema>["selection"]): string | null {
  const hours = selection.lookback_hours ?? 24;
  if (!Number.isInteger(hours) || hours < 1 || hours > 8760) return "Choose a time range between 1 hour and 365 days";
  const percent = selection.sample_percent ?? 100;
  if (!Number.isFinite(percent) || percent <= 0 || percent > 100)
    return "Choose a sampling percentage greater than 0 and up to 100";
  if (selection.sample_size != null && (!Number.isInteger(selection.sample_size) || selection.sample_size < 1))
    return "Choose a positive maximum or leave it blank for no limit";
  return null;
}

export function investigationSchema(step: number) {
  return draftSchema
    .superRefine((draft, ctx) => {
      const issue = (message: string) => ctx.addIssue({ code: "custom", message });
      const selection = draft.selection;
      if (selection.filters.some((f) => !f.key.trim() || !f.value.trim()))
        return issue("Choose a key and value for every condition, or remove it");
      if (step >= 2 && draft.manualSelection && !selection.execution_ids.length)
        return issue("Choose at least one run or turn off individual selection");
      const sampleError = sampleValidationError(selection);
      if (sampleError) return issue(sampleError);
      const checks = draft.questions.filter((check) => check.instruction.trim());
      const nothingToCheck = !draft.context.trim() && !checks.length && !draft.watching.length;
      if (step >= 1 && nothingToCheck) return issue("Describe the expected behavior or pick something to watch for");
      if (checks.some((check) => check.instruction.trim().length < 3))
        return issue("Use at least three characters for each check");
    })
    .transform((draft) => ({
      ...draft,
      context: draft.context.trim(),
      selection: { ...draft.selection, filters: normalizeFilters(draft.selection.filters) },
      questions: draft.questions
        .filter((check) => check.instruction.trim())
        .map((check) => ({ ...check, instruction: check.instruction.trim() })),
    }));
}

export type InvestigationInput = z.input<ReturnType<typeof investigationSchema>>;
export type InvestigationOutput = z.output<ReturnType<typeof investigationSchema>>;

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
    repeat: mode === "edit" && !!initial?.enabled,
    interval: initial?.interval_minutes ?? 30,
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
    name: draft.name.trim() || suggestedName.slice(0, 100),
    context: draft.context,
    model,
    monthly_budget: draft.budget,
    enabled: draft.repeat,
    interval_minutes: draft.interval,
    concurrency: initial?.concurrency ?? 8,
    checks: [...watchChecks(new Set(draft.watching)), ...draft.questions],
  };
}
