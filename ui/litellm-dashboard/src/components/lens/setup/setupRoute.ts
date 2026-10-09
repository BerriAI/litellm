"use client";

import {
  parseAsArrayOf,
  parseAsBoolean,
  parseAsFloat,
  parseAsInteger,
  parseAsString,
  parseAsStringLiteral,
  useQueryStates,
  type inferParserType,
  type Nullable,
} from "nuqs";
import { useCallback } from "react";
import { RUN_FILTER_PARSERS } from "../traces/routing";
import { watches } from "../model/watches";
import { investigationDefaults, SETUP_STEPS, type InvestigationInput, type SetupStep } from "./investigationSchema";

type Source = InvestigationInput["selection"]["source"];
type Filter = InvestigationInput["selection"]["filters"][number];

const NEW_DRAFT = investigationDefaults(undefined, "new", "traces");
const SOURCES = ["traces", "requests", "both"] as const satisfies readonly Source[];
const knownWatch = new Set(watches.map((watch) => watch.id));

/** A new investigation's draft lives in the URL, so the Traces agent filter can open setup already filled in. */
export const SETUP_DRAFT_PARSERS = {
  agent: RUN_FILTER_PARSERS.agent,
  source: parseAsStringLiteral(SOURCES),
  service: parseAsString.withDefault(NEW_DRAFT.selection.service),
  filter: parseAsArrayOf(parseAsString).withDefault([]),
  team: parseAsString.withDefault(NEW_DRAFT.selection.team_id),
  name: parseAsString.withDefault(NEW_DRAFT.name),
  lookback: parseAsInteger.withDefault(NEW_DRAFT.selection.lookback_hours),
  sample: parseAsFloat.withDefault(NEW_DRAFT.selection.sample_percent),
  max: parseAsInteger,
  context: parseAsString.withDefault(NEW_DRAFT.context),
  watch: parseAsArrayOf(parseAsString).withDefault(NEW_DRAFT.watching),
  checks: parseAsArrayOf(parseAsString).withDefault([]),
  monitor: parseAsBoolean.withDefault(NEW_DRAFT.repeat),
  every: parseAsInteger.withDefault(NEW_DRAFT.interval),
  model: parseAsString,
  budget: parseAsFloat.withDefault(NEW_DRAFT.budget),
};

export const SETUP_STEP_PARSERS = { step: parseAsStringLiteral(SETUP_STEPS) };

/** Every setup key but the shared agent filter, which belongs to the Traces tab as much as to the draft. */
export const SETUP_KEYS = [...Object.keys(SETUP_DRAFT_PARSERS), ...Object.keys(SETUP_STEP_PARSERS)].filter(
  (key) => key !== "agent",
);

export type SetupDraftParams = inferParserType<typeof SETUP_DRAFT_PARSERS>;
type SetupDraftUpdate = Nullable<SetupDraftParams>;

export const DRAFT_FIELDS = [
  "name",
  "selection",
  "context",
  "watching",
  "questions",
  "repeat",
  "interval",
  "selectedModel",
  "budget",
] as const;
export type DraftFields = Pick<InvestigationInput, (typeof DRAFT_FIELDS)[number]>;

const filterParam = ({ key, value }: Filter) => `${key}=${value}`;

function filterFromParam(param: string): Filter {
  const split = param.indexOf("=");
  return split < 0 ? { key: param, value: "" } : { key: param.slice(0, split), value: param.slice(split + 1) };
}

export function draftFromParams(params: SetupDraftParams, defaultSource: Source = "traces"): InvestigationInput {
  return {
    ...NEW_DRAFT,
    name: params.name,
    selection: {
      ...NEW_DRAFT.selection,
      source: params.source ?? defaultSource,
      service: params.service,
      agent_name: params.agent,
      filters: params.filter.map(filterFromParam),
      team_id: params.team,
      lookback_hours: params.lookback,
      sample_percent: params.sample,
      sample_size: params.max ?? NEW_DRAFT.selection.sample_size,
    },
    context: params.context,
    watching: params.watch.filter((id) => knownWatch.has(id)),
    questions: params.checks.map((instruction) => ({ id: crypto.randomUUID(), instruction, enabled: true })),
    repeat: params.monitor,
    interval: params.every,
    selectedModel: params.model,
    budget: params.budget,
  };
}

const finite = (value: number | null) => (value != null && Number.isFinite(value) ? value : null);

/** nuqs drops values equal to their defaults from the URL, so an untouched setup keeps a short link. */
export function paramsFromDraft(draft: DraftFields, defaultSource: Source = "traces"): SetupDraftUpdate {
  const { selection } = draft;
  return {
    agent: selection.agent_name,
    source: selection.source === defaultSource ? null : selection.source,
    service: selection.service,
    filter: selection.filters.map(filterParam),
    team: selection.team_id,
    name: draft.name,
    lookback: finite(selection.lookback_hours),
    sample: finite(selection.sample_percent),
    max: finite(selection.sample_size),
    context: draft.context,
    watch: draft.watching,
    checks: draft.questions.map((check) => check.instruction).filter((instruction) => instruction.trim()),
    monitor: draft.repeat,
    every: finite(draft.interval),
    model: draft.selectedModel,
    budget: finite(draft.budget),
  };
}

export function useSetupDraftRoute(defaultSource: Source) {
  const [params, setParams] = useQueryStates(SETUP_DRAFT_PARSERS, { history: "replace" });
  const saveDraft = useCallback(
    (draft: DraftFields) => void setParams(paramsFromDraft(draft, defaultSource)),
    [setParams, defaultSource],
  );
  return { params, saveDraft };
}

export function useSetupStepRoute(): [SetupStep, (step: SetupStep) => void] {
  const [{ step }, setParams] = useQueryStates(SETUP_STEP_PARSERS, { history: "replace" });
  const setStep = useCallback(
    (next: SetupStep) => void setParams({ step: next === SETUP_STEPS[0] ? null : next }),
    [setParams],
  );
  return [step ?? SETUP_STEPS[0], setStep];
}
