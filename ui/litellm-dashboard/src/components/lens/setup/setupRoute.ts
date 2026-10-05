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

const NEW_DRAFT = investigationDefaults(undefined, "new");
const knownWatch = new Set(watches.map((watch) => watch.id));

/** A new investigation's draft lives in the URL, so a search on the Traces tab can open setup already filled in. */
export const SETUP_DRAFT_PARSERS = {
  q: RUN_FILTER_PARSERS.q,
  name: parseAsString.withDefault(NEW_DRAFT.name),
  lookback: parseAsInteger.withDefault(NEW_DRAFT.selection.lookback_hours),
  sample: parseAsFloat.withDefault(NEW_DRAFT.selection.sample_percent),
  max: parseAsInteger,
  context: parseAsString.withDefault(NEW_DRAFT.context),
  watch: parseAsArrayOf(parseAsString).withDefault(NEW_DRAFT.watching),
  checks: parseAsArrayOf(parseAsString).withDefault([]),
  monitor: parseAsBoolean.withDefault(NEW_DRAFT.repeat),
  every: parseAsInteger.withDefault(NEW_DRAFT.interval),
};

export const SETUP_STEP_PARSERS = { step: parseAsStringLiteral(SETUP_STEPS) };

/** Every setup key but the shared run search, which belongs to the Traces tab as much as to the draft. */
export const SETUP_KEYS = [...Object.keys(SETUP_DRAFT_PARSERS), ...Object.keys(SETUP_STEP_PARSERS)].filter(
  (key) => key !== "q",
);

export type SetupDraftParams = inferParserType<typeof SETUP_DRAFT_PARSERS>;
type SetupDraftUpdate = Nullable<SetupDraftParams>;

export const DRAFT_FIELDS = ["name", "selection", "context", "watching", "questions", "repeat", "interval"] as const;
export type DraftFields = Pick<InvestigationInput, (typeof DRAFT_FIELDS)[number]>;

export function draftFromParams(params: SetupDraftParams): InvestigationInput {
  return {
    ...NEW_DRAFT,
    name: params.name,
    selection: {
      ...NEW_DRAFT.selection,
      q: params.q,
      lookback_hours: params.lookback,
      sample_percent: params.sample,
      sample_size: params.max ?? NEW_DRAFT.selection.sample_size,
    },
    context: params.context,
    watching: params.watch.filter((id) => knownWatch.has(id)),
    questions: params.checks.map((instruction) => ({ id: crypto.randomUUID(), instruction, enabled: true })),
    repeat: params.monitor,
    interval: params.every,
  };
}

const finite = (value: number | null) => (value != null && Number.isFinite(value) ? value : null);

/** nuqs drops values equal to their defaults from the URL, so an untouched setup keeps a short link. */
export function paramsFromDraft(draft: DraftFields): SetupDraftUpdate {
  const { selection } = draft;
  return {
    q: selection.q,
    name: draft.name,
    lookback: finite(selection.lookback_hours),
    sample: finite(selection.sample_percent),
    max: finite(selection.sample_size),
    context: draft.context,
    watch: draft.watching,
    checks: draft.questions.map((check) => check.instruction).filter((instruction) => instruction.trim()),
    monitor: draft.repeat,
    every: finite(draft.interval),
  };
}

export function useSetupDraftRoute() {
  const [params, setParams] = useQueryStates(SETUP_DRAFT_PARSERS, { history: "replace" });
  const saveDraft = useCallback((draft: DraftFields) => void setParams(paramsFromDraft(draft)), [setParams]);
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
