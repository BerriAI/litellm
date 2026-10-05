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
} from "nuqs";
import { useCallback } from "react";
import { RUN_FILTER_PARSERS } from "../traces/routing";
import { watches } from "../model/watches";
import { investigationDefaults, SETUP_STEPS, type InvestigationInput, type SetupStep } from "./investigationSchema";

/** A new investigation's draft lives in the URL, so a search on the Traces tab can open setup already filled in. */
export const SETUP_DRAFT_PARSERS = {
  q: RUN_FILTER_PARSERS.q,
  name: parseAsString,
  lookback: parseAsInteger,
  sample: parseAsFloat,
  max: parseAsInteger,
  context: parseAsString,
  watch: parseAsArrayOf(parseAsString),
  checks: parseAsArrayOf(parseAsString),
  monitor: parseAsBoolean,
  every: parseAsInteger,
};

export const SETUP_STEP_PARSERS = { step: parseAsStringLiteral(SETUP_STEPS) };

/** Every setup key but the shared run search, which belongs to the Traces tab as much as to the draft. */
export const SETUP_KEYS = [...Object.keys(SETUP_DRAFT_PARSERS), ...Object.keys(SETUP_STEP_PARSERS)].filter(
  (key) => key !== "q",
);

export type SetupDraftParams = inferParserType<typeof SETUP_DRAFT_PARSERS>;

export const DRAFT_FIELDS = ["name", "selection", "context", "watching", "questions", "repeat", "interval"] as const;
export type DraftFields = Pick<InvestigationInput, (typeof DRAFT_FIELDS)[number]>;

const NEW_DRAFT = investigationDefaults(undefined, "new");
const knownWatch = new Set(watches.map((watch) => watch.id));
const sameList = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((value, index) => value === b[index]);
const unlessDefault = <T>(value: T, fallback: T): T | null => (value === fallback ? null : value);

export function draftFromParams(params: SetupDraftParams): InvestigationInput {
  const selection = NEW_DRAFT.selection;
  return {
    ...NEW_DRAFT,
    name: params.name ?? NEW_DRAFT.name,
    selection: {
      ...selection,
      q: params.q,
      lookback_hours: params.lookback ?? selection.lookback_hours,
      sample_percent: params.sample ?? selection.sample_percent,
      sample_size: params.max ?? selection.sample_size,
    },
    context: params.context ?? NEW_DRAFT.context,
    watching: params.watch ? params.watch.filter((id) => knownWatch.has(id)) : NEW_DRAFT.watching,
    questions: (params.checks ?? []).map((instruction) => ({ id: crypto.randomUUID(), instruction, enabled: true })),
    repeat: params.monitor ?? NEW_DRAFT.repeat,
    interval: params.every ?? NEW_DRAFT.interval,
  };
}

/** Only what differs from a blank draft reaches the URL, so an untouched setup keeps a short link. */
export function paramsFromDraft(draft: DraftFields): SetupDraftParams {
  const { selection } = draft;
  const checks = draft.questions.map((check) => check.instruction).filter((instruction) => instruction.trim());
  const finite = (value: number | null) => (value != null && Number.isFinite(value) ? value : null);
  return {
    q: selection.q,
    name: draft.name || null,
    lookback: unlessDefault(finite(selection.lookback_hours), NEW_DRAFT.selection.lookback_hours),
    sample: unlessDefault(finite(selection.sample_percent), NEW_DRAFT.selection.sample_percent),
    max: finite(selection.sample_size),
    context: draft.context || null,
    watch: sameList(draft.watching, NEW_DRAFT.watching) ? null : draft.watching,
    checks: checks.length ? checks : null,
    monitor: unlessDefault(draft.repeat, NEW_DRAFT.repeat),
    every: unlessDefault(finite(draft.interval), NEW_DRAFT.interval),
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
