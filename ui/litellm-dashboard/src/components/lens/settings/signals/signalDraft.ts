import type { SignalConfig } from "../../model/types";
import { SIGNAL_LIBRARY } from "../../model/signals";

export const MAX_SIGNALS = 20;
const MAX_NAME = 60;
const MAX_QUESTION = 500;
const MAX_ID = 64;

export interface SignalRow {
  readonly key: string;
  readonly id: string;
  readonly name: string;
  readonly question: string;
}

export interface SignalDraft {
  readonly model: string;
  readonly thresholdPercent: number;
  readonly rows: readonly SignalRow[];
}

export interface RowProblems {
  readonly name?: string;
  readonly question?: string;
}

export interface DraftProblems {
  readonly threshold?: string;
  readonly signals?: string;
  readonly rows: ReadonlyMap<string, RowProblems>;
  readonly any: boolean;
}

export const draftFrom = (config: SignalConfig): SignalDraft => ({
  model: config.model ?? "",
  thresholdPercent: Math.round((config.threshold ?? 0.5) * 100),
  rows: (config.signals ?? []).map((signal) => ({
    key: signal.id,
    id: signal.id,
    name: signal.name,
    question: signal.question,
  })),
});

export const newRow = (key: string): SignalRow => ({ key, id: "", name: "", question: "" });

const slug = (name: string): string => {
  const words = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const lettered = /^[a-z]/.test(words) ? words : `signal_${words}`.replace(/_+$/, "");
  return lettered.slice(0, MAX_ID);
};

const uniqueId = (base: string, taken: ReadonlySet<string>, attempt = 1): string => {
  const suffix = attempt === 1 ? "" : `_${attempt}`;
  const id = `${base.slice(0, MAX_ID - suffix.length)}${suffix}`;
  return taken.has(id) ? uniqueId(base, taken, attempt + 1) : id;
};

export const signalIds = (rows: readonly SignalRow[]): string[] => {
  const savedIds: ReadonlySet<string> = new Set(rows.flatMap((row) => (row.id ? [row.id] : [])));
  return rows.reduce<string[]>((ids, row) => {
    if (row.id) return [...ids, row.id];
    const librarySignal = SIGNAL_LIBRARY.find((signal) => signal.question === row.question);
    const libraryIds = new Set(
      SIGNAL_LIBRARY.filter((signal) => signal.id !== librarySignal?.id).map((signal) => signal.id),
    );
    const taken = new Set([...savedIds, ...ids, ...libraryIds]);
    return [...ids, uniqueId(librarySignal?.id ?? slug(row.name), taken)];
  }, []);
};

const rowProblems = (row: SignalRow, duplicateName: boolean): RowProblems => {
  const name = row.name.trim();
  const question = row.question.trim();
  const nameProblem = [
    !name ? "Name the signal" : undefined,
    name.length > MAX_NAME ? `Keep the name under ${MAX_NAME} characters` : undefined,
    duplicateName ? "Another signal has this name" : undefined,
  ].find((problem) => problem !== undefined);
  const questionProblem = [
    question.length < 3 ? "Ask a yes or no question about the run" : undefined,
    question.length > MAX_QUESTION ? `Keep the question under ${MAX_QUESTION} characters` : undefined,
  ].find((problem) => problem !== undefined);
  return { name: nameProblem, question: questionProblem };
};

export function draftProblems(draft: SignalDraft): DraftProblems {
  const names = draft.rows.map((row) => row.name.trim().toLowerCase());
  const rows = new Map(
    draft.rows
      .map((row, index): [string, RowProblems] => [
        row.key,
        rowProblems(row, names[index] !== "" && names.indexOf(names[index]) !== index),
      ])
      .filter(([, problems]) => problems.name || problems.question),
  );
  const { thresholdPercent } = draft;
  const threshold =
    Number.isInteger(thresholdPercent) && thresholdPercent >= 5 && thresholdPercent <= 95
      ? undefined
      : "Use a whole number from 5 to 95";
  const signals = draft.rows.length > MAX_SIGNALS ? `Use at most ${MAX_SIGNALS} signals` : undefined;
  return { threshold, signals, rows, any: Boolean(threshold || signals || rows.size) };
}

export function configFrom(draft: SignalDraft): SignalConfig {
  const ids = signalIds(draft.rows);
  return {
    model: draft.model,
    threshold: draft.thresholdPercent / 100,
    signals: draft.rows.map((row, index) => ({ id: ids[index], name: row.name.trim(), question: row.question.trim() })),
  };
}
