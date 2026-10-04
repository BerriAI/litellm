import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import { previewText, traceAgentNames } from "@/components/view_logs/TraceView/traceUtils";

type RunFieldGroup = "Run attributes" | "Content" | "Identity";

interface RunFieldSpec {
  group: RunFieldGroup;
  read: (run: TraceSummary) => readonly string[];
  suggestValues: boolean;
}

const runStatus = (run: TraceSummary): string => (run.error_count > 0 ? "error" : "ok");

export const RUN_FIELDS = {
  name: { group: "Run attributes", read: (run) => [run.name], suggestValues: true },
  agent: { group: "Run attributes", read: traceAgentNames, suggestValues: true },
  status: { group: "Run attributes", read: (run) => [runStatus(run)], suggestValues: true },
  model: { group: "Run attributes", read: (run) => run.models, suggestValues: true },
  input: { group: "Content", read: (run) => [previewText(run.input_preview)], suggestValues: false },
  trace_id: { group: "Identity", read: (run) => [run.trace_id], suggestValues: false },
} as const satisfies Record<string, RunFieldSpec>;

export type RunField = keyof typeof RUN_FIELDS;

const isRunField = (key: string): key is RunField => Object.hasOwn(RUN_FIELDS, key);

interface Span {
  from: number;
  to: number;
}

export type QueryClause =
  | (Span & { kind: "text"; value: string })
  | (Span & { kind: "field"; field: RunField; negated: boolean; keyTo: number; value: string });

const unquote = (raw: string): string => raw.replace(/^"([^"]*)"?$/, "$1");

/** Whitespace-separated tokens; a double-quoted stretch keeps its spaces. */
function tokenize(text: string): (Span & { raw: string })[] {
  return Array.from(text.matchAll(/(?:"[^"]*"?|\S)+/g), (match) => ({
    raw: match[0],
    from: match.index,
    to: match.index + match[0].length,
  }));
}

function parseToken({ raw, from, to }: Span & { raw: string }): QueryClause {
  const match = /^(-?)([A-Za-z_]+):(.*)$/.exec(raw);
  const key = match?.[2].toLowerCase() ?? "";
  if (!match || !isRunField(key)) return { kind: "text", value: unquote(raw), from, to };
  const negated = match[1] === "-";
  return {
    kind: "field",
    field: key,
    negated,
    keyTo: from + match[1].length + match[2].length,
    value: unquote(match[3]),
    from,
    to,
  };
}

export const parseRunQuery = (text: string): QueryClause[] => tokenize(text).map(parseToken);

/** `bar` matches the whole value, `*bar*` is a glob; both ignore case. */
function valueMatcher(pattern: string): (value: string) => boolean {
  const needle = pattern.toLowerCase();
  if (!needle.includes("*")) return (value) => value.toLowerCase() === needle;
  const segments = needle.split("*");
  const first = segments[0];
  const last = segments.at(-1);
  const middle = segments.slice(1, -1).filter(Boolean);
  return (value) => {
    const candidate = value.toLowerCase();
    if (first && !candidate.startsWith(first)) return false;
    if (last && !candidate.endsWith(last)) return false;
    const suffixStart = last ? candidate.length - last.length : candidate.length;
    let cursor = first ? first.length : 0;
    if (suffixStart < cursor) return false;
    for (const segment of middle) {
      const found = candidate.indexOf(segment, cursor);
      if (found < 0 || found + segment.length > suffixStart) return false;
      cursor = found + segment.length;
    }
    return true;
  };
}

function matchesClause(run: TraceSummary, clause: QueryClause): boolean {
  if (clause.kind === "text") {
    const needle = clause.value.toLowerCase();
    return [run.trace_id, previewText(run.input_preview), run.name].some((text) => text.toLowerCase().includes(needle));
  }
  // A key typed without a value yet narrows nothing, so the list does not blank out mid-typing.
  if (!clause.value) return true;
  const matches = RUN_FIELDS[clause.field].read(run).some(valueMatcher(clause.value));
  return clause.negated ? !matches : matches;
}

/** Every clause must hold: free text searches input, name and trace id; `key:value` clauses target one field. */
export function filterRuns(runs: TraceSummary[], query: string): TraceSummary[] {
  const clauses = parseRunQuery(query);
  if (clauses.length === 0) return runs;
  return runs.filter((run) => clauses.every((clause) => matchesClause(run, clause)));
}

export function fieldValues(runs: readonly TraceSummary[], field: RunField): string[] {
  return Array.from(new Set(runs.flatMap((run) => RUN_FIELDS[field].read(run))))
    .filter(Boolean)
    .sort();
}
