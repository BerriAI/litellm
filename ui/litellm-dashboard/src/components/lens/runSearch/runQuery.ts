import { Bot, Box, Braces, CircleDashed, Hash, SquareChevronRight } from "lucide-react";

import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import { previewText, traceAgentNames } from "@/components/view_logs/TraceView/traceUtils";

import { type FieldSpec, filterItems, type QueryLanguage } from "../search/language";

const runStatus = (run: TraceSummary): string => (run.error_count > 0 ? "error" : "ok");

const RUN_FIELDS = {
  name: { group: "Run attributes", icon: SquareChevronRight, read: (run) => [run.name], suggestValues: true },
  agent: { group: "Run attributes", icon: Bot, read: traceAgentNames, suggestValues: true },
  status: { group: "Run attributes", icon: CircleDashed, read: (run) => [runStatus(run)], suggestValues: true },
  model: { group: "Run attributes", icon: Box, read: (run) => run.models, suggestValues: true },
  input: { group: "Content", icon: Braces, read: (run) => [previewText(run.input_preview)], suggestValues: false },
  trace_id: { group: "Identity", icon: Hash, read: (run) => [run.trace_id], suggestValues: false },
} as const satisfies Record<string, FieldSpec<TraceSummary>>;

export type RunField = keyof typeof RUN_FIELDS;

/** Free text searches input, name and trace id; `key:value` clauses target one run field. */
export const RUN_QUERY: QueryLanguage<TraceSummary, RunField> = {
  fields: RUN_FIELDS,
  freeText: (run) => [run.trace_id, previewText(run.input_preview), run.name],
};

export const filterRuns = (runs: TraceSummary[], query: string): TraceSummary[] => filterItems(RUN_QUERY, runs, query);
