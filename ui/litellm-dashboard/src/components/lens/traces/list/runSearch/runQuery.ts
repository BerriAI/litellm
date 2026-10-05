import { Bot, Box, Braces, CircleDashed, Hash, SquareChevronRight } from "lucide-react";

import type { TraceSummary } from "../../types";
import { previewText, traceAgentNames } from "../../utils";

import { type ClientIndex, filterItems } from "@/components/shared/search/evaluate";
import { ALL_OPERATORS, type FieldSpec, type QueryLanguage } from "@/components/shared/search/language";

const RUN_FIELDS = {
  name: { group: "Run attributes", icon: SquareChevronRight, suggestValues: true },
  agent: { group: "Run attributes", icon: Bot, suggestValues: true },
  status: { group: "Run attributes", icon: CircleDashed, suggestValues: true },
  model: { group: "Run attributes", icon: Box, suggestValues: true },
  input: { group: "Content", icon: Braces, suggestValues: false },
  trace_id: { group: "Identity", icon: Hash, suggestValues: false },
} as const satisfies Record<string, FieldSpec>;

export type RunField = keyof typeof RUN_FIELDS;

export const RUN_QUERY: QueryLanguage<RunField> = { fields: RUN_FIELDS, ops: ALL_OPERATORS };

/** Reads the run fields off a loaded page; free text searches trace id, input and name. */
export const RUN_INDEX: ClientIndex<TraceSummary, RunField> = {
  read: {
    name: (run) => [run.name],
    agent: traceAgentNames,
    status: (run) => [run.error_count > 0 ? "error" : "ok"],
    model: (run) => run.models,
    input: (run) => [previewText(run.input_preview)],
    trace_id: (run) => [run.trace_id],
  },
  freeText: (run) => [run.trace_id, previewText(run.input_preview), run.name],
};

export function filterRuns(
  runs: TraceSummary[],
  query: string,
  filters: { agent: string; status: "all" | "ok" | "error" } = { agent: "", status: "all" },
): TraceSummary[] {
  const matchesAgent = (run: TraceSummary) => !filters.agent || traceAgentNames(run).includes(filters.agent);
  const matchesStatus = (run: TraceSummary) =>
    filters.status === "all" || (run.error_count > 0 ? "error" : "ok") === filters.status;
  return filterItems(RUN_QUERY, RUN_INDEX, runs, query).filter((run) => matchesAgent(run) && matchesStatus(run));
}
