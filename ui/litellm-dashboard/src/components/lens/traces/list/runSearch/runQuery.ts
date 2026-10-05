import { Bot, Box, Braces, CircleDashed, Hash, SquareChevronRight } from "lucide-react";

import type { RunField as TraceRunField, TraceSummary } from "../../types";
import { traceAgentNames } from "../../utils";

import { type ClientIndex, filterItems } from "@/components/shared/search/evaluate";
import { ALL_OPERATORS, type FieldSpec, type QueryLanguage } from "@/components/shared/search/language";

const RUN_FIELDS = {
  name: { group: "Run attributes", icon: SquareChevronRight, suggestValues: true },
  agent: { group: "Run attributes", icon: Bot, suggestValues: true },
  root_status: { group: "Run attributes", icon: CircleDashed, suggestValues: true },
  has_error: { group: "Run attributes", icon: CircleDashed, suggestValues: true },
  service: { group: "Run attributes", icon: Box, suggestValues: true },
  model: { group: "Run attributes", icon: Box, suggestValues: true },
  input: { group: "Content", icon: Braces, suggestValues: false },
  trace_id: { group: "Identity", icon: Hash, suggestValues: false },
} as const satisfies Partial<Record<TraceRunField, FieldSpec>>;

export type RunField = keyof typeof RUN_FIELDS;

export const RUN_QUERY: QueryLanguage<RunField> = { fields: RUN_FIELDS, ops: ALL_OPERATORS };

/** Reads the run fields off a loaded page; free text searches trace id, input and name. */
export const RUN_INDEX: ClientIndex<TraceSummary, RunField> = {
  read: {
    name: (run) => [run.name],
    agent: traceAgentNames,
    root_status: (run) => [run.root_status],
    has_error: (run) => [String(run.has_error)],
    service: (run) => [run.service],
    model: (run) => run.models,
    input: (run) => [run.input_preview],
    trace_id: (run) => [run.trace_id],
  },
  freeText: (run) => [run.trace_id, run.input_preview, run.name],
};

export const filterRuns = (runs: TraceSummary[], query: string): TraceSummary[] =>
  filterItems(RUN_QUERY, RUN_INDEX, runs, query);
