"use client";

import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";

import { SearchBox } from "../search/SearchBox";
import { itemValues } from "../search/valueSource";
import { RUN_INDEX, RUN_QUERY } from "./runQuery";

interface RunSearchProps {
  value: string;
  onChange: (value: string) => void;
  /** Loaded runs, the source of value suggestions. */
  runs: readonly TraceSummary[];
}

/** The runs list query box: free text plus `key:value` filters over run fields. */
export function RunSearch({ value, onChange, runs }: RunSearchProps) {
  return (
    <SearchBox.Root
      language={RUN_QUERY}
      values={itemValues(RUN_INDEX, runs)}
      value={value}
      onValueChange={onChange}
      label="Search runs"
    >
      <SearchBox.Input placeholder="Search runs, or filter like agent:researcher status:error" />
      <SearchBox.Suggestions />
    </SearchBox.Root>
  );
}
