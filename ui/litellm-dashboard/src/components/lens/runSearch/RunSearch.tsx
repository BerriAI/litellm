"use client";

import { useMemo } from "react";

import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import type { TimeWindow } from "@/components/view_logs/TraceView/TracesTimeline";

import { SearchBox } from "../search/SearchBox";
import { itemValues } from "../search/valueSource";
import { RUN_INDEX, RUN_QUERY } from "./runQuery";
import { runApiEquivalent } from "./runSql";

const SQL_HINT = "Lens filters are a subset of the trace query API, which takes full SQL over the same rows.";

interface RunSearchProps {
  value: string;
  onChange: (value: string) => void;
  /** Loaded runs, the source of value suggestions. */
  runs: readonly TraceSummary[];
  /** The range the list shows; the SQL equivalent bounds itself to it. */
  range?: TimeWindow;
}

/** The runs list query box: free text plus `key:value` filters over run fields, mirrored as trace SQL. */
export function RunSearch({ value, onChange, runs, range }: RunSearchProps) {
  const translate = useMemo(() => runApiEquivalent(range), [range]);
  return (
    <SearchBox.Root
      language={RUN_QUERY}
      values={itemValues(RUN_INDEX, runs)}
      value={value}
      onValueChange={onChange}
      label="Search runs"
    >
      <SearchBox.Input placeholder="Search runs, or filter like agent:researcher status:error" />
      <SearchBox.Suggestions>
        <SearchBox.ApiHint dialect="SQL" title={SQL_HINT} translate={translate} />
      </SearchBox.Suggestions>
    </SearchBox.Root>
  );
}
