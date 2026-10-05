"use client";

import { useMemo } from "react";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import { SearchBox } from "@/components/shared/search/SearchBox";
import { itemValues } from "@/components/shared/search/valueSource";
import { NEWEST, type RunOrder } from "../runOrder";
import { RUN_INDEX, RUN_QUERY } from "./runQuery";
import { runQueryCommand } from "./runSql";

const COPY_HINT = "Copy this list as a curl call to the trace query API, which takes full SQL over the same rows.";

interface RunSearchProps {
  value: string;
  onChange: (value: string) => void;
  /** Loaded runs, the source of value suggestions. */
  runs: readonly TraceSummary[];
  /** The range and order the list shows; the copied query follows both. */
  range?: TimeWindow;
  order?: RunOrder;
  busy?: boolean;
}

/** The runs list query box: free text plus `key:value` filters over run fields, copyable as a trace query. */
export function RunSearch({ value, onChange, runs, range, order = NEWEST, busy = false }: RunSearchProps) {
  const command = useMemo(() => runQueryCommand(range, order), [range, order]);
  return (
    <SearchBox.Root
      language={RUN_QUERY}
      values={itemValues(RUN_INDEX, runs)}
      value={value}
      onValueChange={onChange}
      label="Search runs"
      className="h-full min-w-0"
    >
      <SearchBox.Input
        className="h-full rounded-none border-0 px-3 focus-within:bg-muted/40 focus-within:ring-0 dark:bg-transparent"
        busy={busy}
        placeholder="Search runs, or filter like agent:researcher status:error"
      />
      <SearchBox.Suggestions>
        <SearchBox.CopyCommand title={COPY_HINT} command={command} />
      </SearchBox.Suggestions>
    </SearchBox.Root>
  );
}
