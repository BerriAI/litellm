"use client";

import { useMemo } from "react";

import type { TraceSummary } from "../../types";
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import { SearchBox } from "@/components/shared/search/SearchBox";
import { itemValues } from "@/components/shared/search/valueSource";
import { RUN_INDEX, RUN_QUERY } from "./runQuery";
import { runQueryCommand } from "./runSql";

const COPY_HINT = "Copy this list as a curl call to the trace query API, which takes full SQL over the same rows.";

interface RunSearchProps {
  value: string;
  onChange: (value: string) => void;
  /** Loaded runs, the source of value suggestions. */
  runs: readonly TraceSummary[];
  /** The range the list shows; the copied query bounds itself to it. */
  range?: TimeWindow;
  busy?: boolean;
}

/** The runs list query box: free text plus `key:value` filters over run fields, copyable as a trace query. */
export function RunSearch({ value, onChange, runs, range, busy = false }: RunSearchProps) {
  const command = useMemo(() => runQueryCommand(range), [range]);
  return (
    <SearchBox.Root
      language={RUN_QUERY}
      values={itemValues(RUN_INDEX, runs)}
      value={value}
      onValueChange={onChange}
      label="Search runs"
      className="min-w-40 basis-full sm:basis-72 sm:max-w-96"
    >
      <SearchBox.Input
        className="h-8 overflow-hidden rounded-md px-3 whitespace-nowrap"
        placeholder="Search input or trace ID"
        busy={busy}
      />
      <SearchBox.Suggestions>
        <SearchBox.CopyCommand title={COPY_HINT} command={command} />
      </SearchBox.Suggestions>
    </SearchBox.Root>
  );
}
