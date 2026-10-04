"use client";

import { parseAsBoolean, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback } from "react";
import { OPEN_TRACE_PARSERS, RUN_FILTER_PARSERS } from "@/components/view_logs/TraceView/traceRouting";

export const LENS_TABS = { traces: "Traces", investigations: "Investigations" } as const;
export type LensTab = keyof typeof LENS_TABS;
const lensTabs = Object.keys(LENS_TABS) as LensTab[];

export const RESULT_TABS = ["findings", "checks", "runs", "activity"] as const;
export type ResultTab = (typeof RESULT_TABS)[number];
export const FINDING_KINDS = ["issue", "pattern"] as const;
export const LENS_DIALOGS = ["new", "edit", "duplicate", "run_now", "workers", "monitoring"] as const;
export type LensDialog = (typeof LENS_DIALOGS)[number];

const LENS_PARSERS = {
  tab: parseAsStringLiteral(lensTabs),
  lens: parseAsString,
  demo: parseAsBoolean.withDefault(false),
};

const ISSUE_PARSERS = { issue: parseAsString };

const LIST_PARSERS = { search: parseAsString.withDefault("") };

export const RESULT_PARSERS = {
  run: parseAsString.withDefault("latest"),
  section: parseAsStringLiteral(RESULT_TABS).withDefault("findings"),
  finding: parseAsString,
  finding_status: parseAsString.withDefault("open"),
  kind: parseAsStringLiteral(FINDING_KINDS).withDefault("issue"),
  evidence: parseAsString,
  evidence_span: parseAsString,
};

/** A dialog opened from a list row names its investigation in `target`; inside a detail view it acts on `lens`. */
const DIALOG_PARSERS = { dialog: parseAsStringLiteral(LENS_DIALOGS), target: parseAsString };

const SESSION_PARSERS = {
  ...LENS_PARSERS,
  ...OPEN_TRACE_PARSERS,
  ...RUN_FILTER_PARSERS,
  ...ISSUE_PARSERS,
  ...LIST_PARSERS,
  ...RESULT_PARSERS,
  ...DIALOG_PARSERS,
};
const nulls = <K extends string>(keys: readonly K[]) =>
  Object.fromEntries(keys.map((key) => [key, null])) as Record<K, null>;
/** Switching the sample session clears every Lens key but the tab so ids never cross between live and sample data. */
const CLEARED_SESSION = nulls(Object.keys(SESSION_PARSERS).filter((key) => key !== "tab"));
const CLEARED_RESULTS = nulls(Object.keys(RESULT_PARSERS));

export interface LensRoute {
  readonly tab: LensTab | null;
  readonly lensId: string | null;
  readonly demo: boolean;
  setTab(tab: LensTab): void;
  setLensId(lensId: string | null): void;
  setDemo(demo: boolean): void;
}

/** Lens navigation lives in the URL, sample session included, so any view is a shareable link. */
export function useLensRoute(): LensRoute {
  const [{ tab, lens, demo }, setParams] = useQueryStates(SESSION_PARSERS, { history: "push" });
  const setTab = useCallback((next: LensTab) => void setParams({ tab: next }), [setParams]);
  const setLensId = useCallback(
    (next: string | null) => void setParams({ ...CLEARED_RESULTS, lens: next }),
    [setParams],
  );
  const setDemo = useCallback(
    (next: boolean) => void setParams(next ? { ...CLEARED_SESSION, demo: true } : CLEARED_SESSION),
    [setParams],
  );
  return { tab, lensId: lens, demo, setTab, setLensId, setDemo };
}

export function useIssueRoute() {
  const [{ issue }, setParams] = useQueryStates(ISSUE_PARSERS);
  return {
    issueKey: issue,
    setIssueKey: useCallback(
      (next: string | null) => void setParams({ issue: next }, { history: "push" }),
      [setParams],
    ),
  };
}

export function useListSearchRoute(): [string, (search: string) => void] {
  const [{ search }, setParams] = useQueryStates(LIST_PARSERS);
  return [search, useCallback((next: string) => void setParams({ search: next }), [setParams])];
}

export function useDialogRoute() {
  const [{ dialog, target }, setParams] = useQueryStates(DIALOG_PARSERS, { history: "push" });
  const openDialog = useCallback(
    (next: LensDialog, targetId: string | null = null) => void setParams({ dialog: next, target: targetId }),
    [setParams],
  );
  const closeDialog = useCallback(() => void setParams({ dialog: null, target: null }), [setParams]);
  return { dialog, target, openDialog, closeDialog };
}
