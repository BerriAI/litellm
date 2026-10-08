"use client";

import { parseAsBoolean, parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback } from "react";
import { OPEN_TRACE_PARSERS, RUN_FILTER_PARSERS } from "@/components/lens/traces/routing";

export const LENS_TABS = {
  traces: "Traces",
  findings: "Findings",
  investigations: "Investigations",
  datasets: "Datasets",
  settings: "Settings",
} as const;
export type LensTab = keyof typeof LENS_TABS;
const lensTabs = Object.keys(LENS_TABS) as LensTab[];

const RESULT_TABS = ["findings", "checks", "runs", "activity"] as const;
export type ResultTab = (typeof RESULT_TABS)[number];
const FINDING_KINDS = ["issue", "pattern"] as const;
export const LENS_DIALOGS = ["new", "edit", "duplicate", "run_now", "monitoring"] as const;
export type LensDialog = (typeof LENS_DIALOGS)[number];

const LENS_PARSERS = {
  tab: parseAsStringLiteral(lensTabs),
  lens: parseAsString,
  demo: parseAsBoolean.withDefault(false),
  setup: parseAsStringLiteral(["lens"]),
};

const ISSUE_PARSERS = { issue: parseAsString };

const INBOX_PARSERS = {
  inbox_agent: parseAsString.withDefault("all"),
  priority: parseAsStringLiteral(["all", "high", "medium", "low"]).withDefault("all"),
};

const DATASET_PARSERS = { dataset: parseAsString, revision: parseAsInteger, case: parseAsString };

const LIST_PARSERS = { search: parseAsString.withDefault("") };

const RESULT_PARSERS = {
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
  ...INBOX_PARSERS,
  ...RESULT_PARSERS,
  ...DIALOG_PARSERS,
  ...DATASET_PARSERS,
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
  readonly settingUp: boolean;
  setTab(tab: LensTab): void;
  setLensId(lensId: string | null): void;
  setDemo(demo: boolean): void;
  setSetup(settingUp: boolean): void;
}

/** Lens navigation lives in the URL, sample session included, so any view is a shareable link. */
export function useLensRoute(): LensRoute {
  const [{ tab, lens, demo, setup }, setParams] = useQueryStates(SESSION_PARSERS, { history: "push" });
  const setTab = useCallback((next: LensTab) => void setParams({ tab: next }), [setParams]);
  const setLensId = useCallback(
    (next: string | null) => void setParams({ ...CLEARED_RESULTS, lens: next, issue: null }),
    [setParams],
  );
  const setDemo = useCallback(
    (next: boolean) => void setParams(next ? { ...CLEARED_SESSION, demo: true } : CLEARED_SESSION),
    [setParams],
  );
  const setSetup = useCallback((next: boolean) => void setParams({ setup: next ? "lens" : null }), [setParams]);
  return { tab, lensId: lens, demo, settingUp: setup === "lens", setTab, setLensId, setDemo, setSetup };
}

const ISSUE_ROUTE_PARSERS = { ...ISSUE_PARSERS, lens: LENS_PARSERS.lens, ...RESULT_PARSERS };

/** A peeked finding takes the panel over from any open investigation and starts from its summary. */
export function useIssueRoute() {
  const [{ issue }, setParams] = useQueryStates(ISSUE_ROUTE_PARSERS);
  return {
    issueKey: issue,
    setIssueKey: useCallback(
      (next: string | null) => void setParams({ ...CLEARED_RESULTS, lens: null, issue: next }, { history: "push" }),
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

const isResultTab = (tab: string): tab is ResultTab => (RESULT_TABS as readonly string[]).includes(tab);

type FindingKind = (typeof FINDING_KINDS)[number];

export interface EvidenceRef {
  readonly id: string;
  readonly span: string;
}

const NO_RESULT_OPEN = { finding: null, evidence: null, evidence_span: null } as const;

export function useRunRoute() {
  const [{ run }, setParams] = useQueryStates(RESULT_PARSERS);
  const selectRun = useCallback((id: string) => void setParams({ run: id, ...NO_RESULT_OPEN }), [setParams]);
  const openRun = useCallback(
    (id: string) => void setParams({ run: id, section: null, ...NO_RESULT_OPEN }),
    [setParams],
  );
  return { batchId: run, selectRun, openRun };
}

export function useSectionRoute() {
  const [{ section }, setParams] = useQueryStates(RESULT_PARSERS);
  const setSection = useCallback(
    (next: string) => void (isResultTab(next) && setParams({ section: next })),
    [setParams],
  );
  return { section, setSection };
}

/** Picking a finding starts from its summary, so any evidence stacked over the previous one is dropped. */
export function useFindingRoute() {
  const [{ finding }, setParams] = useQueryStates(RESULT_PARSERS);
  const setFindingId = useCallback(
    (next: string | null) =>
      void setParams({ finding: next, evidence: null, evidence_span: null }, { history: "push" }),
    [setParams],
  );
  return { findingId: finding, setFindingId };
}

/** A run opened from the Runs tab is evidence on its own, outside any finding. */
export function useRunEvidenceRoute() {
  const [{ finding, evidence }, setParams] = useQueryStates(RESULT_PARSERS);
  const selectRun = useCallback(
    (id: string | null) => void setParams({ finding: null, evidence: id, evidence_span: null }, { history: "push" }),
    [setParams],
  );
  return { runId: finding === null ? evidence : null, selectRun };
}

export function useFindingFilters() {
  const [{ kind, finding_status }, setParams] = useQueryStates(RESULT_PARSERS);
  const setKind = useCallback((next: FindingKind) => void setParams({ kind: next }), [setParams]);
  const setStatus = useCallback((next: string) => void setParams({ finding_status: next }), [setParams]);
  return { kind, setKind, status: finding_status, setStatus };
}

export function useEvidenceRoute() {
  const [{ evidence, evidence_span }, setParams] = useQueryStates(RESULT_PARSERS);
  const setEvidence = useCallback(
    (next: EvidenceRef | null) =>
      void setParams({ evidence: next?.id ?? null, evidence_span: next?.span || null }, { history: "push" }),
    [setParams],
  );
  const current: EvidenceRef | null = evidence === null ? null : { id: evidence, span: evidence_span ?? "" };
  return { evidence: current, setEvidence };
}

export function useInboxFilters() {
  const [{ inbox_agent, priority }, setParams] = useQueryStates(INBOX_PARSERS);
  return {
    agent: inbox_agent,
    priority,
    setAgent: (agent: string) => void setParams({ inbox_agent: agent }),
    setPriority: (priority: "all" | "high" | "medium" | "low") => void setParams({ priority }),
  };
}

/** `revision` null means the latest revision, so a dataset link keeps following new saves. */
export function useDatasetRoute() {
  const [{ dataset, revision, case: caseId }, setParams] = useQueryStates(DATASET_PARSERS, { history: "push" });
  const openDataset = useCallback(
    (next: string | null) => void setParams({ dataset: next, revision: null, case: null }),
    [setParams],
  );
  const setRevision = useCallback(
    (next: number | null) => void setParams({ revision: next }, { history: "replace" }),
    [setParams],
  );
  const setCaseId = useCallback((next: string | null) => void setParams({ case: next }), [setParams]);
  return { datasetId: dataset, revision, caseId, openDataset, setRevision, setCaseId };
}

const SOURCE_TRACE_PARSERS = { tab: LENS_PARSERS.tab, ...OPEN_TRACE_PARSERS };

export interface SourceTrace {
  readonly traceId: string;
  readonly traceRef: string;
  readonly spanId: string;
}

const FRESH_TRACE_VIEW = { view: null, span_tab: null, steps_q: null, errors: null } as const;

export function useOpenSourceTrace() {
  const [, setParams] = useQueryStates(SOURCE_TRACE_PARSERS, { history: "push" });
  return useCallback(
    (source: SourceTrace) => {
      const opened = {
        ...FRESH_TRACE_VIEW,
        tab: "traces" as const,
        trace: source.traceId,
        trace_ref: source.traceRef || null,
        span: source.spanId || null,
      };
      void setParams(opened);
    },
    [setParams],
  );
}
