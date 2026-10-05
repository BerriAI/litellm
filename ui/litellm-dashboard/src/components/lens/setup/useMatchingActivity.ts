"use client";

import { useCallback, useEffect, useState } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { useFormContext, useWatch } from "react-hook-form";
import { lensKeys, lensQueries } from "../data/queries";
import { useLensApi } from "../data/LensServices";
import { durationLabel } from "../model/format";
import type { Execution, Sample } from "../model/types";
import type { TraceSummary } from "../traces/types";
import { useDebouncedValue } from "./useDebouncedValue";
import type { InvestigationInput } from "./investigationSchema";

const PREVIEW_DEBOUNCE_MS = 350;
const EMPTY_PREVIEW_POLL_MS = 15000;

type Selection = InvestigationInput["selection"];
type PreviewPageData = Pick<Sample, "eligible" | "selected">;

export type { Execution };

/** What the search box needs: the runs already previewed suggest values, and the window bounds the copied query. */
export interface ScopeOptions {
  readonly runs: readonly TraceSummary[];
  readonly range: { readonly startMs: number; readonly endMs: number };
}

export interface PreviewStatus {
  readonly title: string;
  readonly windowLabel: string;
  readonly ready: boolean;
  readonly error: Error | null;
  readonly refresh: () => void;
}

export interface PreviewPage {
  readonly eligible: number | undefined;
  readonly selected: number;
  readonly executions: readonly Execution[];
  readonly hasMore: boolean;
  readonly loadingMore: boolean;
  readonly loadMore: () => void;
}

/** Present only while the user picks individual runs; `count` applies the sampling settings to the picks. */
export interface PreviewSelection {
  readonly ids: readonly string[];
  readonly count: number;
  readonly toggle: (id: string, checked: boolean) => void;
  readonly clear: () => void;
}

export interface MatchingPreview {
  readonly status: PreviewStatus;
  readonly page: PreviewPage;
  readonly selection: PreviewSelection | null;
}

export interface MatchingActivity {
  readonly scope: ScopeOptions;
  readonly preview: MatchingPreview;
  readonly canReview: boolean;
}

function validWindow(selection: Selection): boolean {
  const hours = selection.lookback_hours ?? 24;
  return Number.isInteger(hours) && hours >= 1;
}

function validScope(scope: Selection): boolean {
  const percent = scope.sample_percent ?? 100;
  const cap = scope.sample_size;
  const validCap = cap == null || (Number.isInteger(cap) && cap > 0);
  const validSampling = percent > 0 && percent <= 100 && validCap;
  return validWindow(scope) && validSampling;
}

function previewTitle(
  state: { pending: boolean; validWindow: boolean; valid: boolean },
  data: PreviewPageData | undefined,
): string {
  if (!state.validWindow) return "Choose a history window between 1 hour and 365 days";
  if (!state.valid) return "Complete your sampling settings to preview matches";
  if (state.pending) return "Finding matching runs…";
  if (!data) return "Preview unavailable";
  return `${data.eligible} matching run${data.eligible === 1 ? "" : "s"}`;
}

function manualSelectedCount(selection: Selection): number {
  const sampled = Math.ceil((selection.execution_ids.length * (selection.sample_percent ?? 100)) / 100);
  return Math.min(sampled, selection.sample_size ?? Infinity);
}

function manualPicks(selection: Selection, setExecutionIds: (ids: readonly string[]) => void): PreviewSelection {
  const ids = selection.execution_ids;
  return {
    ids,
    count: manualSelectedCount(selection),
    toggle: (id, checked) => setExecutionIds(checked ? [...ids, id] : ids.filter((other) => other !== id)),
    clear: () => setExecutionIds([]),
  };
}

function windowLabel(selection: Selection): string {
  if (!validWindow(selection)) return "Choose a valid history window";
  return `Last ${durationLabel(selection.lookback_hours ?? 24, "hours")}`;
}

function scopeOptions(selection: Selection, asOf: string, executions: readonly Execution[]): ScopeOptions {
  const endMs = Date.parse(asOf) || Date.now();
  return {
    runs: executions.flatMap((run) => (run.summary ? [run.summary] : [])),
    range: { startMs: endMs - (selection.lookback_hours ?? 24) * 3_600_000, endMs },
  };
}

/** Live preview of the activity a draft selection matches, debounced so typing a filter does not spam the API. */
export function useMatchingActivity(): MatchingActivity {
  const { control, setValue } = useFormContext<InvestigationInput>();
  const [selection, manualSelection] = useWatch({ control, name: ["selection", "manualSelection"] });
  const api = useLensApi();
  const client = useQueryClient();
  const {
    value: scope,
    pending: settling,
    settledAt,
  } = useDebouncedValue<Selection>({ ...selection, execution_ids: [] }, PREVIEW_DEBOUNCE_MS);
  const [refreshedAt, setRefreshedAt] = useState(settledAt);
  const asOf = settledAt > refreshedAt ? settledAt : refreshedAt;
  const windowValid = validWindow(selection);
  const valid = windowValid && validScope(scope);
  const preview = useInfiniteQuery(lensQueries.preview(api, { scope, asOf, enabled: valid }));
  const firstPage = preview.data?.pages[0];
  const executions = preview.data?.pages.flatMap((page) => page.executions) ?? [];
  const empty = firstPage?.eligible === 0;
  const refresh = useCallback(() => {
    setRefreshedAt(new Date().toISOString());
    void client.invalidateQueries({ queryKey: [...lensKeys.all, "preview"] });
  }, [client]);
  useEffect(() => {
    if (!empty || !valid) return;
    const timer = window.setTimeout(refresh, EMPTY_PREVIEW_POLL_MS);
    return () => window.clearTimeout(timer);
  }, [empty, valid, asOf, refresh]);
  const pending = settling || preview.isLoading || preview.isPlaceholderData;
  const ready = !pending && valid;
  const setExecutionIds = (next: readonly string[]) =>
    setValue("selection.execution_ids", [...next], { shouldValidate: true });
  const picked = manualSelection ? manualPicks(selection, setExecutionIds) : null;
  const hasMatches = !preview.error && (firstPage?.selected ?? 0) > 0;
  const hasSelection = !picked || picked.ids.length > 0;
  return {
    scope: scopeOptions(selection, asOf, executions),
    preview: {
      status: {
        title: previewTitle({ pending, validWindow: windowValid, valid }, firstPage),
        windowLabel: windowLabel(selection),
        ready,
        error: preview.error,
        refresh,
      },
      page: {
        eligible: firstPage?.eligible,
        selected: firstPage?.selected ?? 0,
        executions,
        hasMore: preview.hasNextPage,
        loadingMore: preview.isFetchingNextPage,
        loadMore: () => {
          if (preview.hasNextPage && !preview.isFetching) void preview.fetchNextPage({ cancelRefetch: false });
        },
      },
      selection: picked,
    },
    canReview: ready && hasMatches && hasSelection,
  };
}
