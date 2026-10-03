"use client";

import { ScopeFields } from "./fields/ScopeFields";
import { SampleFields } from "./fields/SampleFields";
import type { ActivitySelection } from "../model/types";
import { MatchingActivityPreview } from "./MatchingActivityPreview";

import { lensQueries } from "../api/queries";

import { useEffect, useId, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLensApi } from "../api/useLensApi";
import { Button } from "@/components/ui/button";

import { TraceSheet } from "../investigations/TraceSheet";
import { durationLabel } from "../model/format";

const selectClass = "h-9 w-full rounded-md border border-input bg-background px-3 text-sm";

export function MatchingActivity({
  value,
  onChange,
  accessToken,
  mode = "scope",
  onPreviewReady,
  manualSelection = false,
  nameField,
}: {
  value: ActivitySelection;
  onChange: (selection: ActivitySelection) => void;
  accessToken: string;
  mode?: "scope" | "activity";
  onPreviewReady?: (ready: boolean) => void;
  manualSelection?: boolean;
  nameField?: ReactNode;
}) {
  const apiClient = useLensApi();
  const id = useId();
  const [offset, setOffset] = useState(0);
  const [scope, setScope] = useState(value);
  const [trace, setTrace] = useState<{ id: string; ref?: string } | null>(null);
  const [asOf, setAsOf] = useState(() => new Date().toISOString());
  const serialized = JSON.stringify({ ...value, execution_ids: [] });
  useEffect(() => {
    const timer = setTimeout(() => {
      setScope(JSON.parse(serialized) as ActivitySelection);
      setOffset(0);
      setAsOf(new Date().toISOString());
    }, 350);
    return () => clearTimeout(timer);
  }, [serialized]);
  const historyHours = value.lookback_hours ?? 24;
  const validWindow = Number.isInteger(historyHours) && historyHours >= 1 && historyHours <= 8760;
  const percent = scope.sample_percent ?? 100;
  const cap = scope.sample_size;
  const validCap = cap == null || (Number.isInteger(cap) && cap > 0);
  const validSampling = percent > 0 && percent <= 100 && validCap;
  const validFilters = (scope.filters ?? []).every((f) => f.key.trim() && f.value.trim());
  const valid = validWindow && validSampling && validFilters;
  const discovery = useQuery(lensQueries.discovery(apiClient, accessToken, { value, asOf, enabled: validWindow }));
  const agents = useQuery(lensQueries.agents(apiClient, accessToken, asOf, value.source));
  const previewInput = { scope, offset, asOf, enabled: valid };
  const preview = useQuery(lensQueries.preview(apiClient, accessToken, previewInput));
  const empty = preview.data?.eligible === 0;
  useEffect(() => {
    if (!empty || !valid) return;
    const timer = window.setTimeout(() => setAsOf(new Date().toISOString()), 15000);
    return () => window.clearTimeout(timer);
  }, [empty, valid, asOf]);
  const refreshPreview = () => {
    setOffset(0);
    setAsOf(new Date().toISOString());
  };
  const runs = discovery.data?.executions ?? [];
  const services = [...new Set(runs.map((r) => r.service).filter(Boolean))].sort();
  const selectedName = value.source === "requests" ? value.service : value.agent_name;
  const names = value.source === "requests" ? services : agents.data ?? [];
  const selectName = (name: string) =>
    onChange({ ...value, [value.source === "requests" ? "service" : "agent_name"]: name, execution_ids: [] });
  const attributes = runs.flatMap((r) => r.metadata ?? []);
  const keys = [...new Set(attributes.map((a) => a.key).filter((key) => !key.startsWith("litellm.")))].sort();
  const pending = serialized !== JSON.stringify(scope) || preview.isFetching;
  const ready = !pending && valid;
  const hasSelection = !manualSelection || !!value.execution_ids?.length;
  const hasMatches = !preview.error && (preview.data?.selected ?? 0) > 0;
  const canReview = ready && hasMatches && hasSelection;
  useEffect(() => {
    onPreviewReady?.(canReview);
  }, [canReview, onPreviewReady]);
  const windowLabel = validWindow
    ? `Last ${durationLabel(value.lookback_hours ?? 24, "hours")}`
    : "Choose a valid history window";
  const previewTitle = () => {
    if (pending) return "Finding matching activity…";
    if (!validWindow) return "Choose a history window between 1 hour and 365 days";
    if (!valid) return "Complete your condition to preview matches";
    if (!preview.data) return "Preview unavailable";
    const noun = value.source === "requests" ? "request" : "run";
    return `${preview.data.eligible} matching ${noun}${preview.data.eligible === 1 ? "" : "s"}`;
  };
  return (
    <div className={mode === "activity" ? "grid gap-6 sm:grid-cols-2" : "space-y-5"}>
      <div className="space-y-5">
        {mode === "scope" ? (
          <ScopeFields
            value={value}
            onChange={onChange}
            nameField={nameField}
            names={names}
            selectedName={selectedName}
            selectName={selectName}
            agents={agents}
            attributes={attributes}
            keys={keys}
            id={id}
          />
        ) : (
          <>
            <SampleFields value={value} onChange={onChange} />
          </>
        )}
        {manualSelection && !!value.execution_ids?.length && (
          <Button variant="outline" size="sm" onClick={() => onChange({ ...value, execution_ids: [] })}>
            Clear {value.execution_ids.length} selected runs
          </Button>
        )}
      </div>
      {mode === "activity" && (
        <MatchingActivityPreview
          offset={offset}
          onPage={setOffset}
          onSelect={(runId, checked) =>
            onChange({
              ...value,
              execution_ids: checked
                ? [...(value.execution_ids ?? []), runId]
                : (value.execution_ids ?? []).filter((id) => id !== runId),
            })
          }
          manualSelection={manualSelection}
          selectedIds={value.execution_ids ?? []}
          selectedCount={
            manualSelection
              ? Math.min(
                  Math.ceil(((value.execution_ids?.length ?? 0) * (value.sample_percent ?? 100)) / 100),
                  value.sample_size ?? Infinity,
                )
              : preview.data?.selected ?? 0
          }
          title={previewTitle()}
          windowLabel={windowLabel}
          ready={ready}
          error={preview.error}
          data={preview.data}
          onRetry={refreshPreview}
          onOpen={(run) => setTrace({ id: run.trace_id, ref: run.trace_ref })}
        />
      )}
      {trace && (
        <TraceSheet
          open
          traceId={trace.id}
          traceRef={trace.ref}
          accessToken={accessToken}
          onClose={() => setTrace(null)}
        />
      )}
    </div>
  );
}
