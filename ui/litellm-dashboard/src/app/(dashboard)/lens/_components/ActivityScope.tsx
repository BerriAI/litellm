"use client";

import { useEffect, useId, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus, X, ChevronRight, RotateCw } from "lucide-react";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxEmpty,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { TracePanel } from "./TracePanel";
import { type Sample, type Settings, runTime, durationLabel } from "./lensData";

import { DurationInput } from "./DurationInput";

export type ActivitySelection = Pick<Settings, "source"> &
  Partial<
    Pick<
      Settings,
      | "service"
      | "agent_name"
      | "filters"
      | "lookback_hours"
      | "sample_percent"
      | "sample_size"
      | "team_id"
      | "execution_ids"
    >
  >;

const selectClass = "h-9 w-full rounded-md border border-input bg-background px-3 text-sm";

export function RunList({ executions }: { executions: Sample["executions"] }) {
  return (
    <div className="divide-y">
      {executions.map((run) => (
        <div key={run.id} className="py-3">
          <p className="text-sm font-medium">{run.name}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {runTime(run.start_time)} ·{" "}
            {run.source === "traces" ? `${run.span_count} ${run.span_count === 1 ? "step" : "steps"}` : "LLM request"}
          </p>
        </div>
      ))}
    </div>
  );
}

export function ActivityScope({
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
  const id = useId();
  const hasFilters = !!value.filters?.length || !!value.team_id;
  const [advanced, setAdvanced] = useState(hasFilters || !!value.service || value.source !== "traces");
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
  const load = (selection: ActivitySelection, pageOffset = 0) => {
    const { lookback_hours, ...selectionSettings } = selection;
    return apiClient.post<Sample>("/lens/preview/sample", {
      accessToken,
      body: {
        offset: pageOffset,
        as_of: asOf,
        settings: {
          ...selectionSettings,
          execution_ids: [],
          name: "Preview",
          model: "preview",

          checks: [{ id: "preview", instruction: "Preview recorded activity" }],
        },
        lookback_hours: lookback_hours ?? 24,
      },
    });
  };
  const discoveryScope: ActivitySelection = {
    source: value.source,
    service: "",
    filters: [],
    lookback_hours: value.lookback_hours,
  };
  const discoveryOptions = {
    queryKey: ["lens-activity-options", value.source, value.lookback_hours, asOf, accessToken],
    queryFn: () => load(discoveryScope),
    staleTime: 60000,
    enabled: validWindow,
  };
  const discovery = useQuery(discoveryOptions);
  const agentOptions = {
    queryKey: ["lens-agents", accessToken, asOf],
    queryFn: () => apiClient.get<string[]>("/lens/agents", { accessToken }),
    enabled: value.source !== "requests",
    staleTime: 60000,
  };
  const agents = useQuery(agentOptions);
  const previewOptions = {
    queryKey: ["lens-activity-preview", scope, offset, asOf, accessToken],
    queryFn: () => load(scope, offset),
    enabled: valid,
    staleTime: 30000,
  };
  const preview = useQuery(previewOptions);
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
  const filters = value.filters ?? [];
  const edit = (index: number, field: "key" | "value", text: string) =>
    onChange({ ...value, filters: filters.map((f, i) => (i === index ? { ...f, [field]: text } : f)) });

  const changeSource = (source: Settings["source"]) => {
    const selection = { ...value, source, service: "", agent_name: "", filters: [], execution_ids: [] };
    onChange(selection);
  };
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
          <>
            {nameField}
            <label className="grid gap-2 text-sm font-medium">
              {value.source === "requests" ? "Model group (optional)" : "Agent (optional)"}
              <Combobox
                items={names}
                value={selectedName || null}
                inputValue={selectedName ?? ""}
                onInputValueChange={selectName}
                onValueChange={(name) => selectName(name ?? "")}
              >
                <ComboboxInput
                  aria-label={value.source === "requests" ? "Model group (optional)" : "Agent (optional)"}
                  placeholder={value.source === "requests" ? "All model groups" : "All agents and activity"}
                  showClear={!!selectedName}
                  className="w-full h-9"
                />
                <ComboboxContent>
                  <ComboboxEmpty>
                    {agents.isFetching ? "Loading agents…" : "No matches. You can enter a recorded name."}
                  </ComboboxEmpty>
                  <ComboboxList>
                    {(name: string) => (
                      <ComboboxItem key={name} value={name}>
                        {name}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </label>
            {value.source !== "requests" && agents.isError && (
              <p role="alert" className="text-sm text-destructive">
                Could not load agents.{" "}
                <button type="button" className="underline" onClick={() => void agents.refetch()}>
                  Retry
                </button>
              </p>
            )}
            <details open={advanced} onToggle={(event) => setAdvanced(event.currentTarget.open)} className="group">
              <summary className="cursor-pointer text-sm font-medium">
                Advanced filters{filters.length ? ` (${filters.length})` : ""}
              </summary>
              <div className="mt-4 space-y-4">
                {value.source !== "requests" && (
                  <label className="grid gap-2 text-sm">
                    Application (optional)
                    <Input
                      value={value.service ?? ""}
                      placeholder="All applications"
                      onChange={(event) => onChange({ ...value, service: event.target.value, execution_ids: [] })}
                    />
                  </label>
                )}
                <label className="grid gap-2 text-sm">
                  Activity type
                  <select
                    className={selectClass}
                    value={value.source}
                    onChange={(e) => changeSource(e.target.value as Settings["source"])}
                  >
                    <option value="traces">Agent traces</option>
                    <option value="requests">LLM requests</option>
                    <option value="both">Traces and LLM requests</option>
                  </select>
                </label>
                <p className="text-xs leading-5 text-muted-foreground">
                  Match any recorded metadata, such as a user ID, environment, or tag. All conditions must match.
                </p>
                {filters.map((f, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex gap-2">
                      <Input
                        aria-label={`Metadata key ${index + 1}`}
                        list={`${id}-keys`}
                        placeholder="Metadata key"
                        value={f.key}
                        onChange={(e) => edit(index, "key", e.target.value)}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Remove condition ${index + 1}`}
                        onClick={() => onChange({ ...value, filters: filters.filter((_, i) => i !== index) })}
                      >
                        <X className="size-4" />
                      </Button>
                    </div>
                    <Input
                      aria-label={`Metadata value ${index + 1}`}
                      list={`${id}-values-${index}`}
                      placeholder="Equals"
                      value={f.value}
                      onChange={(e) => edit(index, "value", e.target.value)}
                    />
                    <datalist id={`${id}-values-${index}`}>
                      {[...new Set(attributes.filter((a) => a.key === f.key).map((a) => a.value))].sort().map((v) => (
                        <option key={v} value={v} />
                      ))}
                    </datalist>
                  </div>
                ))}
                <datalist id={`${id}-keys`}>
                  {keys.map((key) => (
                    <option key={key} value={key} />
                  ))}
                </datalist>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={filters.length >= 8}
                  onClick={() => onChange({ ...value, filters: [...filters, { key: "", value: "" }] })}
                >
                  <Plus className="size-3" /> Add condition
                </Button>
                <label className="grid gap-2 text-sm">
                  Team ID (optional)
                  <Input
                    value={value.team_id ?? ""}
                    placeholder="All accessible teams"
                    onChange={(e) => onChange({ ...value, team_id: e.target.value })}
                  />
                </label>
              </div>
            </details>
          </>
        ) : (
          <>
            <DurationInput
              label="Review the last"
              value={value.lookback_hours ?? 24}
              base="hours"
              max={8760}
              onChange={(lookback_hours) => onChange({ ...value, lookback_hours })}
            />
            <label className="grid gap-2 text-sm">
              Sample (%)
              <Input
                type="number"
                min="0.01"
                max="100"
                step="any"
                value={value.sample_percent ?? 100}
                onChange={(e) => onChange({ ...value, sample_percent: Number(e.target.value) })}
              />
            </label>
          </>
        )}
        {manualSelection && !!value.execution_ids?.length && (
          <Button variant="outline" size="sm" onClick={() => onChange({ ...value, execution_ids: [] })}>
            Clear {value.execution_ids.length} selected runs
          </Button>
        )}
      </div>
      {mode === "activity" && (
        <MatchingActivity
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
        <TracePanel
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

function MatchingActivity({
  offset,
  onPage,
  onSelect,
  selectedIds,
  manualSelection,
  selectedCount,
  title,
  windowLabel,
  ready,
  error,
  data,
  onOpen,
  onRetry,
}: {
  offset: number;
  onPage: (offset: number) => void;
  onSelect: (id: string, checked: boolean) => void;
  selectedIds: string[];
  manualSelection: boolean;
  selectedCount: number;
  title: string;
  windowLabel: string;
  ready: boolean;
  error: Error | null;
  data: Sample | undefined;
  onRetry: () => void;
  onOpen: (run: Sample["executions"][number]) => void;
}) {
  const paginated = data?.next_offset != null || offset > 0;
  const showSelection = selectedCount !== data?.eligible || paginated;
  const selectionData = ready && showSelection ? data : undefined;
  return (
    <section aria-label="Matching activity" className="self-start rounded-lg border">
      <div className="border-b px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium" role="status">
            {title}
          </p>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Refresh matching activity"
            onClick={onRetry}
            disabled={!ready}
          >
            <RotateCw className="size-3" />
          </Button>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{windowLabel} · No analysis cost</p>
      </div>
      <div className="max-h-64 overflow-y-auto px-4">
        {ready && error && (
          <p role="alert" className="py-3 text-sm text-destructive">
            {error.message}{" "}
            <Button variant="link" onClick={onRetry}>
              Retry preview
            </Button>
          </p>
        )}
        {ready && data?.eligible === 0 && (
          <p className="py-4 text-sm text-muted-foreground">
            No matches. Try removing a condition or check that your agent records this metadata. Recent trace updates
            need two minutes to settle.
          </p>
        )}
        {ready &&
          data?.executions.map((run) => (
            <div key={run.id} className="flex items-center justify-between gap-3 border-b last:border-0">
              {manualSelection && (
                <input
                  type="checkbox"
                  aria-label={`Select ${run.name}`}
                  checked={selectedIds.includes(run.id)}
                  onChange={(e) => onSelect(run.id, e.target.checked)}
                />
              )}
              <div className="min-w-0">
                <RunList executions={[run]} />
              </div>
              {run.source === "traces" && (
                <Button variant="ghost" size="icon-sm" aria-label={`Open ${run.name}`} onClick={() => onOpen(run)}>
                  <ChevronRight className="size-4" />
                </Button>
              )}
            </div>
          ))}
      </div>
      {selectionData && (
        <div className="border-t px-4 py-3 space-y-2">
          <p className="text-xs text-muted-foreground">
            {selectedCount} selected for analysis
            {paginated && (
              <>
                {" "}
                · Showing {offset + (selectionData.executions.length ? 1 : 0)}–
                {offset + selectionData.executions.length} of {selectionData.eligible}
              </>
            )}
          </p>
          {paginated && (
            <div className="flex justify-between">
              <Button
                size="sm"
                variant="ghost"
                disabled={offset === 0}
                onClick={() => onPage(Math.max(0, offset - 100))}
              >
                Previous
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={selectionData.next_offset == null}
                onClick={() => onPage(selectionData.next_offset ?? offset)}
              >
                Next
              </Button>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
