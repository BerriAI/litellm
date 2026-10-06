"use client";

import { Fragment, useId, useState, type ReactNode } from "react";
import { ArrowLeft, ArrowUpRight, Download, Loader2, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/cva.config";
import { StateMessage } from "../ui/StateMessage";
import { useDatasetRoute, useOpenSourceTrace } from "../route";
import { isRevisionConflict, useDataset, useExportDataset, useSaveRevision } from "./api";
import type { Dataset, DatasetCase, DatasetToolCall } from "./types";

interface CaseEdit {
  readonly included?: boolean;
  readonly expected?: string;
}

type Edits = Readonly<Record<string, CaseEdit>>;

const editedCase = (item: DatasetCase, edit: CaseEdit | undefined): DatasetCase =>
  edit ? { ...item, included: edit.included ?? item.included, expected: edit.expected ?? item.expected } : item;

const isChanged = (item: DatasetCase, edit: CaseEdit | undefined): boolean => {
  const next = editedCase(item, edit);
  return next.included !== item.included || next.expected !== item.expected;
};

export interface DatasetDetailProps {
  readonly datasetId: string;
  readonly readOnly: boolean;
  readonly onBack: () => void;
}

export function DatasetDetail({ datasetId, readOnly, onBack }: DatasetDetailProps) {
  const { revision, setRevision } = useDatasetRoute();
  const latest = useDataset(datasetId);
  const viewed = useDataset(datasetId, revision);
  if (latest.isPending || viewed.isPending)
    return (
      <StateMessage
        role="status"
        icon={<Loader2 className="size-5 animate-spin motion-reduce:animate-none" />}
        title="Loading dataset…"
        description="Fetching its cases."
      />
    );
  const error = latest.error ?? viewed.error;
  if (error || !latest.data || !viewed.data)
    return (
      <StateMessage
        role="alert"
        tone="destructive"
        icon={<TriangleAlert className="size-5" />}
        title="Couldn't load this dataset"
        description={error?.message ?? "The proxy returned no dataset."}
      >
        <Button size="sm" variant="outline" onClick={onBack}>
          All datasets
        </Button>
      </StateMessage>
    );
  return (
    <DatasetRevision
      key={viewed.data.revision}
      dataset={viewed.data}
      latestRevision={latest.data.revision}
      readOnly={readOnly}
      onBack={onBack}
      onPickRevision={(next) => setRevision(next === latest.data.revision ? null : next)}
      onShowLatest={() => setRevision(null)}
      onReload={() => {
        setRevision(null);
        void latest.refetch();
      }}
    />
  );
}

interface DatasetRevisionProps {
  readonly dataset: Dataset;
  readonly latestRevision: number;
  readonly readOnly: boolean;
  readonly onBack: () => void;
  readonly onPickRevision: (revision: number) => void;
  readonly onShowLatest: () => void;
  readonly onReload: () => void;
}

function DatasetRevision(props: DatasetRevisionProps) {
  const { dataset, latestRevision, readOnly, onBack, onPickRevision, onShowLatest, onReload } = props;
  const [edits, setEdits] = useState<Edits>({});
  const save = useSaveRevision();
  const exportDataset = useExportDataset();
  const isLatest = dataset.revision === latestRevision;
  const editable = isLatest && !readOnly;
  const changed = dataset.cases.filter((item) => isChanged(item, edits[item.id])).length;
  const included = dataset.cases.filter((item) => editedCase(item, edits[item.id]).included).length;
  const edit = (id: string, next: CaseEdit) =>
    setEdits((current) => ({ ...current, [id]: { ...current[id], ...next } }));
  const saveRevision = () =>
    save.mutate(
      {
        datasetId: dataset.id,
        body: { base_revision: dataset.revision, cases: dataset.cases.map((item) => editedCase(item, edits[item.id])) },
      },
      { onSuccess: onShowLatest },
    );
  const reload = () => {
    save.reset();
    setEdits({});
    onReload();
  };
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b px-3 py-2 sm:px-4">
        <Button variant="ghost" size="icon" className="size-7" aria-label="All datasets" onClick={onBack}>
          <ArrowLeft className="size-4" />
        </Button>
        <div className="flex min-w-0 flex-1 flex-col">
          <h2 className="truncate text-sm font-semibold">{dataset.name}</h2>
          <p className="truncate text-xs text-muted-foreground">
            {dataset.agent_name || "Any agent"} · {included} of {dataset.cases.length} included
          </p>
        </div>
        <RevisionPicker revision={dataset.revision} latest={latestRevision} onPick={onPickRevision} />
        <Button
          size="sm"
          variant="outline"
          disabled={exportDataset.isPending}
          onClick={() => exportDataset.mutate({ id: dataset.id, name: dataset.name, revision: dataset.revision })}
        >
          <Download className="size-3.5" />
          Export JSONL
        </Button>
        {editable && (
          <Button size="sm" disabled={changed === 0 || save.isPending} onClick={saveRevision}>
            {save.isPending && <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none" />}
            Save as revision {dataset.revision + 1}
          </Button>
        )}
      </header>
      <SaveProblem error={save.error ?? exportDataset.error} onReload={reload} />
      {!isLatest && (
        <p role="status" className="border-b px-4 py-2 text-xs text-muted-foreground">
          Revision {dataset.revision} is read-only. Switch to the latest revision to make changes.
        </p>
      )}
      {dataset.cases.length === 0 ? (
        <p className="px-4 py-10 text-center text-sm text-muted-foreground">
          This revision has no cases. Add some from a trace or a finding.
        </p>
      ) : (
        <ol aria-label="Cases" className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3 sm:p-4">
          {dataset.cases.map((item, index) => (
            <CaseCard
              key={item.id}
              item={editedCase(item, edits[item.id])}
              index={index}
              editable={editable}
              onEdit={(next) => edit(item.id, next)}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

function SaveProblem({ error, onReload }: { error: Error | null; onReload: () => void }) {
  if (!error) return null;
  const conflict = isRevisionConflict(error);
  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-3 border-b border-destructive/30 bg-destructive/5 px-4 py-2 text-sm text-destructive"
    >
      <span className="min-w-0">
        {conflict
          ? "Someone saved a newer revision while you were editing. Reload to see it, then make your changes again."
          : error.message}
      </span>
      {conflict && (
        <Button variant="ghost" size="sm" onClick={onReload}>
          Reload
        </Button>
      )}
    </div>
  );
}

function RevisionPicker({
  revision,
  latest,
  onPick,
}: {
  revision: number;
  latest: number;
  onPick: (revision: number) => void;
}) {
  const revisions = Array.from({ length: latest + 1 }, (_, offset) => latest - offset);
  return (
    <select
      aria-label="Revision"
      className="h-8 rounded-md border-0 bg-transparent pr-7 pl-2 text-xs text-muted-foreground hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
      value={revision}
      onChange={(event) => onPick(Number(event.target.value))}
    >
      {revisions.map((value) => (
        <option key={value} value={value}>
          Revision {value}
          {value === latest ? " (latest)" : ""}
        </option>
      ))}
    </select>
  );
}

interface CaseCardProps {
  readonly item: DatasetCase;
  readonly index: number;
  readonly editable: boolean;
  readonly onEdit: (edit: CaseEdit) => void;
}

function CaseCard({ item, index, editable, onEdit }: CaseCardProps) {
  const expectedId = useId();
  const label = `Case ${index + 1}`;
  const openSourceTrace = useOpenSourceTrace();
  const { trace_id: traceId, trace_ref: traceRef, span_id: spanId } = item.source;
  return (
    <li
      aria-label={label}
      className={cn(
        "flex flex-col gap-3 rounded-xl bg-muted/40 p-3 transition-opacity",
        !item.included && "opacity-60",
      )}
    >
      <div className="flex items-center gap-2">
        <label className="flex items-center gap-2 text-sm font-medium">
          <input
            type="checkbox"
            aria-label={`Include ${label}`}
            className="size-4 rounded border-input accent-foreground disabled:opacity-50"
            checked={item.included}
            disabled={!editable}
            onChange={(event) => onEdit({ included: event.target.checked })}
          />
          {label}
        </label>
        {!item.included && <span className="text-xs text-muted-foreground">excluded</span>}
        {traceId && (
          <button
            type="button"
            className="ml-auto inline-flex items-center gap-0.5 text-xs text-muted-foreground hover:text-foreground"
            onClick={() => openSourceTrace({ traceId, traceRef, spanId })}
          >
            Source trace
            <ArrowUpRight aria-hidden="true" className="size-3" />
          </button>
        )}
      </div>
      <div className="flex max-h-72 flex-col gap-2 overflow-y-auto">
        {item.messages.map((message, position) => (
          <Fragment key={position}>
            {message.content && (
              <Turn role={message.name ? `${message.role} · ${message.name}` : message.role}>{message.content}</Turn>
            )}
            <ToolCalls calls={message.tool_calls} />
          </Fragment>
        ))}
        {item.reply && <Turn role="reply">{item.reply}</Turn>}
        <ToolCalls calls={item.tool_calls} />
      </div>
      <div className="flex flex-col gap-1">
        <label htmlFor={expectedId} className="text-xs font-medium text-muted-foreground">
          Expected
        </label>
        {editable ? (
          <Textarea
            id={expectedId}
            value={item.expected}
            placeholder="What a good reply looks like"
            className="bg-background text-sm"
            onChange={(event) => onEdit({ expected: event.target.value })}
          />
        ) : (
          <p id={expectedId} className="text-sm whitespace-pre-wrap text-foreground">
            {item.expected || <span className="text-muted-foreground">Not set</span>}
          </p>
        )}
      </div>
    </li>
  );
}

function ToolCalls({ calls }: { calls: readonly DatasetToolCall[] }) {
  return calls.map((call, position) => (
    <Turn key={position} role={`tool call · ${call.name}`}>
      <code className="text-xs break-all">{call.arguments}</code>
    </Turn>
  ));
}

function Turn({ role, children }: { role: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs font-medium text-muted-foreground">{role}</span>
      <div className="text-sm whitespace-pre-wrap break-words text-foreground">{children}</div>
    </div>
  );
}
