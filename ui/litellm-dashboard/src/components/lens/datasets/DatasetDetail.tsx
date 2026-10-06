"use client";

import { useState } from "react";
import { ChevronRight, Database, Download, Loader2, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StateMessage } from "../ui/StateMessage";
import { useDatasetRoute } from "../route";
import { IdChip } from "../traces/ui/IdChip";
import { isRevisionConflict, useDataset, useExportDataset, useSaveRevision } from "./api";
import { CasePanel } from "./CasePanel";
import { CaseTable } from "./CaseTable";
import { changedCount, withEdits, type CaseEdit, type CaseEdits } from "./caseView";
import type { Dataset } from "./types";

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
  const { caseId, setCaseId } = useDatasetRoute();
  const [edits, setEdits] = useState<CaseEdits>({});
  const save = useSaveRevision();
  const exportDataset = useExportDataset();
  const isLatest = dataset.revision === latestRevision;
  const editable = isLatest && !readOnly;
  const cases = withEdits(dataset.cases, edits);
  const changed = changedCount(dataset.cases, edits);
  const edit = (id: string, next: CaseEdit) =>
    setEdits((current) => ({ ...current, [id]: { ...current[id], ...next } }));
  const saveRevision = () =>
    save.mutate(
      { datasetId: dataset.id, body: { base_revision: dataset.revision, cases } },
      { onSuccess: onShowLatest },
    );
  const reload = () => {
    save.reset();
    setEdits({});
    onReload();
  };
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex shrink-0 flex-col gap-1 border-b bg-background px-3 pt-2.5 pb-2 sm:px-4">
        <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-xs text-muted-foreground">
          <button type="button" onClick={onBack} className="rounded hover:text-foreground hover:underline">
            Datasets
          </button>
          <ChevronRight aria-hidden="true" className="size-3" />
          <span className="truncate">{dataset.name}</span>
        </nav>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <Database aria-hidden="true" className="size-4 shrink-0 text-muted-foreground" />
            <h2 className="min-w-0 truncate text-base font-semibold">{dataset.name}</h2>
            <IdChip value={dataset.id} label="Copy dataset ID" />
            <span className="hidden truncate text-xs text-muted-foreground sm:inline">
              {dataset.agent_name || "Any agent"}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
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
          </div>
        </div>
      </header>
      <SaveProblem error={save.error ?? exportDataset.error} onReload={reload} />
      {!isLatest && (
        <p role="status" className="border-b px-4 py-2 text-xs text-muted-foreground">
          Revision {dataset.revision} is read-only. Switch to the latest revision to make changes.
        </p>
      )}
      {cases.length === 0 ? (
        <p className="px-4 py-10 text-center text-sm text-muted-foreground">
          This revision has no cases. Add some from a trace or a finding.
        </p>
      ) : (
        <CaseTable
          cases={cases}
          editable={editable}
          selectedId={caseId}
          onSelect={setCaseId}
          onToggle={(id, included) => edit(id, { included })}
        >
          {(item) => (
            <CasePanel
              key={item.id}
              item={item}
              datasetName={dataset.name}
              editable={editable}
              onEdit={(next) => edit(item.id, next)}
            />
          )}
        </CaseTable>
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
