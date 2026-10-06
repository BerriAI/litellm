"use client";

import { Database, Loader2, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";
import { StateMessage } from "../ui/StateMessage";
import { useDatasetRoute } from "../route";
import { useDatasets } from "./api";
import { DatasetDetail } from "./DatasetDetail";
import type { DatasetSummary } from "./types";

export interface DatasetsViewProps {
  readonly readOnly?: boolean;
}

export function DatasetsView({ readOnly = false }: DatasetsViewProps) {
  const { datasetId, openDataset } = useDatasetRoute();
  return (
    <section aria-label="Datasets" className="flex w-full min-w-0 flex-1 flex-col">
      {datasetId ? (
        <DatasetDetail key={datasetId} datasetId={datasetId} readOnly={readOnly} onBack={() => openDataset(null)} />
      ) : (
        <DatasetList onOpen={openDataset} />
      )}
    </section>
  );
}

function DatasetList({ onOpen }: { onOpen: (id: string) => void }) {
  const datasets = useDatasets();
  if (datasets.isPending)
    return (
      <StateMessage
        role="status"
        icon={<Loader2 className="size-5 animate-spin motion-reduce:animate-none" />}
        title="Loading datasets…"
        description="Fetching the datasets you can see."
      />
    );
  if (datasets.isError)
    return (
      <StateMessage
        role="alert"
        tone="destructive"
        icon={<TriangleAlert className="size-5" />}
        title="Couldn't load datasets"
        description={datasets.error.message}
      >
        <Button size="sm" onClick={() => void datasets.refetch()}>
          Try again
        </Button>
      </StateMessage>
    );
  if (datasets.data.length === 0)
    return (
      <StateMessage
        role="status"
        icon={<Database className="size-5" />}
        title="No datasets yet"
        description="Open a trace or a finding and choose Add to dataset to save real conversations for an agent."
      />
    );
  return (
    <Table aria-label="Datasets">
      <TableHeader>
        <TableRow>
          <TableHead className="w-full">Name</TableHead>
          <TableHead className="hidden sm:table-cell">Agent</TableHead>
          <TableHead className="text-right">Revision</TableHead>
          <TableHead className="text-right">Cases</TableHead>
          <TableHead className="hidden text-right md:table-cell">Updated</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {datasets.data.map((dataset) => (
          <DatasetRow key={dataset.id} dataset={dataset} onOpen={onOpen} />
        ))}
      </TableBody>
    </Table>
  );
}

function DatasetRow({ dataset, onOpen }: { dataset: DatasetSummary; onOpen: (id: string) => void }) {
  return (
    <TableRow className="cursor-pointer" onClick={() => onOpen(dataset.id)}>
      <TableCell className="max-w-0 truncate">
        <button
          type="button"
          className="truncate font-medium text-foreground outline-none hover:underline focus-visible:underline"
          onClick={(event) => {
            event.stopPropagation();
            onOpen(dataset.id);
          }}
        >
          {dataset.name}
        </button>
      </TableCell>
      <TableCell className="hidden text-muted-foreground sm:table-cell">{dataset.agent_name || "Any agent"}</TableCell>
      <TableCell className="text-right tabular-nums text-muted-foreground">{dataset.revision}</TableCell>
      <TableCell className="text-right tabular-nums">{dataset.case_count}</TableCell>
      <TableCell className="hidden text-right text-muted-foreground md:table-cell">
        {formatActivityTimestamp(dataset.updated_at)}
      </TableCell>
    </TableRow>
  );
}
