"use client";

import { Waypoints } from "lucide-react";
import { useMemo } from "react";

import {
  DataTable,
  DEFAULT_PAGE_SIZE_OPTIONS,
  useUrlTableState,
  type UrlTableStateOptions,
} from "@/components/shared/DataTable";

import { getPassThroughEndpointsTableColumns } from "./PassThroughEndpointsTableColumns";
import type { passThroughItem } from "./PassThroughSettings";

interface PassThroughEndpointsTableProps {
  endpoints: passThroughItem[];
  isLoading: boolean;
  onEndpointClick: (endpointId: string) => void;
  onDeleteClick: (endpointId: string) => void;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: [],
  defaultSort: { id: "", desc: false },
  defaultPageSize: DEFAULT_PAGE_SIZE_OPTIONS[0],
  filterColumns: [],
  keyPrefix: "pass_through_",
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Waypoints className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No pass-through endpoints configured</div>
      <div className="text-sm text-muted-foreground">Add a pass-through endpoint to route custom paths.</div>
    </div>
  );
}

export function PassThroughEndpointsTable({
  endpoints,
  isLoading,
  onEndpointClick,
  onDeleteClick,
}: PassThroughEndpointsTableProps) {
  const { pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);
  const columns = useMemo(
    () => getPassThroughEndpointsTableColumns({ onEndpointClick, onDeleteClick }),
    [onEndpointClick, onDeleteClick],
  );

  return (
    <DataTable
      data={endpoints}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(endpoint, index) => endpoint.id || endpoint.path || String(index)}
      isLoading={isLoading}
      loadingMessage="Loading pass-through endpoints…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
