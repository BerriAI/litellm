"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { Waypoints } from "lucide-react";
import { useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";

import { getPassThroughEndpointsTableColumns } from "./PassThroughEndpointsTableColumns";
import type { passThroughItem } from "./PassThroughSettings";

interface PassThroughEndpointsTableProps {
  endpoints: passThroughItem[];
  isLoading: boolean;
  onEndpointClick: (endpointId: string) => void;
  onDeleteClick: (endpointId: string) => void;
}

function EmptyState() {
  const { t } = useLanguage();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Waypoints className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("No pass-through endpoints configured")}</div>
      <div className="text-sm text-muted-foreground">{t("Add a pass-through endpoint to route custom paths.")}</div>
    </div>
  );
}

export function PassThroughEndpointsTable({
  endpoints,
  isLoading,
  onEndpointClick,
  onDeleteClick,
}: PassThroughEndpointsTableProps) {
  const { t } = useLanguage();
  const columns = useMemo(
    () => getPassThroughEndpointsTableColumns({ ...{ onEndpointClick, onDeleteClick }, t }),
    [onEndpointClick, onDeleteClick, t],
  );

  return (
    <DataTable
      data={endpoints}
      paginationMode="client"
      columns={columns}
      getRowId={(endpoint, index) => endpoint.id || endpoint.path || String(index)}
      isLoading={isLoading}
      loadingMessage={t("Loading pass-through endpoints…")}
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
}
