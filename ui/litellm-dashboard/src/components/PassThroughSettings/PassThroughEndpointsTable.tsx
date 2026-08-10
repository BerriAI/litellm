"use client";

import { Waypoints } from "lucide-react";
import { useMemo } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import { DataTable } from "@/components/shared/DataTable";

import { getPassThroughEndpointsTableColumns } from "./PassThroughEndpointsTableColumns";
import type { passThroughItem } from "./PassThroughSettings";

interface PassThroughEndpointsTableProps {
  endpoints: passThroughItem[];
  isLoading: boolean;
  onEndpointClick: (endpointId: string) => void;
  onDeleteClick: (endpointId: string) => void;
}

function EmptyState({ t }: { t: TFunction<"gateway"> }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Waypoints className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("models.passThrough.emptyTitle")}</div>
      <div className="text-sm text-muted-foreground">{t("models.passThrough.emptyDescription")}</div>
    </div>
  );
}

export function PassThroughEndpointsTable({
  endpoints,
  isLoading,
  onEndpointClick,
  onDeleteClick,
}: PassThroughEndpointsTableProps) {
  const { t } = useTranslation("gateway");
  const columns = useMemo(
    () => getPassThroughEndpointsTableColumns({ onEndpointClick, onDeleteClick, t }),
    [onEndpointClick, onDeleteClick, t],
  );

  return (
    <DataTable
      data={endpoints}
      columns={columns}
      getRowId={(endpoint, index) => endpoint.id || endpoint.path || String(index)}
      isLoading={isLoading}
      loadingMessage={t("models.passThrough.loading")}
      noDataMessage={<EmptyState t={t} />}
      size="compact"
    />
  );
}
