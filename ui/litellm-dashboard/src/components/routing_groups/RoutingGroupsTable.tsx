"use client";

import type { ExpandedState, SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import React, { useCallback, useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";

import { RoutingGroupUsagePanel } from "./RoutingGroupUsagePanel";
import { getRoutingGroupsTableColumns } from "./RoutingGroupsTableColumns";
import type { RoutingGroup } from "./types";

interface RoutingGroupsTableProps {
  groups: RoutingGroup[];
  isLoading?: boolean;
  onEdit: (group: RoutingGroup) => void;
  onDelete: (group: RoutingGroup) => void;
  proxyBaseUrl?: string;
}

const resolveBaseUrl = (proxyBaseUrl?: string): string => {
  if (proxyBaseUrl && proxyBaseUrl.trim()) return proxyBaseUrl;
  if (typeof window !== "undefined" && window.location?.origin) return window.location.origin;
  return "<your_proxy_base_url>";
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No routing groups yet</div>
      <div className="text-sm text-muted-foreground">
        Create a group to load-balance a set of models behind one name.
      </div>
    </div>
  );
}

const RoutingGroupsTable: React.FC<RoutingGroupsTableProps> = ({
  groups,
  isLoading,
  onEdit,
  onDelete,
  proxyBaseUrl,
}) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [expanded, setExpanded] = useState<ExpandedState>({});
  const baseUrl = resolveBaseUrl(proxyBaseUrl);

  const toggleUsage = useCallback((group: RoutingGroup) => {
    setExpanded((previous) => {
      const current = previous === true ? {} : previous;
      return { ...current, [group.group_name]: current[group.group_name] !== true };
    });
  }, []);

  const columns = useMemo(() => {
    const deps = { onEdit, onDelete, onToggleUsage: toggleUsage };
    return getRoutingGroupsTableColumns(deps);
  }, [onEdit, onDelete, toggleUsage]);

  return (
    <DataTable
      data={groups}
      columns={columns}
      getRowId={(group) => group.group_name}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      expanded={expanded}
      onExpandedChange={setExpanded}
      getRowCanExpand={() => true}
      renderSubComponent={({ row }) => <RoutingGroupUsagePanel group={row.original} baseUrl={baseUrl} />}
      isLoading={isLoading}
      loadingMessage="Loading routing groups…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default RoutingGroupsTable;
