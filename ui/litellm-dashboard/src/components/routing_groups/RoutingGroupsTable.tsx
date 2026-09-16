"use client";

import { functionalUpdate, type ExpandedState, type OnChangeFn, type SortingState } from "@tanstack/react-table";
import { Inbox } from "lucide-react";
import { parseAsStringLiteral, useQueryStates } from "nuqs";
import React, { useCallback, useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";

import { RoutingGroupUsagePanel } from "./RoutingGroupUsagePanel";
import { getRoutingGroupsTableColumns } from "./RoutingGroupsTableColumns";
import type { RoutingGroup } from "./types";
import { useExpandedRoutingGroups } from "./useExpandedRoutingGroups";

interface RoutingGroupsTableProps {
  groups: RoutingGroup[];
  isLoading?: boolean;
  onEdit: (group: RoutingGroup) => void;
  onDelete: (group: RoutingGroup) => void;
  proxyBaseUrl?: string;
}

const SORTABLE_COLUMNS = ["group_name", "routing_strategy"] as const;

const SORT_URL_STATE = {
  sort_by: parseAsStringLiteral(SORTABLE_COLUMNS),
  sort_order: parseAsStringLiteral(["asc", "desc"] as const).withDefault("asc"),
};

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
  const [{ sort_by: sortBy, sort_order: sortOrder }, setSort] = useQueryStates(SORT_URL_STATE);
  const [expandedGroups, setExpandedGroups] = useExpandedRoutingGroups();
  const baseUrl = resolveBaseUrl(proxyBaseUrl);

  const sorting = useMemo<SortingState>(
    () => (sortBy ? [{ id: sortBy, desc: sortOrder === "desc" }] : []),
    [sortBy, sortOrder],
  );

  const onSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updaterOrValue) => {
      const active = functionalUpdate(updaterOrValue, sorting)[0];
      const column = SORTABLE_COLUMNS.find((id) => id === active?.id) ?? null;
      void setSort({ sort_by: column, sort_order: column && active?.desc ? "desc" : null });
    },
    [setSort, sorting],
  );

  const expanded = useMemo<ExpandedState>(
    () => Object.fromEntries(expandedGroups.map((name) => [name, true])),
    [expandedGroups],
  );

  const onExpandedChange = useCallback<OnChangeFn<ExpandedState>>(
    (updaterOrValue) => {
      const next = functionalUpdate(updaterOrValue, expanded);
      void setExpandedGroups(
        next === true ? groups.map((group) => group.group_name) : Object.keys(next).filter((name) => next[name]),
      );
    },
    [expanded, groups, setExpandedGroups],
  );

  const toggleUsage = useCallback(
    (group: RoutingGroup) => {
      void setExpandedGroups((previous) =>
        previous.includes(group.group_name)
          ? previous.filter((name) => name !== group.group_name)
          : [...previous, group.group_name],
      );
    },
    [setExpandedGroups],
  );

  const columns = useMemo(() => {
    const deps = { onEdit, onDelete, onToggleUsage: toggleUsage };
    return getRoutingGroupsTableColumns(deps);
  }, [onEdit, onDelete, toggleUsage]);

  return (
    <DataTable
      data={groups}
      paginationMode="client"
      columns={columns}
      getRowId={(group) => group.group_name}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      expanded={expanded}
      onExpandedChange={onExpandedChange}
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
