"use client";

import { Building2, SearchX } from "lucide-react";
import React, { useMemo } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { Organization } from "@/components/networking";

import { getOrganizationsTableColumns } from "./OrganizationsTableColumns";
import { useOrganizationsTableState } from "./useOrganizationsTableState";

interface OrganizationsTableProps {
  organizations: Organization[];
  isLoading: boolean;
  userRole: string;
  searchActive: boolean;
  onOrganizationClick: (organizationId: string) => void;
  onEditClick: (organizationId: string) => void;
  onDeleteClick: (organizationId: string) => void;
}

function EmptyState({ searchActive }: { searchActive: boolean }) {
  const Icon = searchActive ? SearchX : Building2;
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Icon className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {searchActive ? "No matching organizations" : "No organizations yet"}
      </div>
      <div className="text-sm text-muted-foreground">
        {searchActive
          ? "No organizations match your search. Try a different name or ID."
          : "Create an organization to group teams, models, and budgets."}
      </div>
    </div>
  );
}

const OrganizationsTable: React.FC<OrganizationsTableProps> = ({
  organizations,
  isLoading,
  userRole,
  searchActive,
  onOrganizationClick,
  onEditClick,
  onDeleteClick,
}) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useOrganizationsTableState();

  const columns = useMemo(() => {
    const deps = { userRole, onOrganizationClick, onEditClick, onDeleteClick };
    return getOrganizationsTableColumns(deps);
  }, [userRole, onOrganizationClick, onEditClick, onDeleteClick]);

  return (
    <DataTable
      data={organizations}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(organization, index) => organization.organization_id || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage="Loading organizations…"
      noDataMessage={<EmptyState searchActive={searchActive} />}
      size="compact"
    />
  );
};

export default OrganizationsTable;
