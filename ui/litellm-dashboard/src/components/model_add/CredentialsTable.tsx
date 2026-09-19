"use client";

import { KeyRound } from "lucide-react";
import React, { useMemo } from "react";

import { CredentialItem } from "@/components/networking";
import {
  DataTable,
  DEFAULT_PAGE_SIZE_OPTIONS,
  useUrlTableState,
  type UrlTableStateOptions,
} from "@/components/shared/DataTable";

import { getCredentialsTableColumns } from "./CredentialsTableColumns";

interface CredentialsTableProps {
  credentials: CredentialItem[];
  canModifyCredentials: boolean;
  onEdit: (credential: CredentialItem) => void;
  onDelete: (credential: CredentialItem) => void;
  isLoading?: boolean;
}

const TABLE_STATE_OPTIONS: UrlTableStateOptions<never> = {
  sortFields: ["credential_name"],
  defaultSort: { id: "credential_name", desc: false },
  defaultPageSize: DEFAULT_PAGE_SIZE_OPTIONS[0],
  filterColumns: [],
  keyPrefix: "credentials_",
};

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <KeyRound className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No credentials configured</div>
      <div className="text-sm text-muted-foreground">Add a credential to connect an AI provider.</div>
    </div>
  );
}

const CredentialsTable: React.FC<CredentialsTableProps> = ({
  credentials,
  canModifyCredentials,
  onEdit,
  onDelete,
  isLoading = false,
}) => {
  const { sorting, onSortingChange, pagination, onPaginationChange } = useUrlTableState(TABLE_STATE_OPTIONS);

  const columns = useMemo(
    () => getCredentialsTableColumns({ canModifyCredentials, onEdit, onDelete }),
    [canModifyCredentials, onEdit, onDelete],
  );

  return (
    <DataTable
      data={credentials}
      paginationMode="client"
      pagination={pagination}
      onPaginationChange={onPaginationChange}
      columns={columns}
      getRowId={(credential, index) => credential.credential_name || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={onSortingChange}
      isLoading={isLoading}
      loadingMessage="Loading credentials…"
      noDataMessage={<EmptyState />}
      size="compact"
    />
  );
};

export default CredentialsTable;
