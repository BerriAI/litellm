"use client";

import { useTerminology } from "@/contexts/TerminologyContext";
import { SortingState } from "@tanstack/react-table";
import { KeyRound } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { CredentialItem } from "@/components/networking";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { DataTable, DataTableToolbar } from "@/components/shared/DataTable";
import { MultiSelect } from "@/components/shared/MultiSelect";

import { credentialProviderOptions, filterCredentials, type ProviderLabel } from "./credential_filters";
import { getCredentialsTableColumns } from "./CredentialsTableColumns";

interface CredentialsTableProps {
  credentials: CredentialItem[];
  canModifyCredentials: boolean;
  onEdit: (credential: CredentialItem) => void;
  onDelete: (credential: CredentialItem) => void;
  isLoading?: boolean;
}

const DEFAULT_SORTING: SortingState = [{ id: "credential_name", desc: false }];

const providerLabel: ProviderLabel = (provider) => getProviderLogoAndName(provider).displayName || provider;

function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  const { t } = useTranslation();
  const term = useTerminology();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <KeyRound className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {t(hasFilters ? "modelAdd.credentialsTable.noMatchesTitle" : "modelAdd.credentialsTable.emptyTitle")}
      </div>
      <div className="text-sm text-muted-foreground">
        {hasFilters
          ? t("modelAdd.credentialsTable.noMatchesDescription")
          : term(t("modelAdd.credentialsTable.emptyDescription"))}
      </div>
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
  const { t } = useTranslation();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);
  const [query, setQuery] = useState("");
  const [providers, setProviders] = useState<string[]>([]);

  const columns = useMemo(
    () => getCredentialsTableColumns({ canModifyCredentials, onEdit, onDelete }, t),
    [canModifyCredentials, onEdit, onDelete, t],
  );

  const providerOptions = useMemo(() => credentialProviderOptions(credentials, providerLabel), [credentials]);
  const visibleCredentials = useMemo(
    () => filterCredentials(credentials, { query, providers }, providerLabel),
    [credentials, query, providers],
  );
  const hasFilters = query.trim() !== "" || providers.length > 0;

  return (
    <DataTable
      data={visibleCredentials}
      paginationMode="client"
      columns={columns}
      getRowId={(credential, index) => credential.credential_name || String(index)}
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage={t("modelAdd.credentialsTable.loading")}
      noDataMessage={<EmptyState hasFilters={hasFilters} />}
      size="compact"
      toolbar={(table) => (
        <DataTableToolbar
          table={table}
          searchValue={query}
          onSearchChange={setQuery}
          searchPlaceholder={t("modelAdd.credentialsTable.searchPlaceholder")}
          showViewOptions={false}
        >
          <MultiSelect
            options={providerOptions}
            value={providers}
            onValueChange={setProviders}
            placeholder={t("modelAdd.credentialsTable.providerFilter")}
            emptyText={t("modelAdd.credentialsTable.providersEmpty")}
            className="w-64"
          />
        </DataTableToolbar>
      )}
    />
  );
};

export default CredentialsTable;
