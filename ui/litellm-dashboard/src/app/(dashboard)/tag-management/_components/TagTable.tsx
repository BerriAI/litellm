"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { SortingState, Table } from "@tanstack/react-table";
import { Inbox, SearchX } from "lucide-react";
import React, { useMemo, useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { Tag } from "@/components/tag_management/types";
import { Input } from "@/components/ui/input";

import { getTagTableColumns } from "./tagTableColumns";

interface TagTableProps {
  data: Tag[];
  onEdit: (tag: Tag) => void;
  onDelete: (tagName: string) => void;
  onSelectTag: (tagName: string) => void;
  isLoading?: boolean;
}

const DEFAULT_SORTING: SortingState = [{ id: "created_at", desc: true }];

function EmptyState() {
  const { t } = useLanguage();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("No tags yet")}</div>
      <div className="text-sm text-muted-foreground">
        {t("Create a tag to start routing and restricting model usage.")}
      </div>
    </div>
  );
}

function NoMatchingTags() {
  const { t } = useLanguage();
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <SearchX className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{t("No matching tags")}</div>
      <div className="text-sm text-muted-foreground">{t("Try a different tag name or description.")}</div>
    </div>
  );
}

function TagFilters({ table }: { table: Table<Tag> }) {
  const { t } = useLanguage();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        aria-label={t("Filter by tag name")}
        placeholder={t("Filter by tag name…")}
        className="h-8 w-56"
        value={(table.getColumn("name")?.getFilterValue() as string | undefined) ?? ""}
        onChange={(event) => table.getColumn("name")?.setFilterValue(event.target.value)}
      />
      <Input
        aria-label={t("Filter by description")}
        placeholder={t("Filter by description…")}
        className="h-8 w-56"
        value={(table.getColumn("description")?.getFilterValue() as string | undefined) ?? ""}
        onChange={(event) => table.getColumn("description")?.setFilterValue(event.target.value)}
      />
    </div>
  );
}

const TagTable: React.FC<TagTableProps> = ({ data, onEdit, onDelete, onSelectTag, isLoading = false }) => {
  const { t } = useLanguage();
  const [sorting, setSorting] = useState<SortingState>(DEFAULT_SORTING);

  const columns = useMemo(() => {
    const columnOptions = { onSelectTag, onEdit, onDelete, t };
    return getTagTableColumns(columnOptions);
  }, [onSelectTag, onEdit, onDelete, t]);

  return (
    <DataTable
      data={data}
      paginationMode="client"
      columns={columns}
      getRowId={(tag, index) => tag.name || String(index)}
      fillHeight
      sortingMode="client"
      sorting={sorting}
      onSortingChange={setSorting}
      isLoading={isLoading}
      loadingMessage={t("Loading tags…")}
      filterMode="client"
      toolbar={(table) => <TagFilters table={table} />}
      noDataMessage={data.length === 0 ? <EmptyState /> : <NoMatchingTags />}
      size="compact"
    />
  );
};

export default TagTable;
