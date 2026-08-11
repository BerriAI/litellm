"use client";

import { Inbox, ShieldAlert } from "lucide-react";
import React, { useCallback, useMemo, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import {
  BUDGET_DURATION_FILTER_OPTIONS,
  BUDGET_DURATION_UNSET,
  type CreatedAtFilterValue,
  type MaxBudgetFilterValue,
} from "@/app/(dashboard)/hooks/budgets/budgetFilters";
import type { budgetItem } from "@/app/(dashboard)/hooks/budgets/useBudgets";
import type { ResourceListResult } from "@/app/(dashboard)/hooks/common/useResourceList";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableToolbar,
  type FilterDraft,
} from "@/components/shared/DataTable";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError } from "@/lib/http/client";

import { BUDGET_TABLE_HIDDEN_COLUMNS, getBudgetTableColumns } from "./BudgetTableColumns";

interface BudgetTableProps {
  list: ResourceListResult<budgetItem>;
  canModify: boolean;
  onEditClick: (budget: budgetItem) => void;
  onDeleteClick: (budget: budgetItem) => void;
}

const PAGE_SIZE_OPTIONS = [25, 50, 100];

const durationLabel = (value: string, t: TFunction<"gateway">): string => {
  const keys: Record<string, string> = {
    "1h": "hourly",
    "24h": "daily",
    "7d": "weekly",
    "30d": "monthly",
    [BUDGET_DURATION_UNSET]: "notSet",
  };
  const key = keys[value];
  return key ? t(`budgets.duration.${key}`) : value;
};

const formatFilterValue = (columnId: string, value: unknown, t: TFunction<"gateway">): string => {
  if (columnId === "budget_duration") {
    return (Array.isArray(value) ? value : []).map((entry) => durationLabel(String(entry), t)).join(", ");
  }
  if (columnId === "max_budget") {
    const { min, max, unlimitedOnly } = (value ?? {}) as MaxBudgetFilterValue;
    return unlimitedOnly === true
      ? t("budgets.table.unlimitedOnly")
      : t("budgets.table.range", {
          min: min ? `$${min}` : t("budgets.table.any"),
          max: max ? `$${max}` : t("budgets.table.any"),
        });
  }
  if (columnId === "created_at") {
    const { from, to } = (value ?? {}) as CreatedAtFilterValue;
    return t("budgets.table.range", {
      min: from || t("budgets.table.any"),
      max: to || t("budgets.table.any"),
    });
  }
  return String(value);
};

/** The drawer keeps any non-empty object as an active filter, so collapse a blank draft to nothing. */
const normalizeMaxBudget = (draft: MaxBudgetFilterValue): MaxBudgetFilterValue | undefined => {
  if (draft.unlimitedOnly === true) {
    return { unlimitedOnly: true };
  }
  const min = draft.min?.trim() ?? "";
  const max = draft.max?.trim() ?? "";
  if (min === "" && max === "") {
    return undefined;
  }
  return { ...(min === "" ? {} : { min }), ...(max === "" ? {} : { max }) };
};

const normalizeCreatedAt = (draft: CreatedAtFilterValue): CreatedAtFilterValue | undefined => {
  const from = draft.from ?? "";
  const to = draft.to ?? "";
  if (from === "" && to === "") {
    return undefined;
  }
  return { ...(from === "" ? {} : { from }), ...(to === "" ? {} : { to }) };
};

function EmptyState({ hasQuery }: { hasQuery: boolean }) {
  const { t } = useTranslation("gateway");
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {hasQuery ? t("budgets.table.noMatches") : t("budgets.table.empty")}
      </div>
      <div className="text-sm text-muted-foreground">
        {hasQuery ? t("budgets.table.noMatchesDescription") : t("budgets.table.emptyDescription")}
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: Error }) {
  const { t } = useTranslation("gateway");
  const forbidden = error instanceof ApiError && error.status === 403;
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <ShieldAlert className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">
        {forbidden ? t("budgets.table.forbidden") : t("budgets.table.loadError")}
      </div>
      <div className="text-sm text-muted-foreground">
        {forbidden ? t("budgets.table.forbiddenDescription") : error.message}
      </div>
    </div>
  );
}

/** "Not set" and the concrete durations are exclusive; see serializeBudgetFilters for why. */
function DurationFilter({
  selected,
  onChange,
  t,
}: {
  selected: string[];
  onChange: (selected: string[]) => void;
  t: TFunction<"gateway">;
}) {
  const toggle = (value: string, checked: boolean): void => {
    if (!checked) {
      onChange(selected.filter((entry) => entry !== value));
      return;
    }
    const kept = value === BUDGET_DURATION_UNSET ? [] : selected.filter((entry) => entry !== BUDGET_DURATION_UNSET);
    onChange([...kept, value]);
  };

  return (
    <div className="flex flex-col gap-2">
      {BUDGET_DURATION_FILTER_OPTIONS.map((option) => (
        <Label key={option.value} className="font-normal">
          <Checkbox
            checked={selected.includes(option.value)}
            onCheckedChange={(checked) => toggle(option.value, checked === true)}
            data-testid={`budget-filter-duration-${option.value}`}
          />
          {durationLabel(option.value, t)}
        </Label>
      ))}
    </div>
  );
}

function BudgetFilterFields({ get, set, t }: FilterDraft & { t: TFunction<"gateway"> }) {
  const maxBudget = (get("max_budget") as MaxBudgetFilterValue | undefined) ?? {};
  const created = (get("created_at") as CreatedAtFilterValue | undefined) ?? {};
  const unlimitedOnly = maxBudget.unlimitedOnly === true;

  return (
    <>
      <DataTableFilterField label={t("budgets.fields.reset")}>
        <DurationFilter
          selected={(get("budget_duration") as string[] | undefined) ?? []}
          onChange={(selected) => set("budget_duration", selected)}
          t={t}
        />
      </DataTableFilterField>
      <DataTableFilterField label={t("budgets.fields.maxBudgetUsd")}>
        <div className="flex items-center gap-2">
          <Input
            type="number"
            min={0}
            step="0.01"
            value={maxBudget.min ?? ""}
            disabled={unlimitedOnly}
            onChange={(event) => set("max_budget", normalizeMaxBudget({ ...maxBudget, min: event.target.value }))}
            placeholder={t("budgets.table.min")}
            aria-label={t("budgets.table.minimumMaxBudget")}
            data-testid="budget-filter-max-budget-min"
          />
          <Input
            type="number"
            min={0}
            step="0.01"
            value={maxBudget.max ?? ""}
            disabled={unlimitedOnly}
            onChange={(event) => set("max_budget", normalizeMaxBudget({ ...maxBudget, max: event.target.value }))}
            placeholder={t("budgets.table.max")}
            aria-label={t("budgets.table.maximumMaxBudget")}
            data-testid="budget-filter-max-budget-max"
          />
        </div>
        <Label className="mt-1 font-normal">
          <Checkbox
            checked={unlimitedOnly}
            onCheckedChange={(checked) => set("max_budget", normalizeMaxBudget({ unlimitedOnly: checked === true }))}
            data-testid="budget-filter-max-budget-unlimited"
          />
          {t("budgets.table.unlimitedOnly")}
        </Label>
      </DataTableFilterField>
      <DataTableFilterField label={t("budgets.fields.created")}>
        <div className="flex items-center gap-2">
          <Input
            type="date"
            value={created.from ?? ""}
            onChange={(event) => set("created_at", normalizeCreatedAt({ ...created, from: event.target.value }))}
            aria-label={t("budgets.table.createdFrom")}
            data-testid="budget-filter-created-from"
          />
          <Input
            type="date"
            value={created.to ?? ""}
            onChange={(event) => set("created_at", normalizeCreatedAt({ ...created, to: event.target.value }))}
            aria-label={t("budgets.table.createdTo")}
            data-testid="budget-filter-created-to"
          />
        </div>
      </DataTableFilterField>
    </>
  );
}

const BudgetTable: React.FC<BudgetTableProps> = ({ list, canModify, onEditClick, onDeleteClick }) => {
  const { t } = useTranslation("gateway");
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columns = useMemo(
    () => getBudgetTableColumns({ canModify, onEditClick, onDeleteClick, t }),
    [canModify, onEditClick, onDeleteClick, t],
  );
  const filterLabels = useMemo<Record<string, string>>(
    () => ({
      budget_duration: t("budgets.fields.reset"),
      max_budget: t("budgets.fields.maxBudget"),
      created_at: t("budgets.fields.created"),
    }),
    [t],
  );
  const renderFilterValue = useCallback(
    (columnId: string, value: unknown) => formatFilterValue(columnId, value, t),
    [t],
  );

  const hasQuery = list.searchValue.trim() !== "" || list.columnFilters.length > 0;
  const emptyMessage = list.error === null ? <EmptyState hasQuery={hasQuery} /> : <ErrorState error={list.error} />;

  return (
    <DataTable
      data={list.rows}
      columns={columns}
      getRowId={(budget, index) => budget.budget_id || String(index)}
      defaultColumnVisibility={BUDGET_TABLE_HIDDEN_COLUMNS}
      fillHeight
      sortingMode="server"
      sorting={list.sorting}
      onSortingChange={list.onSortingChange}
      paginationMode="server"
      pagination={list.pagination}
      onPaginationChange={list.onPaginationChange}
      rowCount={list.rowCount}
      pageSizeOptions={PAGE_SIZE_OPTIONS}
      filterMode="server"
      columnFilters={list.columnFilters}
      onColumnFiltersChange={list.onColumnFiltersChange}
      isLoading={list.isLoading}
      loadingMessage={t("budgets.table.loading")}
      noDataMessage={emptyMessage}
      size="compact"
      toolbar={(table) => (
        <>
          <DataTableToolbar
            table={table}
            searchValue={list.searchValue}
            onSearchChange={list.onSearchChange}
            searchPlaceholder={t("budgets.table.search")}
            onOpenFilters={() => setFiltersOpen(true)}
            onRefresh={list.refetch}
            isRefreshing={list.isFetching}
            filterLabels={filterLabels}
            formatFilterValue={renderFilterValue}
          />
          <DataTableFilterDrawer
            table={table}
            open={filtersOpen}
            onOpenChange={setFiltersOpen}
            title={t("budgets.table.filters")}
            description={t("budgets.table.filtersDescription")}
          >
            {(draft) => <BudgetFilterFields {...draft} t={t} />}
          </DataTableFilterDrawer>
        </>
      )}
    />
  );
};

export default BudgetTable;
