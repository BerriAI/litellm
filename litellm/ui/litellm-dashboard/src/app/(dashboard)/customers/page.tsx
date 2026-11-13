"use client";

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import { queryOptions, skipToken, useQuery } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import UsageStatStrip from "@/app/(dashboard)/usage/_components/components/overview/UsageStatStrip";
import { overviewTotals } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { ErrorState } from "@/components/shared/ErrorState";
import CreateCustomerDialog from "./_components/CreateCustomerDialog";
import CustomerDetail from "./_components/CustomerDetail";
import CustomerOrganizationsTable from "./_components/CustomerOrganizationsTable";
import {
  CUSTOMER_ORGANIZATION_TABLE,
  customerOrganizationsCall,
  customerOrganizationsUsageCall,
  customerOrganizationRows,
  sortCustomerOrganizations,
} from "./_components/customerOrganizations";
import { mergeCustomerActivityPages } from "./_components/customerUsageData";
import {
  calendarDate,
  CUSTOMER_USAGE_MAX_RANGE_DAYS,
  customerPresetRange,
  customerUsagePresets,
  useCustomerUsageRange,
} from "./_components/customerUsageRange";

const EMPTY_USAGE = mergeCustomerActivityPages([]);

export default function CustomersPage() {
  const [selectedId, setSelectedId] = useQueryState("id", parseAsString);
  if (selectedId) return <CustomerDetail customerId={selectedId} onBack={() => void setSelectedId(null)} />;
  return <CustomerOrganizations />;
}

function CustomerOrganizations() {
  const { t } = useTranslation();
  const { accessToken, userId } = useAuthorized();
  const [createOpen, setCreateOpen] = useState(false);
  const { dateValue, onDateChange, preset } = useCustomerUsageRange();
  const { search, setSearch, sorting, onSortingChange, pagination, onPaginationChange } =
    useUrlTableState(CUSTOMER_ORGANIZATION_TABLE);
  const organizations = useQuery({
    queryKey: ["customer-organizations", userId],
    queryFn: accessToken ? ({ signal }) => customerOrganizationsCall(accessToken, signal) : skipToken,
  });
  const ids = useMemo(
    () => organizations.data?.map((organization) => organization.customer_org_id).sort(),
    [organizations.data],
  );
  const startDate = calendarDate(dateValue.from);
  const endDate = calendarDate(dateValue.to);
  const month = customerPresetRange("month");
  const monthStartDate = calendarDate(month.from);
  const monthEndDate = calendarDate(month.to);
  const usageRequest = (from: string, to: string) =>
    queryOptions({
      queryKey: ["customer-organizations", "usage", userId, ids, from, to],
      queryFn:
        accessToken && ids
          ? ({ signal }: { signal: AbortSignal }) => {
              const request = { accessToken, ids, startDate: from, endDate: to, signal };
              return customerOrganizationsUsageCall(request);
            }
          : skipToken,
    });
  const usage = useQuery(usageRequest(startDate, endDate));
  const monthlyUsage = useQuery(usageRequest(monthStartDate, monthEndDate));
  const rows = useMemo(
    () => customerOrganizationRows(organizations.data ?? [], monthlyUsage.data ?? EMPTY_USAGE),
    [organizations.data, monthlyUsage.data],
  );
  const sorted = useMemo(() => sortCustomerOrganizations(rows, search, sorting), [rows, search, sorting]);
  const lastPage = Math.max(0, Math.ceil(sorted.length / pagination.pageSize) - 1);
  const page = Math.min(pagination.pageIndex, lastPage);
  const start = page * pagination.pageSize;
  const loading = organizations.isPending || monthlyUsage.isPending;
  const activity = usage.data ?? EMPTY_USAGE;
  const totals = useMemo(() => overviewTotals(activity.metadata, null), [activity]);

  const onCreated = () => {
    setSearch("");
    onSortingChange([CUSTOMER_ORGANIZATION_TABLE.defaultSort]);
    void organizations.refetch();
  };

  return (
    <div className="w-full space-y-5 px-6 py-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h1 className="text-lg font-semibold tracking-tight">{t("customerPortal.list.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("customerPortal.list.subtitle")}</p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="size-4" /> {t("customerPortal.list.create")}
        </Button>
      </div>
      <section className="space-y-3" aria-label={t("customerPortal.list.overallUsage")}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">{t("customerPortal.list.overallUsage")}</h2>
            <p className="mt-1 text-xs text-muted-foreground">{t("customerPortal.list.overallUsageHelp")}</p>
          </div>
          <AdvancedDatePicker
            value={dateValue}
            onValueChange={onDateChange}
            presetShortLabel={preset}
            presets={customerUsagePresets}
            maxRangeDays={CUSTOMER_USAGE_MAX_RANGE_DAYS}
            label={t("customerPortal.range.label")}
            showTimeRange={false}
            compact
            className="max-w-sm"
          />
        </div>
        {organizations.isError || usage.isError ? (
          <ErrorState
            message={t("customerPortal.list.error")}
            onRetry={() => void (organizations.isError ? organizations.refetch() : usage.refetch())}
          />
        ) : (
          <UsageStatStrip
            results={activity.results}
            totals={totals}
            loading={organizations.isPending || usage.isPending}
            requestCountsPending={organizations.isPending || usage.isPending}
            budget={null}
            showBudgetHint={false}
          />
        )}
      </section>
      <div className="rounded-lg border bg-card">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b p-3">
          <Input
            className="max-w-xs"
            aria-label={t("customerPortal.list.searchPlaceholder")}
            placeholder={t("customerPortal.list.searchPlaceholder")}
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={sorting[0]?.id === "created_at" && sorting[0].desc}
            onClick={() => onSortingChange([CUSTOMER_ORGANIZATION_TABLE.defaultSort])}
          >
            {t("customerPortal.list.newestFirst")}
          </Button>
        </div>
        {organizations.isError || monthlyUsage.isError ? (
          <ErrorState
            message={t("customerPortal.list.error")}
            onRetry={() => void (organizations.isError ? organizations.refetch() : monthlyUsage.refetch())}
          />
        ) : (
          <CustomerOrganizationsTable
            rows={sorted.slice(start, start + pagination.pageSize)}
            sorting={sorting}
            onSort={onSortingChange}
            loading={loading}
            searching={Boolean(search)}
            startDate={startDate}
            endDate={endDate}
          />
        )}
        <DataTablePagination
          page={page}
          pageSize={pagination.pageSize}
          rowCount={sorted.length}
          onPageChange={(pageIndex) => onPaginationChange({ ...pagination, pageIndex })}
          onPageSizeChange={(pageSize) => onPaginationChange({ pageSize, pageIndex: 0 })}
          isLoading={organizations.isFetching || monthlyUsage.isFetching}
        />
      </div>
      <p className="text-xs text-muted-foreground">{t("customerPortal.list.budgetAndShareHelp")}</p>
      <p className="text-xs text-muted-foreground">{t("customerPortal.list.attribution")}</p>
      <CreateCustomerDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={onCreated} />
    </div>
  );
}
