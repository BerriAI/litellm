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
import { customerUsageTotals } from "./_components/customerUsageTotals";
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
  const loading = organizations.isPending || monthlyUsage.isPending || usage.isPending;
  const activity = usage.data ?? EMPTY_USAGE;
  const totals = useMemo(() => overviewTotals(activity.metadata, null), [activity]);
  const usageTotals = useMemo(() => (usage.data ? customerUsageTotals(usage.data.results) : null), [usage.data]);

  const onCreated = () => {
    setSearch("");
    onSortingChange([CUSTOMER_ORGANIZATION_TABLE.defaultSort]);
    void organizations.refetch();
  };

  return (
    <div className="mx-auto w-full max-w-[1920px] space-y-6 px-4 py-6 sm:px-6">
      <section className="space-y-3" aria-label={t("customerPortal.list.overallUsage")}>
        <div className="flex flex-wrap items-end justify-between gap-3">
          <h2 className="pb-2.5 text-sm font-semibold">{t("customerPortal.list.overallUsage")}</h2>
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
          <Button className="ml-auto" onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" /> {t("customerPortal.list.create")}
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
            usageTotals={usageTotals}
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
      <CreateCustomerDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={onCreated} />
    </div>
  );
}
