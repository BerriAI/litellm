"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { ErrorState } from "@/components/shared/ErrorState";
import CreateCustomerDialog from "./_components/CreateCustomerDialog";
import CustomerDetail from "./_components/CustomerDetail";
import { calendarDate, customerUsagePresets, useCustomerUsageRange } from "./_components/customerUsageRange";

const TABLE_OPTIONS = {
  sortFields: ["spend"],
  defaultSort: { id: "spend", desc: true },
  defaultPageSize: 25,
  filterColumns: [],
};

export default function CustomersPage() {
  const [selectedId, setSelectedId] = useQueryState("id", parseAsString);
  if (selectedId) return <CustomerDetail customerId={selectedId} onBack={() => void setSelectedId(null)} />;
  return <CustomerOrganizations />;
}

function CustomerOrganizations() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [createOpen, setCreateOpen] = useState(false);
  const { dateValue, onDateChange, preset } = useCustomerUsageRange();
  const { search, setSearch, pagination, onPaginationChange } = useUrlTableState(TABLE_OPTIONS);
  const usage = $api.useQuery(
    "get",
    "/management/v1/customer_organization_usage",
    {
      params: {
        query: {
          start_date: calendarDate(dateValue.from),
          end_date: calendarDate(dateValue.to),
          q: search,
          page: pagination.pageIndex + 1,
          page_size: pagination.pageSize,
        },
      },
    },
    { enabled: Boolean(accessToken) },
  );
  const rows = usage.data?.data ?? [];

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h1 className="text-lg font-semibold tracking-tight">{t("customerPortal.list.title")}</h1>
          <p className="text-sm text-muted-foreground">{t("customerPortal.list.subtitle")}</p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="size-4" /> {t("customerPortal.list.create")}
        </Button>
      </div>
      <div className="rounded-lg border bg-card">
        <div className="flex flex-wrap items-end justify-between gap-3 border-b p-3">
          <Input
            className="max-w-xs"
            aria-label={t("customerPortal.list.searchPlaceholder")}
            placeholder={t("customerPortal.list.searchPlaceholder")}
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <AdvancedDatePicker
            value={dateValue}
            onValueChange={(value, selectedPreset) => {
              onDateChange(value, selectedPreset);
              onPaginationChange({ ...pagination, pageIndex: 0 });
            }}
            presetShortLabel={preset}
            presets={customerUsagePresets}
            label={t("customerPortal.range.label")}
            showTimeRange={false}
            compact
            className="max-w-sm"
          />
        </div>
        {usage.isError ? (
          <ErrorState message={t("customerPortal.list.error")} onRetry={() => void usage.refetch()} />
        ) : (
          <Table aria-label={t("customerPortal.list.title")}>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">{t("customerPortal.list.columns.customer")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.list.columns.spend")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.list.columns.requests")}</TableHead>
                <TableHead className="pr-4 text-right">{t("customerPortal.list.columns.tokens")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {usage.isPending && (
                <TableRow>
                  <TableCell colSpan={4} className="py-8 text-center text-muted-foreground">
                    {t("customerPortal.list.loading")}
                  </TableCell>
                </TableRow>
              )}
              {!usage.isPending && rows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="py-8 text-center text-muted-foreground">
                    {t(search ? "customerPortal.list.noMatches" : "customerPortal.list.empty")}
                  </TableCell>
                </TableRow>
              )}
              {!usage.isPending &&
                rows.map((customer) => (
                  <TableRow key={customer.customer_org_id}>
                    <TableCell className="pl-4">
                      <p className="font-medium">{customer.alias || customer.customer_org_id}</p>
                      <p className="font-mono text-[11px] text-muted-foreground">{customer.customer_org_id}</p>
                    </TableCell>
                    <TableCell className="text-right font-medium tabular-nums">
                      {t("format.usd", { val: customer.spend })}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {t("format.number", { val: customer.requests })}
                    </TableCell>
                    <TableCell className="pr-4 text-right tabular-nums">
                      {t("format.number", { val: customer.tokens })}
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        )}
        <DataTablePagination
          page={pagination.pageIndex}
          pageSize={pagination.pageSize}
          rowCount={usage.data?.meta.total_count ?? 0}
          onPageChange={(pageIndex) => onPaginationChange({ ...pagination, pageIndex })}
          onPageSizeChange={(pageSize) => onPaginationChange({ pageSize, pageIndex: 0 })}
          isLoading={usage.isFetching}
        />
      </div>
      <p className="text-xs text-muted-foreground">{t("customerPortal.list.attribution")}</p>
      <CreateCustomerDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={() => void usage.refetch()} />
    </div>
  );
}
