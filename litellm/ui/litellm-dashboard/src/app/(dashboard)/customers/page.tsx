"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { createSerializer, parseAsString, useQueryState } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import { customerUsageTotals } from "./_components/customerUsageTotals";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { DataTablePagination, useUrlTableState } from "@/components/shared/DataTable";
import { ErrorState } from "@/components/shared/ErrorState";
import { uiHref } from "@/utils/uiHref";
import CreateCustomerDialog from "./_components/CreateCustomerDialog";
import CustomerDetail from "./_components/CustomerDetail";
import {
  calendarDate,
  CUSTOMER_USAGE_MAX_RANGE_DAYS,
  customerUsagePresets,
  useCustomerUsageRange,
} from "./_components/customerUsageRange";

const TABLE_OPTIONS = {
  sortFields: ["created_at"],
  defaultSort: { id: "created_at", desc: false },
  defaultPageSize: 25,
  filterColumns: [],
};

const usageHref = createSerializer({ org: parseAsString, from: parseAsString, to: parseAsString });

export default function CustomersPage() {
  const [selectedId, setSelectedId] = useQueryState("id", parseAsString);
  if (selectedId) return <CustomerDetail customerId={selectedId} onBack={() => void setSelectedId(null)} />;
  return <CustomerOrganizations />;
}

function CustomerSpend({
  spend,
  budgetSpend,
  maxBudget,
}: {
  spend: number;
  budgetSpend: number;
  maxBudget: number | null;
}) {
  const { t } = useTranslation();
  const spendLabel = t("format.usd", { val: spend });
  if (maxBudget === null) {
    return (
      <>
        <p className="font-medium">{spendLabel}</p>
        <p className="text-xs text-muted-foreground">{t("customerPortal.list.noMaxBudget")}</p>
      </>
    );
  }
  const percentage =
    maxBudget > 0
      ? t("customerPortal.list.budgetUsed", {
          percentage: t("format.number", {
            val: budgetSpend / maxBudget,
            style: "percent",
            maximumFractionDigits: 1,
          }),
        })
      : t("customerPortal.list.zeroBudgetPercentage");
  const budgetSpendLabel = t("format.usd", { val: budgetSpend });
  const budgetLabel = t("customerPortal.list.budgetUsage", {
    spend: budgetSpendLabel,
    max: t("format.usd", { val: maxBudget }),
  });
  const matchesBudgetSpend = spendLabel === budgetSpendLabel;
  return (
    <>
      <p className="whitespace-nowrap font-medium">{matchesBudgetSpend ? budgetLabel : spendLabel}</p>
      {!matchesBudgetSpend && (
        <p className="text-xs text-muted-foreground">
          {t("customerPortal.list.currentBudget", { budget: budgetLabel })}
        </p>
      )}
      <p className="text-xs text-muted-foreground">{percentage}</p>
    </>
  );
}

function CustomerOrganizations() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [createOpen, setCreateOpen] = useState(false);
  const { dateValue, onDateChange, preset } = useCustomerUsageRange();
  const { search, setSearch, pagination, onPaginationChange } = useUrlTableState(TABLE_OPTIONS);
  const organizations = $api.useQuery(
    "get",
    "/management/v1/customer_organizations",
    {
      params: {
        query: { q: search, page: pagination.pageIndex + 1, page_size: pagination.pageSize },
      },
    },
    { enabled: Boolean(accessToken) },
  );
  const rows = organizations.data?.data ?? [];
  const usage = $api.useQuery(
    "get",
    "/management/v1/customer_organizations/daily/activity/aggregated",
    {
      params: {
        query: {
          start_date: calendarDate(dateValue.from),
          end_date: calendarDate(dateValue.to),
          customer_org_ids: rows.map((customer) => customer.customer_org_id).join(","),
        },
      },
    },
    { enabled: Boolean(accessToken) && rows.length > 0 },
  );
  const loading = organizations.isPending || (rows.length > 0 && usage.isPending);
  const totals = customerUsageTotals(usage.data?.results ?? []);

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
          <Table aria-label={t("customerPortal.list.title")}>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">{t("customerPortal.list.columns.customer")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.list.columns.spend")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.list.columns.requests")}</TableHead>
                <TableHead className="text-right">{t("customerPortal.list.columns.tokens")}</TableHead>
                <TableHead className="pr-4 text-right">{t("customerPortal.list.columns.members")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading && (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                    {t("customerPortal.list.loading")}
                  </TableCell>
                </TableRow>
              )}
              {!loading && rows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="py-8 text-center text-muted-foreground">
                    {t(search ? "customerPortal.list.noMatches" : "customerPortal.list.empty")}
                  </TableCell>
                </TableRow>
              )}
              {!loading &&
                rows.map((customer) => (
                  <TableRow key={customer.customer_org_id}>
                    <TableCell className="pl-4">
                      <a
                        className="font-medium text-primary hover:underline"
                        href={usageHref(uiHref("customers/usage"), {
                          org: customer.customer_org_id,
                          from: calendarDate(dateValue.from),
                          to: calendarDate(dateValue.to),
                        })}
                      >
                        {customer.alias || customer.customer_org_id}
                      </a>
                      <p className="font-mono text-[11px] text-muted-foreground">{customer.customer_org_id}</p>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      <CustomerSpend
                        spend={totals[customer.customer_org_id]?.spend ?? 0}
                        budgetSpend={customer.spend}
                        maxBudget={customer.budget?.max_budget ?? null}
                      />
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {t("format.number", { val: totals[customer.customer_org_id]?.requests ?? 0 })}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {t("format.number", { val: totals[customer.customer_org_id]?.tokens ?? 0 })}
                    </TableCell>
                    <TableCell className="pr-4 text-right tabular-nums">
                      {t("format.number", { val: customer.member_count })}
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        )}
        <DataTablePagination
          page={pagination.pageIndex}
          pageSize={pagination.pageSize}
          rowCount={organizations.data?.meta.total_count ?? 0}
          onPageChange={(pageIndex) => onPaginationChange({ ...pagination, pageIndex })}
          onPageSizeChange={(pageSize) => onPaginationChange({ pageSize, pageIndex: 0 })}
          isLoading={organizations.isFetching || usage.isFetching}
        />
      </div>
      <p className="text-xs text-muted-foreground">{t("customerPortal.list.budgetAndMembersHelp")}</p>
      <p className="text-xs text-muted-foreground">{t("customerPortal.list.attribution")}</p>
      <CreateCustomerDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={() => void organizations.refetch()}
      />
    </div>
  );
}
