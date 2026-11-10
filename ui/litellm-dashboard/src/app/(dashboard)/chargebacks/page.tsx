"use client";

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { Download } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { dailyActivityAggregatedCall } from "@/components/networking";
import { toDailyData } from "@/components/UsagePage/dailyActivityApi";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { monthRange, useCustomerUsage } from "../customers/_components/customerPortalData";
import {
  rowsByCustomer,
  rowsByCustomerAndKey,
  rowsByCustomerAndModel,
  sumRows,
  toCsv,
  type GroupBy,
} from "./chargebackRows";

const GROUPS: readonly GroupBy[] = ["customer", "key", "model"];
const MONTHS = [0, 1, 2] as const;

const downloadCsv = (csv: string, filename: string) => {
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  Object.assign(document.createElement("a"), { href: url, download: filename }).click();
  URL.revokeObjectURL(url);
};

export default function ChargebacksPage() {
  const { t, i18n } = useTranslation();
  const { accessToken } = useAuthorized();
  const [monthsBack, setMonthsBack] = useState(0);
  const [groupBy, setGroupBy] = useState<GroupBy>("customer");
  const customersQuery = useCustomers(100);
  const customers = useMemo(() => customersQuery.data?.pages.flat() ?? [], [customersQuery.data]);
  const aliasOf = useMemo(() => new Map(customers.map((c) => [c.user_id, c.alias || c.user_id])), [customers]);
  const usageQuery = useCustomerUsage(accessToken, monthsBack);
  const customerIds = useMemo(() => [...(usageQuery.data?.keys() ?? [])].sort(), [usageQuery.data]);

  const perModel = useQuery({
    queryKey: ["customer-portal", "usage-by-model", monthsBack, customerIds],
    enabled: Boolean(accessToken) && groupBy === "model" && usageQuery.isSuccess,
    queryFn: async () => {
      const { start, end } = monthRange(monthsBack);
      const entries = await Promise.all(
        customerIds.map(async (id) => {
          const request = { accessToken: accessToken!, startTime: start, endTime: end, entityIds: [id] };
          const res = await dailyActivityAggregatedCall("customer", request);
          return [id, toDailyData(res)] as const;
        }),
      );
      return new Map(entries);
    },
  });

  const rows = useMemo(() => {
    if (groupBy === "model") return perModel.data ? rowsByCustomerAndModel(perModel.data) : [];
    if (!usageQuery.data) return [];
    return groupBy === "key" ? rowsByCustomerAndKey(usageQuery.data) : rowsByCustomer(usageQuery.data);
  }, [groupBy, usageQuery.data, perModel.data]);
  const total = sumRows(rows);
  const isLoading = usageQuery.isLoading || (groupBy === "model" && perModel.isLoading);
  const dimensionLabel = groupBy === "customer" ? null : t(`customerPortal.chargebacks.dimension.${groupBy}`);
  const monthLabel = (back: number) =>
    new Intl.DateTimeFormat(i18n.language, { month: "long", year: "numeric", timeZone: "UTC" }).format(
      monthRange(back).start,
    );
  const exportCsv = () =>
    downloadCsv(
      toCsv(rows, aliasOf, dimensionLabel?.toLowerCase() ?? null),
      `chargebacks-${monthRange(monthsBack).start.toISOString().slice(0, 7)}-${groupBy}.csv`,
    );

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">{t("customerPortal.chargebacks.title")}</h1>
          <p className="truncate text-sm text-muted-foreground">{t("customerPortal.chargebacks.subtitle")}</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={String(monthsBack)} onValueChange={(v) => setMonthsBack(Number(v))}>
            <SelectTrigger className="w-44">
              <SelectValue>{monthLabel(monthsBack)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {MONTHS.map((b) => (
                <SelectItem key={b} value={String(b)}>
                  {monthLabel(b)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={groupBy} onValueChange={(v) => setGroupBy(v as GroupBy)}>
            <SelectTrigger className="w-48">
              <SelectValue>{t(`customerPortal.chargebacks.groupBy.${groupBy}`)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {GROUPS.map((g) => (
                <SelectItem key={g} value={g}>
                  {t(`customerPortal.chargebacks.groupBy.${g}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={exportCsv} disabled={rows.length === 0}>
            <Download className="size-4" /> {t("customerPortal.chargebacks.exportCsv")}
          </Button>
        </div>
      </div>
      <div className="rounded-lg border bg-card">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">{t("customerPortal.chargebacks.columns.customer")}</TableHead>
              {dimensionLabel && <TableHead>{dimensionLabel}</TableHead>}
              <TableHead className="text-right">{t("customerPortal.chargebacks.columns.requests")}</TableHead>
              <TableHead className="text-right">{t("customerPortal.chargebacks.columns.inputTokens")}</TableHead>
              <TableHead className="text-right">{t("customerPortal.chargebacks.columns.outputTokens")}</TableHead>
              <TableHead className="pr-4 text-right">{t("customerPortal.chargebacks.columns.spend")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && (
              <TableRow>
                <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                  {t("customerPortal.common.loading")}
                </TableCell>
              </TableRow>
            )}
            {!isLoading && rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                  {t("customerPortal.chargebacks.empty", { month: monthLabel(monthsBack) })}
                </TableCell>
              </TableRow>
            )}
            {rows.map((r) => (
              <TableRow key={`${r.customer}-${r.dimension ?? ""}`}>
                <TableCell className="pl-4">
                  <p className="font-medium">{aliasOf.get(r.customer) ?? r.customer}</p>
                  <p className="font-mono text-[11px] text-muted-foreground">{r.customer}</p>
                </TableCell>
                {dimensionLabel && <TableCell>{r.dimension}</TableCell>}
                <TableCell className="text-right tabular-nums">{t("format.number", { val: r.requests })}</TableCell>
                <TableCell className="text-right tabular-nums">{t("format.number", { val: r.inputTokens })}</TableCell>
                <TableCell className="text-right tabular-nums">{t("format.number", { val: r.outputTokens })}</TableCell>
                <TableCell className="pr-4 text-right tabular-nums">{t("format.usd", { val: r.spend })}</TableCell>
              </TableRow>
            ))}
            {rows.length > 0 && (
              <TableRow className="font-medium">
                <TableCell className="pl-4" colSpan={dimensionLabel ? 2 : 1}>
                  {t("customerPortal.common.total")}
                </TableCell>
                <TableCell className="text-right tabular-nums">{t("format.number", { val: total.requests })}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {t("format.number", { val: total.inputTokens })}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {t("format.number", { val: total.outputTokens })}
                </TableCell>
                <TableCell className="pr-4 text-right tabular-nums">{t("format.usd", { val: total.spend })}</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
