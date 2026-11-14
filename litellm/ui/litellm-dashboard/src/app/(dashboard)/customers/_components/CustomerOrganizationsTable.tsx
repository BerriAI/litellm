import { useTranslation } from "react-i18next";
import type { SortingState } from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { createSerializer, parseAsString } from "nuqs";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import { uiHref } from "@/utils/uiHref";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import type { CustomerOrganizationRow } from "./customerOrganizations";

const usageHref = createSerializer({ org: parseAsString, from: parseAsString, to: parseAsString });
const SORT_DIRECTIONS = ["ascending", "descending"] as const;
const COLUMNS = [
  { id: "customer", label: "customerPortal.list.columns.customer" },
  { id: "spend", label: "customerPortal.analytics.monthlySpend" },
  { id: "share", label: "customerPortal.list.columns.overallSpendShare" },
  { id: "status", label: "customerPortal.list.columns.status" },
] as const;

export default function CustomerOrganizationsTable({
  rows,
  sorting,
  onSort,
  loading,
  searching,
  startDate,
  endDate,
}: {
  rows: readonly CustomerOrganizationRow[];
  sorting: SortingState;
  onSort: (sorting: SortingState) => void;
  loading: boolean;
  searching: boolean;
  startDate: string;
  endDate: string;
}) {
  const { t } = useTranslation();
  return (
    <Table aria-label={t("customerPortal.list.title")}>
      <TableHeader>
        <TableRow>
          {COLUMNS.map(({ id, label }) => {
            const active = sorting[0]?.id === id;
            const descending = sorting[0]?.desc ?? false;
            const SortIcon = descending ? ArrowDown : ArrowUp;
            const Icon = active ? SortIcon : ArrowUpDown;
            const direction = SORT_DIRECTIONS[Number(descending)];
            return (
              <TableHead
                key={id}
                aria-sort={active ? direction : "none"}
                className={cn("px-4", id === "spend" && "min-w-80", id === "status" && "text-right")}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  className="-mx-2 gap-2"
                  onClick={() => onSort([{ id, desc: active ? !descending : id !== "customer" }])}
                >
                  {t(label)}
                  <Icon aria-hidden="true" className="size-3.5" />
                </Button>
              </TableHead>
            );
          })}
        </TableRow>
      </TableHeader>
      <TableBody>
        {loading && (
          <TableRow>
            <TableCell colSpan={4} className="py-8 text-center text-muted-foreground">
              {t("customerPortal.list.loading")}
            </TableCell>
          </TableRow>
        )}
        {!loading && rows.length === 0 && (
          <TableRow>
            <TableCell colSpan={4} className="py-8 text-center text-muted-foreground">
              {t(searching ? "customerPortal.list.noMatches" : "customerPortal.list.empty")}
            </TableCell>
          </TableRow>
        )}
        {!loading &&
          rows.map(({ organization, monthlySpend, monthlyBudgets, share, status }) => {
            const label = organization.alias || organization.customer_org_id;
            const percentage = Math.max(0, Math.min(100, share * 100));
            const shareLabel = t("format.number", { val: share, style: "percent", maximumFractionDigits: 1 });
            return (
              <TableRow key={organization.customer_org_id}>
                <TableCell className="px-4 py-5">
                  <div className="flex items-center gap-3">
                    <span
                      aria-hidden="true"
                      className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium"
                    >
                      {label
                        .split(/\s+/)
                        .slice(0, 2)
                        .map((part) => part.charAt(0))
                        .join("")
                        .toLocaleUpperCase()}
                    </span>
                    <div className="min-w-0">
                      <a
                        className="font-medium hover:text-primary hover:underline"
                        href={usageHref(uiHref("customers/usage"), {
                          org: organization.customer_org_id,
                          from: startDate,
                          to: endDate,
                        })}
                      >
                        {label}
                      </a>
                      <p className="font-mono text-[11px] text-muted-foreground">{organization.customer_org_id}</p>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="px-4">
                  <div className="max-w-lg space-y-3">
                    {monthlyBudgets.map((budget) => (
                      <CustomerBudgetWindow
                        key={`${budget.source}-${budget.budget_duration}`}
                        budget={budget}
                        showReset={monthlyBudgets.length > 1}
                      />
                    ))}
                    {monthlyBudgets.length === 0 && (
                      <CustomerBudgetSpend
                        spend={monthlySpend}
                        limit={null}
                        label={t("customerPortal.analytics.monthlySpend")}
                        noLimitLabel={t("customerPortal.analytics.noMonthlyLimit")}
                      />
                    )}
                  </div>
                </TableCell>
                <TableCell className="px-4">
                  <div className="flex items-center gap-3">
                    <div
                      role="meter"
                      aria-label={t("customerPortal.list.columns.overallSpendShare")}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={percentage}
                      aria-valuetext={shareLabel}
                      className="h-2 w-24 overflow-hidden rounded-full bg-muted"
                    >
                      <div className="h-full rounded-full bg-primary/60" style={{ width: `${percentage}%` }} />
                    </div>
                    <span className="text-xs tabular-nums text-muted-foreground">{shareLabel}</span>
                  </div>
                </TableCell>
                <TableCell className="px-4 text-right">
                  <span
                    className={cn(
                      "inline-flex whitespace-nowrap rounded-full bg-muted px-2 py-0.5 text-xs",
                      status === "customerPortal.analytics.reachedLimit" && "bg-destructive/10 text-destructive",
                    )}
                  >
                    {t(status)}
                  </span>
                </TableCell>
              </TableRow>
            );
          })}
      </TableBody>
    </Table>
  );
}
