import { useTranslation } from "react-i18next";
import type { SortingState } from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { createSerializer, parseAsString } from "nuqs";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import { uiHref } from "@/utils/uiHref";
import { $api } from "@/lib/http/api";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import { customerOrganizationBudgetSource, type CustomerOrganizationRow } from "./customerOrganizations";
import { CustomerBudgetLimitCell } from "./CustomerBudgetColumn";
import CustomerTokenTotal from "./CustomerTokenTotal";
import { EMPTY_CUSTOMER_USAGE, type CustomerUsageTotals } from "./customerUsageTotals";

const usageHref = createSerializer({ org: parseAsString, from: parseAsString, to: parseAsString });
const SORT_DIRECTIONS = ["ascending", "descending"] as const;
const COLUMNS = [
  { id: "customer", label: "customerPortal.list.columns.customer" },
  { id: "group", label: "customerPortal.list.columns.tier" },
  { id: "spend", label: "customerPortal.list.columns.monthlySpend" },
  { id: "budget", label: "customerPortal.hierarchy.maxBudget" },
  { id: "share", label: "customerPortal.list.columns.overallSpendShare" },
  { id: "tokens", label: "customerPortal.list.columns.tokens", numeric: true },
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
  usageTotals,
}: {
  rows: readonly CustomerOrganizationRow[];
  sorting: SortingState;
  onSort: (sorting: SortingState) => void;
  loading: boolean;
  searching: boolean;
  startDate: string;
  endDate: string;
  usageTotals: CustomerUsageTotals | null;
}) {
  const { t } = useTranslation();
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {});
  return (
    <Table aria-label={t("customerPortal.list.title")}>
      <TableHeader>
        <TableRow>
          {COLUMNS.map((column) => {
            const { id, label } = column;
            if (id === "group" || "numeric" in column) {
              return (
                <TableHead key={id} className={cn("px-4", "numeric" in column && "text-right whitespace-normal")}>
                  {t(label)}
                </TableHead>
              );
            }
            const active = sorting[0]?.id === id;
            const descending = sorting[0]?.desc ?? false;
            const SortIcon = descending ? ArrowDown : ArrowUp;
            const Icon = active ? SortIcon : ArrowUpDown;
            const direction = SORT_DIRECTIONS[Number(descending)];
            return (
              <TableHead
                key={id}
                aria-sort={active ? direction : "none"}
                className={cn("px-4", id === "spend" && "w-[30%] min-w-80", id === "status" && "text-right")}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  className={cn("-mx-2 gap-2", id === "share" && "h-auto min-h-8 py-1 text-left whitespace-normal")}
                  onClick={() => onSort([{ id, desc: active ? !descending : id !== "customer" && id !== "budget" }])}
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
            <TableCell colSpan={COLUMNS.length} className="py-8 text-center text-muted-foreground">
              {t("customerPortal.list.loading")}
            </TableCell>
          </TableRow>
        )}
        {!loading && rows.length === 0 && (
          <TableRow>
            <TableCell colSpan={COLUMNS.length} className="py-8 text-center text-muted-foreground">
              {t(searching ? "customerPortal.list.noMatches" : "customerPortal.list.empty")}
            </TableCell>
          </TableRow>
        )}
        {!loading &&
          rows.map(({ organization, monthlySpend, monthlyBudgets, budgetLimit, share, status }) => {
            const label = organization.alias || organization.customer_org_id;
            const percentage = Math.max(0, Math.min(100, share * 100));
            const shareLabel = t("format.number", { val: share, style: "percent", maximumFractionDigits: 1 });
            const budgetSource = customerOrganizationBudgetSource(organization, groups.data?.default_budgets);
            const customLabel =
              budgetSource?.kind === "custom" ? t("customerPortal.organizationGroups.customBadge") : null;
            const budgetGroup =
              budgetSource?.kind === "group"
                ? budgetSource.name ?? t("customerPortal.organizationGroups.groupBadge")
                : customLabel;
            const usage = usageTotals ? usageTotals.get(organization.customer_org_id) ?? EMPTY_CUSTOMER_USAGE : null;
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
                        className="block max-w-40 truncate font-medium hover:text-primary hover:underline"
                        title={label}
                        href={usageHref(uiHref("customers/usage"), {
                          org: organization.customer_org_id,
                          from: startDate,
                          to: endDate,
                        })}
                      >
                        {label}
                      </a>
                      {label !== organization.customer_org_id && (
                        <p
                          className="max-w-40 truncate font-mono text-[11px] text-muted-foreground"
                          title={organization.customer_org_id}
                        >
                          {organization.customer_org_id}
                        </p>
                      )}
                      <a
                        className="mt-1 inline-block text-xs text-muted-foreground hover:text-foreground hover:underline"
                        href={`${usageHref(uiHref("customers/usage"), { org: organization.customer_org_id, from: startDate, to: endDate })}#app-budgets`}
                      >
                        {t("apps.organizationTitle")}
                      </a>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="px-4 text-muted-foreground">
                  <span className="block max-w-48 truncate" title={budgetGroup ?? undefined}>
                    {budgetGroup ?? "-"}
                  </span>
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
                      />
                    )}
                  </div>
                </TableCell>
                <CustomerBudgetLimitCell limit={budgetLimit} rateLimits={organization.budget} />
                <TableCell className="px-4">
                  <div className="flex items-center gap-3">
                    <div
                      role="meter"
                      aria-label={t("customerPortal.list.columns.overallSpendShare")}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={percentage}
                      aria-valuetext={shareLabel}
                      className="h-2 w-14 overflow-hidden rounded-full bg-muted"
                    >
                      <div className="h-full rounded-full bg-primary/60" style={{ width: `${percentage}%` }} />
                    </div>
                    <span className="text-xs tabular-nums text-muted-foreground">{shareLabel}</span>
                  </div>
                </TableCell>
                <TableCell className="px-4 text-right tabular-nums">
                  <CustomerTokenTotal usage={usage} />
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
