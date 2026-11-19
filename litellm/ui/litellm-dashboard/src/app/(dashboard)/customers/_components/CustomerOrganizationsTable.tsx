import { useTranslation } from "react-i18next";
import type { SortingState } from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { createSerializer, parseAsString } from "nuqs";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import { uiHref } from "@/utils/uiHref";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import type { CustomerOrganizationRow } from "./customerOrganizations";
import { CustomerBudgetLimitCell } from "./CustomerBudgetColumn";
import CustomerCacheCost from "./CustomerCacheCost";
import { EMPTY_CUSTOMER_USAGE, type CustomerUsageTotal, type CustomerUsageTotals } from "./customerUsageTotals";

const usageHref = createSerializer({ org: parseAsString, from: parseAsString, to: parseAsString });
const SORT_DIRECTIONS = ["ascending", "descending"] as const;
const COLUMNS = [
  { id: "customer", label: "customerPortal.list.columns.customer" },
  { id: "spend", label: "customerPortal.analytics.monthlySpend" },
  { id: "budget", label: "customerPortal.hierarchy.maxBudget" },
  { id: "share", label: "customerPortal.list.columns.overallSpendShare" },
  { id: "tokens", label: "customerPortal.list.columns.tokens", numeric: true },
  { id: "status", label: "customerPortal.list.columns.status" },
] as const;
const TOKEN_BREAKDOWN = [
  { id: "inputTokens", label: "usage.statStrip.tokenPopover.inputTokens", nested: false },
  { id: "cacheReadTokens", label: "customerPortal.tokenUsage.cacheReadTokens", nested: true },
  { id: "cacheWriteTokens", label: "customerPortal.tokenUsage.cacheWriteTokens", nested: true },
  { id: "outputTokens", label: "usage.statStrip.tokenPopover.outputTokens", nested: false },
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
  return (
    <Table aria-label={t("customerPortal.list.title")}>
      <TableHeader>
        <TableRow>
          {COLUMNS.map((column) => {
            const { id, label } = column;
            if ("numeric" in column) {
              return (
                <TableHead key={id} className="px-3 text-right whitespace-normal">
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
                className={cn("px-4", id === "spend" && "min-w-40", id === "status" && "text-right")}
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
                <TableCell className="px-3 text-right tabular-nums">
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

function CustomerTokenTotal({ usage }: { usage: CustomerUsageTotal | null }) {
  const { t } = useTranslation();
  if (!usage) return "—";
  const total = t("format.number", { val: usage.tokens });
  const cacheReadRatio = usage.inputTokens > 0 ? usage.cacheReadTokens / usage.inputTokens : null;
  const percentCached =
    cacheReadRatio !== null && cacheReadRatio >= 0 && cacheReadRatio <= 1
      ? t("customerPortal.tokenUsage.percentCached", {
          percent: t("format.number", { val: cacheReadRatio, style: "percent", maximumFractionDigits: 0 }),
        })
      : undefined;
  return (
    <Popover>
      <PopoverTrigger
        openOnHover
        delay={200}
        aria-label={t("customerPortal.tokenUsage.viewTokenBreakdown", { total })}
        aria-description={percentCached}
        className="rounded-sm text-right focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="underline decoration-muted-foreground/60 decoration-dotted underline-offset-4">
          {t("format.number", { val: usage.tokens, notation: "compact", maximumFractionDigits: 2 })}
        </span>
        {percentCached && <span className="block text-xs text-muted-foreground">{percentCached}</span>}
      </PopoverTrigger>
      <PopoverContent align="end" className="gap-2">
        <PopoverTitle>{t("customerPortal.tokenUsage.tokenBreakdown")}</PopoverTitle>
        <dl className="space-y-1 tabular-nums">
          {TOKEN_BREAKDOWN.map(({ id, label, nested }) => (
            <div key={id} className="flex justify-between gap-3">
              <dt className={cn("text-muted-foreground", nested && "pl-3")}>{t(label)}</dt>
              <dd>{t("format.number", { val: usage[id] })}</dd>
            </div>
          ))}
          <div className="mt-2 flex justify-between gap-3 border-t border-border pt-2">
            <dt className="text-muted-foreground">{t("customerPortal.tokenUsage.cacheCost")}</dt>
            <dd className="text-right">
              <CustomerCacheCost usage={usage} />
            </dd>
          </div>
        </dl>
      </PopoverContent>
    </Popover>
  );
}
