"use client";

import { useMemo } from "react";
import { parseAsString, parseAsInteger, useQueryStates } from "nuqs";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";
import { Search } from "lucide-react";
import { formatCompact, formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import { customerBudgetPeriod, customerBudgetStatus, customerBudgetUsage } from "./customerBudgetUsage";

const MEMBER_TABLE_PARSERS = { memberSearch: parseAsString.withDefault(""), memberPage: parseAsInteger.withDefault(1) };
const PAGE_SIZE = 20;
export default function CustomerMemberSpend({
  data,
  onMemberClick,
}: {
  data: CustomerHierarchyUsage;
  onMemberClick: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [table, setTable] = useQueryStates(MEMBER_TABLE_PARSERS);
  const search = table.memberSearch;
  const usage = useMemo(() => new Map(data.monthlyMembers.map((member) => [member.id, member])), [data.monthlyMembers]);
  const yearlyUsage = useMemo(
    () => new Map(data.yearlyMembers.map((member) => [member.id, member.spend])),
    [data.yearlyMembers],
  );
  const members = useMemo(
    () =>
      data.memberBudgets
        .filter((member) => member.label.toLocaleLowerCase().includes(search.toLocaleLowerCase()))
        .map((member) => ({ ...member, monthlyUsage: usage.get(member.id) }))
        .sort((a, b) => (b.monthlyUsage?.spend ?? 0) - (a.monthlyUsage?.spend ?? 0) || a.label.localeCompare(b.label)),
    [data.memberBudgets, search, usage],
  );

  const lastPage = Math.max(1, Math.ceil(members.length / PAGE_SIZE));
  const page = Math.max(1, Math.min(table.memberPage, lastPage));
  const start = (page - 1) * PAGE_SIZE;
  return (
    <section className="border-b pb-6" aria-label={t("customerPortal.analytics.memberSpend")}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-medium tracking-tight">{t("customerPortal.analytics.memberSpend")}</h2>
          <p className="mt-1 text-xs text-muted-foreground">{t("customerPortal.analytics.memberSpendHint")}</p>
        </div>
        <div className="relative w-full sm:w-56">
          <Search aria-hidden="true" className="absolute top-2.5 left-3 size-3.5 text-muted-foreground" />
          <Input
            className="h-8 pl-9 text-xs"
            value={search}
            onChange={(event) => void setTable({ memberSearch: event.target.value, memberPage: null })}
            aria-label={t("customerPortal.analytics.searchMembers")}
            placeholder={t("customerPortal.analytics.searchMembers")}
          />
        </div>
      </div>
      <Table aria-label={t("customerPortal.analytics.memberSpend")}>
        <TableHeader>
          <TableRow>
            <TableHead className="pl-0">{t("customerPortal.analytics.member")}</TableHead>
            <TableHead className="min-w-72">{t("customerPortal.analytics.monthlySpend")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.analytics.yearlySpend")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.analytics.customerSpendShare")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.analytics.tokens")}</TableHead>
            <TableHead className="pr-0 text-right">{t("customerPortal.analytics.status")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {members.slice(start, start + PAGE_SIZE).map((member) => {
            const monthlySpend = member.monthlyUsage?.spend ?? 0;
            const budgets = customerBudgetUsage(member);
            const monthlyBudgets = budgets.filter(
              (budget) => customerBudgetPeriod(budget.budget_duration) === "monthly",
            );
            const yearlyBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "yearly");
            const otherBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "other");
            const yearlySpend = yearlyUsage.get(member.id) ?? 0;
            const spendShare =
              data.monthlyOrganizationSpend > 0 ? (monthlySpend / data.monthlyOrganizationSpend) * 100 : 0;
            const sharePercentage = Math.max(0, Math.min(100, spendShare));
            const shareLabel = t("format.number", {
              val: spendShare / 100,
              style: "percent",
              maximumFractionDigits: 1,
            });
            const status = customerBudgetStatus(budgets);
            const atLimit = status === "customerPortal.analytics.reachedLimit";
            return (
              <TableRow key={member.id}>
                <TableCell className="py-4 pl-0">
                  <div className="flex items-center gap-3">
                    <span
                      aria-hidden="true"
                      className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium"
                    >
                      {member.label
                        .split(/\s+/)
                        .slice(0, 2)
                        .map((part) => part.charAt(0))
                        .join("")
                        .toLocaleUpperCase()}
                    </span>
                    <div className="min-w-0">
                      <button
                        type="button"
                        className="block text-left font-medium hover:text-primary focus-visible:outline-2 focus-visible:outline-ring"
                        onClick={() => onMemberClick(member.id)}
                      >
                        {member.label}
                      </button>
                      <span className="block font-mono text-[11px] text-muted-foreground">{member.id}</span>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="pr-6">
                  <div className="space-y-3">
                    {monthlyBudgets.map((budget) => (
                      <CustomerBudgetWindow key={budget.source} budget={budget} showReset={monthlyBudgets.length > 1} />
                    ))}
                    {monthlyBudgets.length === 0 && (
                      <CustomerBudgetSpend
                        spend={monthlySpend}
                        limit={null}
                        label={t("customerPortal.analytics.monthlySpend")}
                        noLimitLabel={t("customerPortal.analytics.noMonthlyLimit")}
                      />
                    )}
                    {otherBudgets.map((budget) => (
                      <div key={budget.source} className="space-y-1 text-xs text-muted-foreground">
                        <span>
                          {t("customerPortal.analytics.budgetPeriod")}
                          {budget.budget_duration && ` (${budget.budget_duration})`}
                        </span>
                        <CustomerBudgetWindow budget={budget} />
                      </div>
                    ))}
                  </div>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  <div className="space-y-2">
                    {yearlyBudgets.map((budget) => (
                      <CustomerBudgetWindow key={budget.source} budget={budget} showReset={yearlyBudgets.length > 1} />
                    ))}
                    {yearlyBudgets.length === 0 &&
                      `${formatUsd(yearlySpend)} / ${t("customerPortal.analytics.noYearlyLimit")}`}
                  </div>
                </TableCell>
                <TableCell>
                  <div
                    role="meter"
                    aria-label={t("customerPortal.analytics.customerSpendShare")}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={sharePercentage}
                    aria-valuetext={shareLabel}
                    title={shareLabel}
                    className="ml-auto h-2.5 w-24 overflow-hidden rounded-full bg-muted"
                  >
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{ width: `${sharePercentage}%`, opacity: 0.3 + sharePercentage * 0.007 }}
                    />
                  </div>
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatCompact(member.monthlyUsage?.tokens ?? 0)}
                </TableCell>
                <TableCell className="pr-0 text-right">
                  <span
                    className={cn(
                      "inline-flex rounded-full bg-muted px-2 py-0.5 text-xs",
                      atLimit && "bg-destructive/10 text-destructive",
                    )}
                  >
                    {t(status)}
                  </span>
                </TableCell>
              </TableRow>
            );
          })}
          {members.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} className="py-10 text-center text-muted-foreground">
                {t(search ? "customerPortal.analytics.noMatches" : "customerPortal.analytics.noMembers")}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {members.length > PAGE_SIZE && (
        <div className="mt-3 flex items-center justify-between border-t pt-3 text-xs text-muted-foreground">
          <span>
            {t("customerPortal.analytics.pagination", {
              from: start + 1,
              to: Math.min(start + PAGE_SIZE, members.length),
              total: members.length,
            })}
          </span>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={page === 1}
              onClick={() => void setTable({ memberPage: page - 1 })}
            >
              {t("customerPortal.analytics.previous")}
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={page === lastPage}
              onClick={() => void setTable({ memberPage: page + 1 })}
            >
              {t("customerPortal.analytics.next")}
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}
