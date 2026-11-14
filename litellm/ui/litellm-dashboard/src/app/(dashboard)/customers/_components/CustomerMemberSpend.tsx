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
import type { CustomerHierarchyUsage, CustomerUsageEntity } from "./customerUsageApi";

const MEMBER_TABLE_PARSERS = { memberSearch: parseAsString.withDefault(""), memberPage: parseAsInteger.withDefault(1) };
const PAGE_SIZE = 20;
const memberStatus = (member: CustomerUsageEntity) => {
  if (member.budgetLimit === null) return "customerPortal.analytics.noCap" as const;
  if (member.spend > member.budgetLimit) return "customerPortal.analytics.overBudget" as const;
  if (member.spend === member.budgetLimit) return "customerPortal.analytics.atBudget" as const;
  return "customerPortal.analytics.withinBudget" as const;
};

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
  const usage = useMemo(() => new Map(data.members.map((member) => [member.id, member])), [data.members]);
  const members = useMemo(
    () =>
      data.memberBudgets
        .filter((member) => member.label.toLocaleLowerCase().includes(search.toLocaleLowerCase()))
        .map((member) => ({ ...member, rangeUsage: usage.get(member.id) }))
        .sort((a, b) => (b.rangeUsage?.spend ?? 0) - (a.rangeUsage?.spend ?? 0) || a.label.localeCompare(b.label)),
    [data.memberBudgets, search, usage],
  );

  const lastPage = Math.max(1, Math.ceil(members.length / PAGE_SIZE));
  const page = Math.max(1, Math.min(table.memberPage, lastPage));
  const start = (page - 1) * PAGE_SIZE;
  return (
    <section className="border-t pt-6" aria-label={t("customerPortal.analytics.memberSpend")}>
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
            <TableHead className="text-right">{t("customerPortal.analytics.spend")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.analytics.tokens")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.analytics.currentBudget")}</TableHead>
            <TableHead className="pr-0 text-right">{t("customerPortal.analytics.status")}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {members.slice(start, start + PAGE_SIZE).map((member) => {
            const atLimit = member.budgetLimit !== null && member.spend >= member.budgetLimit;
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
                <TableCell className="text-right tabular-nums">{formatUsd(member.rangeUsage?.spend ?? 0)}</TableCell>
                <TableCell className="text-right tabular-nums">
                  {formatCompact(member.rangeUsage?.tokens ?? 0)}
                </TableCell>
                <TableCell className="text-right text-xs tabular-nums">
                  {member.budgetLimit === null
                    ? t("customerPortal.analytics.noBudget")
                    : `${formatUsd(member.spend)} / ${formatUsd(member.budgetLimit)}`}
                </TableCell>
                <TableCell className="pr-0 text-right">
                  <span
                    className={cn(
                      "inline-flex rounded-full bg-muted px-2 py-0.5 text-xs",
                      atLimit && "bg-destructive/10 text-destructive",
                    )}
                  >
                    {t(memberStatus(member))}
                  </span>
                </TableCell>
              </TableRow>
            );
          })}
          {members.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="py-10 text-center text-muted-foreground">
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
