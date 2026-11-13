"use client";

import { useMemo } from "react";
import { parseAsString, parseAsInteger, useQueryStates } from "nuqs";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import { CustomerSpendCells, CustomerSpendHeaders, CustomerSpendIdentity } from "./CustomerSpendCells";

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
            <CustomerSpendHeaders />
          </TableRow>
        </TableHeader>
        <TableBody>
          {members.slice(start, start + PAGE_SIZE).map((member) => (
            <TableRow key={member.id}>
              <TableCell className="py-4 pl-0">
                <CustomerSpendIdentity entity={member} onClick={() => onMemberClick(member.id)} />
              </TableCell>
              <CustomerSpendCells
                entity={member}
                monthlySpend={member.monthlyUsage?.spend ?? 0}
                yearlySpend={yearlyUsage.get(member.id) ?? 0}
                tokens={member.monthlyUsage?.tokens ?? 0}
                organizationSpend={data.monthlyOrganizationSpend}
              />
            </TableRow>
          ))}
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
