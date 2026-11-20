"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { parseAsString, parseAsInteger, parseAsNumberLiteral, useQueryStates } from "nuqs";
import { DataTablePagination } from "@/components/shared/DataTable";
import { useTranslation } from "react-i18next";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import {
  CustomerSpendCells,
  CustomerSpendHeaders,
  CustomerSpendIdentity,
  CustomerSpendTier,
} from "./CustomerSpendCells";
import { compareCustomerBudgets, customerBudgetSortParser } from "./customerBudgetUsage";
import { CustomerBudgetSortHeader } from "./CustomerBudgetColumn";
import { customerUsageTotals, EMPTY_CUSTOMER_USAGE } from "./customerUsageTotals";
import CustomerTokenTotal from "./CustomerTokenTotal";
import CustomerEntityBudgetDialog from "./CustomerEntityBudgetDialog";

const PAGE_SIZE_OPTIONS = [5, 10, 25];
const MEMBER_TABLE_PARSERS = {
  memberSearch: parseAsString.withDefault(""),
  memberPage: parseAsInteger.withDefault(1),
  memberPageSize: parseAsNumberLiteral(PAGE_SIZE_OPTIONS).withDefault(5),
  budgetSort: customerBudgetSortParser,
};
export default function CustomerMemberSpend({
  data,
  onMemberClick,
}: {
  data: CustomerHierarchyUsage;
  onMemberClick: (id: string) => void;
}) {
  const { t } = useTranslation();
  const [editingMember, setEditingMember] = useState<string | null>(null);
  const [table, setTable] = useQueryStates(MEMBER_TABLE_PARSERS);
  const search = table.memberSearch;
  const tokenTotals = useMemo(() => customerUsageTotals(data.memberResults), [data.memberResults]);
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
        .sort((a, b) =>
          table.budgetSort
            ? compareCustomerBudgets(a, b, table.budgetSort === "budget_desc")
            : (b.monthlyUsage?.spend ?? 0) - (a.monthlyUsage?.spend ?? 0) || a.label.localeCompare(b.label),
        ),
    [data.memberBudgets, search, usage, table.budgetSort],
  );

  const pageSize = table.memberPageSize;
  const lastPage = Math.max(1, Math.ceil(members.length / pageSize));
  const page = Math.max(1, Math.min(table.memberPage, lastPage));
  const start = (page - 1) * pageSize;
  return (
    <section className="border-b pb-6" aria-label={t("customerPortal.analytics.memberSpend")}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <h2 className="text-base font-medium tracking-tight">
          {t(data.member ? "customerPortal.analytics.memberSpend" : "customerPortal.analytics.topMembers")}
        </h2>
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
      <div className="overflow-hidden rounded-lg border">
        <Table aria-label={t("customerPortal.analytics.memberSpend")}>
          <TableHeader>
            <TableRow>
              <TableHead className="px-4">{t("customerPortal.analytics.member")}</TableHead>
              <TableHead className="px-4">{t("customerPortal.budgetEditor.group")}</TableHead>
              <CustomerSpendHeaders usageLayout>
                <CustomerBudgetSortHeader
                  sort={table.budgetSort}
                  onSort={() =>
                    void setTable(
                      {
                        budgetSort: table.budgetSort === "budget_asc" ? "budget_desc" : "budget_asc",
                        memberPage: null,
                      },
                      { history: "push" },
                    )
                  }
                />
              </CustomerSpendHeaders>
              <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {members.slice(start, start + pageSize).map((member) => (
              <TableRow key={member.id}>
                <TableCell className="px-4 py-5">
                  <CustomerSpendIdentity entity={member} onClick={() => onMemberClick(member.id)} usageLayout />
                </TableCell>
                <CustomerSpendTier entity={member} />
                <CustomerSpendCells
                  entity={member}
                  monthlySpend={member.monthlyUsage?.spend ?? 0}
                  yearlySpend={yearlyUsage.get(member.id) ?? 0}
                  tokens={member.monthlyUsage?.tokens ?? 0}
                  organizationSpend={data.monthlyOrganizationSpend}
                  usageLayout
                  tokenContent={<CustomerTokenTotal usage={tokenTotals.get(member.id) ?? EMPTY_CUSTOMER_USAGE} />}
                />
                <TableCell className="px-4 text-right">
                  <Button
                    variant="ghost"
                    size="xs"
                    aria-label={t("customerPortal.budgetEditor.memberAction", { name: member.label })}
                    onClick={() => setEditingMember(member.id)}
                  >
                    {t("customerPortal.hierarchy.edit")}
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {members.length === 0 && (
              <TableRow>
                <TableCell colSpan={9} className="py-10 text-center text-muted-foreground">
                  {t(search ? "customerPortal.analytics.noMatches" : "customerPortal.analytics.noMembers")}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        <DataTablePagination
          className="justify-end border-t px-4 py-3"
          page={page - 1}
          pageSize={pageSize}
          pageSizeOptions={PAGE_SIZE_OPTIONS}
          rowCount={members.length}
          onPageChange={(index) => void setTable({ memberPage: index + 1 })}
          onPageSizeChange={(size) => void setTable({ memberPageSize: size, memberPage: null })}
        />
      </div>
      {editingMember && (
        <CustomerEntityBudgetDialog
          memberId={editingMember}
          organizationId={data.organization.id}
          onClose={() => setEditingMember(null)}
        />
      )}
    </section>
  );
}
