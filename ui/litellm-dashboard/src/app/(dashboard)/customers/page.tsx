"use client";

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { parseAsString, useQueryState } from "nuqs";
import { Plus } from "lucide-react";
import { toast } from "sonner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCustomers, type EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { $api } from "@/lib/http/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import CustomerDetail from "./_components/CustomerDetail";
import CreateCustomerDialog from "./_components/CreateCustomerDialog";
import StatusBadge from "./_components/StatusBadge";
import { rankBySpend, useCustomerUsage } from "./_components/customerPortalData";

export default function CustomersPage() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [selectedId, setSelectedId] = useQueryState("id", parseAsString);
  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const customersQuery = useCustomers(100, search || undefined);
  const usageQuery = useCustomerUsage(accessToken);
  const customers: EndUser[] = useMemo(() => customersQuery.data?.pages.flat() ?? [], [customersQuery.data]);
  const rows = useMemo(() => rankBySpend(customers, usageQuery.data), [customers, usageQuery.data]);
  const block = $api.useMutation("post", "/customer/block", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.blocked"));
      customersQuery.refetch();
    },
  });
  const unblock = $api.useMutation("post", "/customer/update", {
    onSuccess: () => {
      toast.success(t("customerPortal.toasts.unblocked"));
      customersQuery.refetch();
    },
  });

  if (selectedId) return <CustomerDetail customerId={selectedId} onBack={() => setSelectedId(null)} />;

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">{t("customerPortal.list.title")}</h1>
          <p className="truncate text-sm text-muted-foreground">{t("customerPortal.list.subtitle")}</p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="size-4" /> {t("customerPortal.list.create")}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">{t("customerPortal.common.blockHint")}</p>
      <div className="rounded-lg border bg-card">
        <div className="border-b p-3">
          <Input
            className="max-w-xs"
            placeholder={t("customerPortal.list.searchPlaceholder")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">{t("customerPortal.list.columns.customer")}</TableHead>
              <TableHead className="text-right">{t("customerPortal.list.columns.spend")}</TableHead>
              <TableHead className="text-right">{t("customerPortal.list.columns.budget")}</TableHead>
              <TableHead className="text-right">{t("customerPortal.list.columns.keys")}</TableHead>
              <TableHead>{t("customerPortal.list.columns.status")}</TableHead>
              <TableHead className="w-28" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {customersQuery.isLoading && (
              <TableRow>
                <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                  {t("customerPortal.list.loading")}
                </TableCell>
              </TableRow>
            )}
            {!customersQuery.isLoading && rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                  {t("customerPortal.list.empty")}
                </TableCell>
              </TableRow>
            )}
            {rows.map(({ c, spend, keys, status }) => {
              const max = c.litellm_budget_table?.max_budget ?? null;
              return (
                <TableRow
                  key={c.user_id}
                  className="cursor-pointer"
                  onClick={() => setSelectedId(c.user_id, { history: "push" })}
                >
                  <TableCell className="pl-4">
                    <p className="font-medium">{c.alias || c.user_id}</p>
                    <p className="font-mono text-[11px] text-muted-foreground">{c.user_id}</p>
                  </TableCell>
                  <TableCell className="text-right font-medium tabular-nums">
                    {t("format.usd", { val: spend })}
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">
                    {max === null
                      ? t("customerPortal.common.noBudget")
                      : t("customerPortal.list.budgetUsage", {
                          spend: t("format.usd", { val: spend }),
                          max: t("format.usd", { val: max }),
                        })}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{keys}</TableCell>
                  <TableCell>
                    <StatusBadge status={status} />
                  </TableCell>
                  <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                    {c.blocked ? (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => unblock.mutate({ body: { user_id: c.user_id, blocked: false } })}
                      >
                        {t("customerPortal.common.unblock")}
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => block.mutate({ body: { user_ids: [c.user_id] } })}
                      >
                        {t("customerPortal.common.block")}
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      <CreateCustomerDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={(id) => {
          customersQuery.refetch();
          setSelectedId(id, { history: "push" });
        }}
      />
    </div>
  );
}
