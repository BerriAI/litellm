"use client";

import { customerLimitsOnKey, useAllKeys } from "./_components/customerKeys";
import { useMemo, useState } from "react";
import { parseAsString, useQueryState } from "nuqs";
import { Plus } from "lucide-react";
import { toast } from "sonner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCustomers, type EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { $api } from "@/lib/http/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import CustomerDetail from "./_components/CustomerDetail";
import CreateCustomerDialog from "./_components/CreateCustomerDialog";
import { customerStatus, formatUSD, useCustomerSpendByCustomer, type CustomerStatus } from "./_components/customerPortalData";

const STATUS_LABEL: Record<CustomerStatus, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
  active: { label: "Active", variant: "secondary" },
  over_budget: { label: "Over budget", variant: "destructive" },
  blocked: { label: "Blocked", variant: "outline" },
};

export default function CustomersPage() {
  const { accessToken } = useAuthorized();
  const [selectedId, setSelectedId] = useQueryState("id", parseAsString);
  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const customersQuery = useCustomers(100, search || undefined);
  const spendQuery = useCustomerSpendByCustomer(accessToken);
  const keysQuery = useAllKeys(accessToken);
  const customers: EndUser[] = useMemo(() => customersQuery.data?.pages.flat() ?? [], [customersQuery.data]);
  const rows = useMemo(() => {
    const spend = spendQuery.data;
    return customers
      .map((c) => {
        const s = spend?.get(c.user_id);
        const limited = (keysQuery.data ?? []).filter((k) => {
          const l = customerLimitsOnKey(k, c.user_id);
          return l.tpm !== null || l.rpm !== null;
        }).map((k) => k.token);
        const keys = new Set([...(s?.keys ?? []), ...limited]).size;
        return { c, spend: s?.spend ?? 0, keys, status: customerStatus(c, s?.spend ?? 0) };
      })
      .sort((a, b) => b.spend - a.spend);
  }, [customers, spendQuery.data, keysQuery.data]);
  const block = $api.useMutation("post", "/customer/block", { onSuccess: () => { toast.success("Customer blocked"); customersQuery.refetch(); } });
  const unblock = $api.useMutation("post", "/customer/update", { onSuccess: () => { toast.success("Customer unblocked"); customersQuery.refetch(); } });

  if (selectedId) return <CustomerDetail customerId={selectedId} onBack={() => setSelectedId(null)} />;

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">Customers</h1>
          <p className="truncate text-sm text-muted-foreground">End users of your product, identified by customer id on every request.</p>
        </div>
        <Button onClick={() => setCreateOpen(true)}><Plus className="size-4" /> Create Customer</Button>
      </div>
      <div className="rounded-lg border bg-card">
        <div className="border-b p-3">
          <Input className="max-w-xs" placeholder="Search by customer id or alias" value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">Customer</TableHead>
              <TableHead className="text-right">Spend this month</TableHead>
              <TableHead className="text-right">Budget</TableHead>
              <TableHead className="text-right">Keys</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="w-28" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {customersQuery.isLoading && <TableRow><TableCell colSpan={6} className="py-8 text-center text-muted-foreground">Loading customers...</TableCell></TableRow>}
            {!customersQuery.isLoading && rows.length === 0 && <TableRow><TableCell colSpan={6} className="py-8 text-center text-muted-foreground">No customers yet. Create one to start tracking spend.</TableCell></TableRow>}
            {rows.map(({ c, spend, keys, status }) => {
              const max = c.litellm_budget_table?.max_budget ?? null;
              return (
                <TableRow key={c.user_id} className="cursor-pointer" onClick={() => setSelectedId(c.user_id, { history: "push" })}>
                  <TableCell className="pl-4">
                    <p className="font-medium">{c.alias || c.user_id}</p>
                    <p className="font-mono text-[11px] text-muted-foreground">{c.user_id}</p>
                  </TableCell>
                  <TableCell className="text-right font-medium tabular-nums">{formatUSD(spend)}</TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">{max === null || max === undefined ? "No budget" : `${formatUSD(spend)} / ${formatUSD(max)}`}</TableCell>
                  <TableCell className="text-right tabular-nums">{keys}</TableCell>
                  <TableCell><Badge variant={STATUS_LABEL[status].variant}>{STATUS_LABEL[status].label}</Badge></TableCell>
                  <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                    {c.blocked ? (
                      <Button size="sm" variant="outline" onClick={() => unblock.mutate({ body: { user_id: c.user_id, blocked: false } })}>Unblock</Button>
                    ) : (
                      <Button size="sm" variant="outline" onClick={() => block.mutate({ body: { user_ids: [c.user_id] } })}>Block</Button>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      <CreateCustomerDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={(id) => { customersQuery.refetch(); setSelectedId(id, { history: "push" }); }} />
    </div>
  );
}
