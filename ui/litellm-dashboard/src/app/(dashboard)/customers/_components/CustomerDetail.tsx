"use client";

import { useMemo, useState } from "react";
import { Ban, Pencil, Plus, X } from "lucide-react";
import { toast } from "sonner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { customerStatus, formatUSD, useCustomerSpendByCustomer } from "./customerPortalData";
import { keyTaggedWith, setCustomerTag, useAllKeys, useCustomerTrafficKeys, type PortalKey } from "./customerKeys";
import AddKeyDialog from "./AddKeyDialog";
import EditCustomerDialog from "./EditCustomerDialog";

const STATUS_BADGE = {
  blocked: { variant: "outline", label: "Blocked" },
  over_budget: { variant: "destructive", label: "Over budget" },
  active: { variant: "secondary", label: "Active" },
} as const;

export default function CustomerDetail({ customerId, onBack }: { customerId: string; onBack: () => void }) {
  const { accessToken } = useAuthorized();
  const info = $api.useQuery("get", "/customer/info", { params: { query: { end_user_id: customerId } } }, { enabled: Boolean(accessToken) });
  const spendQuery = useCustomerSpendByCustomer(accessToken);
  const keysQuery = useAllKeys(accessToken);
  const trafficQuery = useCustomerTrafficKeys(accessToken, customerId);
  const [addOpen, setAddOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const block = $api.useMutation("post", "/customer/block", { onSuccess: () => { toast.success("Customer blocked"); info.refetch(); } });
  const unblock = $api.useMutation("post", "/customer/update", { onSuccess: () => { toast.success("Customer unblocked"); info.refetch(); } });

  const c = info.data;
  const spend = spendQuery.data?.get(customerId)?.spend ?? 0;
  const rows = useMemo(() => {
    const keys = keysQuery.data ?? [];
    const traffic = trafficQuery.data ?? new Set<string>();
    return keys
      .map((k) => ({ k, tagged: keyTaggedWith(k, customerId), traffic: traffic.has(k.token) }))
      .filter((r) => r.traffic || r.tagged);
  }, [keysQuery.data, trafficQuery.data, customerId]);
  const shownTokens = useMemo(() => new Set(rows.map((r) => r.k.token)), [rows]);
  const label = c?.alias || customerId;
  const status = c ? customerStatus(c, spend) : "active";
  const max = c?.litellm_budget_table?.max_budget ?? null;
  const refresh = () => { keysQuery.refetch(); trafficQuery.refetch(); };
  const removeKey = async (k: PortalKey) => {
    if (!accessToken) return;
    try {
      await setCustomerTag(accessToken, k, customerId, false);
      toast.success(`Removed ${k.alias} from ${label}`);
      refresh();
    } catch (e) {
      toast.error(`Failed to update key: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <div className="w-full space-y-4 px-6 py-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight">{label}</h1>
            <Badge variant={STATUS_BADGE[status].variant}>{STATUS_BADGE[status].label}</Badge>
            <span className="font-mono text-xs text-muted-foreground">{customerId}</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Spend this month <span className="font-medium text-foreground">{formatUSD(spend)}</span>
            {max !== null && max !== undefined ? <> of <span className="font-medium text-foreground">{formatUSD(max)}</span> budget</> : ", no budget"}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" disabled={!c} onClick={() => setEditOpen(true)}><Pencil className="size-4" /> Edit</Button>
          {c?.blocked ? (
            <Button variant="outline" onClick={() => unblock.mutate({ body: { user_id: customerId, blocked: false } })}><Ban className="size-4" /> Unblock</Button>
          ) : (
            <Button variant="outline" className="text-destructive" onClick={() => block.mutate({ body: { user_ids: [customerId] } })}><Ban className="size-4" /> Block</Button>
          )}
        </div>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-baseline gap-3">
            <h2 className="text-sm font-semibold">Keys</h2>
            <p className="text-xs text-muted-foreground">Keys tagged with this customer, or with traffic from it this month.</p>
          </div>
          <Button size="sm" variant="outline" onClick={() => setAddOpen(true)}><Plus className="size-4" /> Add key</Button>
        </div>
        {rows.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-10 text-center text-sm text-muted-foreground">
            <p>No keys yet. Add a gateway key this customer will use.</p>
            <Button size="sm" onClick={() => setAddOpen(true)}><Plus className="size-4" /> Add key</Button>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">Key</TableHead>
                <TableHead>Models</TableHead>
                <TableHead>TPM limit</TableHead>
                <TableHead>RPM limit</TableHead>
                <TableHead>Source</TableHead>
                <TableHead className="w-32" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(({ k, tagged, traffic }) => (
                <TableRow key={k.token}>
                  <TableCell className="pl-4">
                    <p className="font-medium">{k.alias}</p>
                    <p className="font-mono text-[11px] text-muted-foreground">{k.token.slice(0, 12)}...</p>
                  </TableCell>
                  <TableCell className="max-w-56 truncate text-muted-foreground">{k.models.length ? k.models.join(", ") : "All models"}</TableCell>
                  <TableCell className="tabular-nums">{k.tpmLimit ? k.tpmLimit.toLocaleString() : <span className="text-muted-foreground">No limit</span>}</TableCell>
                  <TableCell className="tabular-nums">{k.rpmLimit ? k.rpmLimit.toLocaleString() : <span className="text-muted-foreground">No limit</span>}</TableCell>
                  <TableCell className="space-x-1">
                    {tagged && <Badge variant="secondary">Added</Badge>}
                    {traffic && <Badge variant="outline">Traffic this month</Badge>}
                  </TableCell>
                  <TableCell className="text-right">
                    {tagged && <Button size="sm" variant="ghost" onClick={() => removeKey(k)}><X className="size-4" /> Remove</Button>}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
      <Button variant="link" size="sm" className="h-auto p-0 text-xs text-muted-foreground" onClick={onBack}>Back to customers</Button>
      {c && <EditCustomerDialog open={editOpen} onOpenChange={setEditOpen} customer={c} onSaved={() => info.refetch()} />}
      {accessToken && (
        <AddKeyDialog
          open={addOpen}
          onOpenChange={setAddOpen}
          accessToken={accessToken}
          customerId={customerId}
          customerLabel={label}
          keys={keysQuery.data ?? []}
          excludeTokens={shownTokens}
          onSaved={refresh}
        />
      )}
    </div>
  );
}
