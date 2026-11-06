"use client";

import { useMemo, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Ban, Pencil, Plus, SlidersHorizontal } from "lucide-react";
import { toast } from "sonner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api } from "@/lib/http/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { uiHref } from "@/utils/uiHref";
import { customerStatus, formatUSD, useCustomerSpendByCustomer } from "../_components/customerPortalData";
import { customerLimitsOnKey, useAllKeys, useCustomerTrafficKeys, type PortalKey } from "../_components/customerKeys";
import KeyLimitDialog from "../_components/KeyLimitDialog";

const STATUS_BADGE = {
  blocked: { variant: "outline", label: "Blocked" },
  over_budget: { variant: "destructive", label: "Over budget" },
  active: { variant: "secondary", label: "Active" },
} as const;

export default function CustomerDetailPage() {
  const params = useParams<{ id: string }>();
  const customerId = decodeURIComponent(params.id);
  const { accessToken } = useAuthorized();
  const info = $api.useQuery("get", "/customer/info", { params: { query: { end_user_id: customerId } } }, { enabled: Boolean(accessToken) });
  const spendQuery = useCustomerSpendByCustomer(accessToken);
  const keysQuery = useAllKeys(accessToken);
  const trafficQuery = useCustomerTrafficKeys(accessToken, customerId);
  const [dialog, setDialog] = useState<{ open: boolean; editKey: PortalKey | null }>({ open: false, editKey: null });
  const block = $api.useMutation("post", "/customer/block", { onSuccess: () => { toast.success("Customer blocked"); info.refetch(); } });
  const unblock = $api.useMutation("post", "/customer/update", { onSuccess: () => { toast.success("Customer unblocked"); info.refetch(); } });

  const c = info.data;
  const spend = spendQuery.data?.get(customerId)?.spend ?? 0;
  const rows = useMemo(() => {
    const keys = keysQuery.data ?? [];
    const traffic = trafficQuery.data ?? new Set<string>();
    return keys
      .map((k) => ({ k, limits: customerLimitsOnKey(k, customerId), traffic: traffic.has(k.token) }))
      .filter((r) => r.traffic || r.limits.tpm !== null || r.limits.rpm !== null);
  }, [keysQuery.data, trafficQuery.data, customerId]);
  const shownTokens = useMemo(() => new Set(rows.map((r) => r.k.token)), [rows]);
  const label = c?.alias || customerId;
  const status = c ? customerStatus(c, spend) : "active";
  const max = c?.litellm_budget_table?.max_budget ?? null;
  const refresh = () => { keysQuery.refetch(); trafficQuery.refetch(); };

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
          <Button variant="outline" disabled><Pencil className="size-4" /> Edit</Button>
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
            <p className="text-xs text-muted-foreground">Keys with traffic from this customer this month, or with a limit set for this customer.</p>
          </div>
          <Button size="sm" variant="outline" onClick={() => setDialog({ open: true, editKey: null })}><Plus className="size-4" /> Add key</Button>
        </div>
        {rows.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-10 text-center text-sm text-muted-foreground">
            <p>No keys yet. Add a key to set limits for this customer.</p>
            <Button size="sm" onClick={() => setDialog({ open: true, editKey: null })}><Plus className="size-4" /> Add key</Button>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-4">Key</TableHead>
                <TableHead>TPM limit for this key</TableHead>
                <TableHead>RPM limit for this key</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="w-32" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(({ k, limits }) => (
                <TableRow key={k.token}>
                  <TableCell className="pl-4">
                    <p className="font-medium">{k.alias}</p>
                    <p className="font-mono text-[11px] text-muted-foreground">{k.token.slice(0, 12)}...</p>
                  </TableCell>
                  <TableCell className="tabular-nums">{limits.tpm !== null ? limits.tpm.toLocaleString() : <span className="text-muted-foreground">Inherits key limit{k.tpmLimit ? ` (${k.tpmLimit.toLocaleString()})` : ""}</span>}</TableCell>
                  <TableCell className="tabular-nums">{limits.rpm !== null ? limits.rpm.toLocaleString() : <span className="text-muted-foreground">Inherits key limit{k.rpmLimit ? ` (${k.rpmLimit.toLocaleString()})` : ""}</span>}</TableCell>
                  <TableCell><Badge variant={c?.blocked ? "outline" : "secondary"}>{c?.blocked ? "Blocked" : "Active"}</Badge></TableCell>
                  <TableCell className="text-right">
                    <Button size="sm" variant="outline" onClick={() => setDialog({ open: true, editKey: k })}><SlidersHorizontal className="size-4" /> Edit limits</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
      <p className="text-xs text-muted-foreground"><Link href={uiHref("customers")} className="hover:underline">Back to customers</Link></p>
      {accessToken && (
        <KeyLimitDialog
          open={dialog.open}
          onOpenChange={(open) => setDialog((d) => ({ ...d, open }))}
          accessToken={accessToken}
          customerId={customerId}
          customerLabel={label}
          keys={keysQuery.data ?? []}
          excludeTokens={shownTokens}
          editKey={dialog.editKey}
          onSaved={refresh}
        />
      )}
    </div>
  );
}
